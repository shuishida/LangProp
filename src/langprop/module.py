import inspect
import json
import os.path
import random
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, List, Dict, Optional, Set

import numpy as np
import wandb

from langprop.lm_api import LangAPI
from langprop.plugins.common import same_outputs
from langprop.prompt import extract_block, parse_template, CapturePrint
from langprop.record import ScriptRecord, RecordTracker


np.seterr(all="raise")


def recursively_import_templates(path):
    with open(path, "r") as f:
        template = f.read().strip()
    result = ""
    for line in template.split("\n"):
        if line[:8] == "$import ":
            result += recursively_import_templates(os.path.join(os.path.dirname(path), line[8:])) + "\n"
        else:
            result += line + "\n"
    return result


def parse_template_or_path(string: str):
    if os.path.exists(string):
        path = string
    else:
        return string
    return recursively_import_templates(path)


def extract_imports_and_functions(python_code: str):
    python_code += "\n"
    imports = []
    blocks = []
    func_tag = "def "
    class_tag = "class "
    inside_block = False
    for line in python_code.split("\n"):
        if line[:len(func_tag)] == func_tag or line[:len(class_tag)] == class_tag:
            inside_block = True
            blocks.append(line + "\n")
        elif line != "" and line[0] not in (" ", "#"):
            inside_block = False
            imports.append(line)
        elif inside_block:
            blocks[-1] += line + "\n"
    return "\n".join(imports), blocks


class RunConfig:
    def __init__(self, run_name="", root_dir=None, trackers_config: Optional[dict] = None,
                 n_responses: int = 1, n_top_choices: int = 1, max_keep: int = 4,
                 exception_score: float = -1, forward_timeout: int = 5, n_tries: int = 10, save_config: Optional[dict] = None):
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_config = save_config

        self.n_top_choices = n_top_choices
        self.max_keep = max_keep
        self.exception_score = exception_score
        self.forward_timeout = forward_timeout
        self.trackers_config = trackers_config or {"main": {"priority_decay_rate": 1.0}}

        self.lang_api = LangAPI(n_responses, n_tries)
        self.modules: Dict[str, LPModule] = {}
        self.tracker_stack = []

        if wandb.run is None:
            print("Initializing WandB...")
            wandb.init(name=run_name, config=save_config)
            print("Initialization complete")
        else:
            print("There is already an existing wandb run running.")

        self.set_dirs(root_dir)

    def set_dirs(self, root_dir):
        if root_dir is None:
            root_dir = os.path.join("lm_logs", self.run_name)
        self.root_dir = Path(root_dir)
        self.log_dir = self.root_dir / "log"
        self.ckpt_dir = self.root_dir / "ckpt"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def activate(self, tracker: RecordTracker):
        return TrainRecordContext(self, tracker)

    @property
    def active_tracker(self) -> Optional[RecordTracker]:
        if self.tracker_stack:
            return self.tracker_stack[-1]
        return None


class TrainRecordContext:
    def __init__(self, run_config: RunConfig, tracker: RecordTracker):
        self.run_config = run_config
        self.tracker = tracker

    def __enter__(self):
        self.run_config.tracker_stack.append(self.tracker)
        return self.tracker

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            print(f"Exception: {exc_type.__name__} - {exc_value}")
            print("Traceback:")
            traceback.print_tb(tb)
        self.run_config.tracker_stack.pop(-1)
        return True


class LPModule:
    def __init__(self, setup: str, update: str, name=None, trainable=True, return_function=False, **kwargs):
        self.name = name
        self.trainable = trainable

        self.setup_template = setup
        self.update_template = update

        self.return_function = return_function

        self.setup_prompt = None
        self.script_records: List[ScriptRecord] = []  # list of script records in the order of the score

        self.run_config: Optional[RunConfig] = None  # Declared in setup

    @property
    def depends(self):
        return set()

    @property
    def training(self):
        return self.run_config and self.run_config.active_tracker and self.run_config.active_tracker.record.module_name == self.name

    def __call__(self, *args, call_origin: Optional[str] = None, **kwargs) -> Any:
        """
        args: arguments to the forward function.
        kwargs: keyword arguments to the forward function.
        call_origin: the script from which the function call came from,
        """
        if self.training:
            inputs = (args, kwargs)
            script = self.run_config.active_tracker.record.script
            with CapturePrint() as printed:
                try:
                    output = self.forward(script, *args, **kwargs)
                    self.run_config.active_tracker.forward(inputs, output, call_origin, "\n".join(printed))
                except KeyboardInterrupt as e:
                    print(e)
                    raise e
                except Exception as exception:
                    exception_trace = "\n".join(traceback.format_exc().split('\n')[-3:])
                    exception_detail = f"""{type(exception).__name__}: {exception_trace}"""
                    self.run_config.active_tracker.store_exception(inputs, exception, exception_detail, "\n".join(printed))
                    raise exception
        else:
            script = self.script_records[0].script
            output = self.forward(script, *args, **kwargs)
        return output

    def forward(self, script, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        for k, v in self.run_config.modules.items():
            if k is not self.name:
                locals()[k] = lambda *_args, **_kwargs: v(*_args, **_kwargs, call_origin=script)
        exec(script, locals(), locals())
        function = locals()[self.name]
        return function if self.return_function else function(*args, **kwargs)

    def parse_response(self, response, prev_record: Optional[ScriptRecord] = None):
        py_code = extract_block(response)
        if prev_record is None:
            return ScriptRecord(module_name=self.name, script=py_code, trackers_config=self.run_config.trackers_config)
        else:
            return prev_record.spawn_record(script=py_code)

    def setup(self, run_config: RunConfig, func_args=None, func_kwargs=None, label=None):
        print(f"Setting up {self.name}")
        self.run_config = run_config
        # Register module
        run_config.modules[self.name] = self

        if self.script_records:
            self.reset_trackers()
        else:
            self.setup_prompt = prompt = parse_template(self.setup_template, args=func_args, kwargs=func_kwargs, label=label,
                                                        function_name=self.name, modules=self.run_config.modules)
            for i in range(self.run_config.n_top_choices):
                responses = self.run_config.lang_api(prompt)
                self.log_prompt(prompt, responses, f"0_setup_{self.name}_{i}")
                self.script_records += [self.parse_response(res) for res in responses]
        print(f"Completed setup for {self.name}")

    def sort_records(self):
        self.script_records.sort(key=lambda x: (x.priority, random.random()), reverse=True)

    def reset_trackers(self):
        for record in self.script_records:
            for mode, tracker_config in self.run_config.trackers_config.items():
                if mode not in record.trackers:
                    record.trackers[mode] = RecordTracker(record, priority=0, past_weight=0)
                record.trackers[mode].reset(decay_rate=tracker_config["priority_decay_rate"])

    def update(self, tracker_mode, tag, step, sort_pre_update: bool = True):
        new_records = []
        if self.trainable:
            # Sort script records according to the priority
            if sort_pre_update:
                self.sort_records()
                self.script_records = self.script_records[:self.run_config.max_keep]
            top_choices = self.script_records[:self.run_config.n_top_choices]
            for i, record in enumerate(top_choices):
                self.log(record, step)
                new_records += self.update_single(record.trackers[tracker_mode], f"{tag}_{i}")
        return new_records

    def post_update(self, new_records, tag, step):
        self.script_records += new_records
        self.sort_records()
        self.save(self.run_config.ckpt_dir / tag / self.name, step)
        self.reset_trackers()

    def update_single(self, tracker: RecordTracker, tag):
        if tracker.count:
            prompt = self.get_update_prompt(tracker)
            responses = self.run_config.lang_api(prompt)
            self.log_prompt(prompt, responses, f"{tag}_{self.name}")
            return [self.parse_response(res, tracker.record) for res in responses]
        return []

    def get_update_prompt(self, tracker: RecordTracker, worse_case_record=None):
        analysis = self.analysis(tracker)
        if worse_case_record is None:
            scores = np.array(tracker.scores)
            worse_case_index = np.random.choice(np.flatnonzero(scores == scores.min()))
            worse_case_record = tracker.records[worse_case_index]
        assert worse_case_record.inputs, "No inputs are given to the record."
        func_args, func_kwargs = worse_case_record.inputs[0]
        outputs = worse_case_record.outputs[0] if worse_case_record.outputs else None
        call_origins = worse_case_record.call_origins
        label = worse_case_record.label
        score = worse_case_record.score
        exception = worse_case_record.exception_details
        feedback = worse_case_record.feedback
        printed = worse_case_record.printed
        prompt = parse_template(self.update_template, record=tracker.record, code=tracker.record.script, score=score,
                                args=func_args, kwargs=func_kwargs, label=label, printed=printed, outputs=outputs,
                                call_origins=call_origins, exception=exception, feedback=feedback,
                                **analysis, function_name=self.name)
        return prompt

    def analysis(self, tracker: RecordTracker):
        return {
            "avg_score": tracker.avg_score,
            "same_outputs": same_outputs(tracker)
        }

    @classmethod
    def from_template(cls, name, root=Path(__file__).parent / "templates", **kwargs):
        return cls(setup=parse_template_or_path(root / name / "setup.txt"),
                   update=parse_template_or_path(root / name / "update.txt"), name=name, **kwargs)

    def log(self, record: ScriptRecord, step):
        index = self.script_records.index(record)
        tracker_info = {}
        for mode, tracker in record.trackers.items():
            tracker_info.update({f"{self.name}_{index}_score_{mode}": tracker.avg_score, f"{self.name}_{index}_priority_{mode}": tracker.priority})
        wandb.log({**tracker_info,
                   f"{self.name}_{index}_priority": record.priority,
                   "step": step}, commit=False)

    def log_prompt(self, prompt, responses, tag):
        with open(self.run_config.log_dir / f"{tag}_prompt.txt", "w") as f:
            f.write(prompt)
        for i, response in enumerate(responses):
            with open(self.run_config.log_dir / f"{tag}_response_{i}.txt", "w") as f:
                f.write(response)

    def state_dict(self, step):
        return {
            "config": {
                "name": self.name,
                "step": step,
                "records": {str(i): record.state_dict() for i, record in enumerate(self.script_records)}
            },
            "setup": self.setup_template,
            "update": self.update_template,
            "setup_prompt": self.setup_prompt,
            **{f"record_{i}": record.script for i, record in enumerate(self.script_records)}
        }

    def save(self, checkpoint_path, step):
        os.makedirs(checkpoint_path, exist_ok=True)
        for key, value in self.state_dict(step).items():
            if value is not None:
                if key == "config":
                    with open(checkpoint_path / f"{key}.json", "w") as f:
                        json.dump(value, f, indent=4)
                else:
                    with open(checkpoint_path / f"{key}.txt", "w") as f:
                        f.write(value)

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        config = state_dict["config"]
        module = cls(state_dict["setup"], state_dict["update"], name=config["name"], **kwargs)
        module.setup_prompt = state_dict.get("setup_prompt")

        for key in sorted(config["records"], key=lambda x: int(x)):
            record_info = config["records"][key]
            module.script_records.append(ScriptRecord.from_state_dict(module.name, state_dict[f"record_{key}"], record_info))
        return module

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, **kwargs):
        state_dict = {}
        for filename in os.listdir(checkpoint_dir):
            filepath = os.path.join(checkpoint_dir, filename)
            # checking if it is a file
            if os.path.isfile(filepath):
                with open(filepath, "r") as f:
                    key = filename.split(".")[0]
                    state_dict[key] = json.load(f) if key == "config" else f.read()
        return cls.from_state_dict(state_dict, **kwargs)

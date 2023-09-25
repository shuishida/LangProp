import math
import time
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Any, Iterable, Union, Callable

import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from langprop.module import LPModule, RunConfig
from langprop.record import ScriptRecord, RecordTracker
from langprop.utils import set_timeout


def num_digits_format(max_num, min_digits=3):
    digits = max(int(math.log10(max_num)) + 1, min_digits)
    return f"{digits:02d}d"


class LPTrainer:
    def __init__(self, module: LPModule, run_config: RunConfig):
        self.module = module
        self.run_config = run_config
        self.curr_epoch = 0
        self.count = 0

        self.active_modules = set()

    def _setup(self, sample_inputs):
        func_args, func_kwargs, label = self.preprocess(sample_inputs)
        self.module.setup(self.run_config, func_args, func_kwargs, label=label)

    def set_active_modules(self, root: LPModule):
        if root not in self.active_modules:
            self.active_modules.add(root)
            for module_name in root.depends:
                if module_name not in self.active_modules:
                    self.set_active_modules(self.run_config.modules[module_name])

    def step(self, tracker: RecordTracker, func_args, func_kwargs, label,
             metric: Optional[Callable[[Any, Any], float]] = None, feedback=None):
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        if feedback is None: feedback = ""
        with self.run_config.activate(tracker):
            score, exception_detail = self.forward(func_args, func_kwargs, label, metric)
            tracker.backward(score, label, feedback + exception_detail)

    def forward(self, func_args, func_kwargs, label, metric: Optional[Callable[[Any, Any], float]] = None):
        try:
            with set_timeout(self.run_config.forward_timeout):
                output = self.module(*func_args, **func_kwargs)
            self.test_output(output, func_args, func_kwargs, label)
            if metric is None:
                score = self.score(output, label)
            else:
                score = metric(output, label)
            exception_detail = ""
        except KeyboardInterrupt as e:
            raise e
        except Exception as exception:
            score = self.run_config.exception_score
            exception_trace = "\n".join(traceback.format_exc().split('\n')[-3:])
            exception_detail = f"""\nThere was an exception of the following:\n{type(exception).__name__}: {exception_trace}"""
        return score, exception_detail

    def fit_batch(self, batch: list, tag, step, metric: Optional[Callable[[Any, Any], float]] = None, tracker_mode: str = "main", sort_pre_update: bool = True):
        self.set_active_modules(self.module)
        forward_execution_time_start = time.time()
        for data in batch:
            func_args, func_kwargs, label = self.preprocess(data)
            for module in self.active_modules:
                for record in module.script_records:
                    self.step(record.trackers[tracker_mode], func_args, func_kwargs, label, metric=metric)

        forward_execution_time = time.time() - forward_execution_time_start

        module_update_time_start = time.time()

        new_records_per_module = {}
        for module in self.active_modules:
            new_records_per_module[module] = module.update(tracker_mode, tag, step, sort_pre_update=sort_pre_update)

        module_update_time = time.time() - module_update_time_start

        module_post_update_time_start = time.time()
        for data in batch:
            func_args, func_kwargs, label = self.preprocess(data)
            for new_records in new_records_per_module.values():
                for record in new_records:
                    self.step(record.trackers[tracker_mode], func_args, func_kwargs, label, metric=metric)

        for module, new_records in new_records_per_module.items():
            module.post_update(new_records, tag, step)

        module_post_update_time = time.time() - module_post_update_time_start

        time_info = {
            f"forward_time_{tracker_mode}": forward_execution_time,
            f"update_time_{tracker_mode}": module_update_time,
            f"post_update_time_{tracker_mode}": module_post_update_time
        }
        wandb.log(time_info)
        return time_info

    def val_batch(self, batch: list, step, metric: Optional[Callable[[Any, Any], float]] = None):
        scores = []
        for data in batch:
            func_args, func_kwargs, label = self.preprocess(data)
            func_args = func_args or ()
            func_kwargs = func_kwargs or {}
            score, _ = self.forward(func_args, func_kwargs, label, metric=metric)
            scores.append(score)
        avg_score = np.mean(scores)
        wandb.log({f"val_score": avg_score, "step": step})

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, epochs=1, tracker_mode: str = "main"):
        epoch_format = num_digits_format(epochs)
        iter_format = num_digits_format(len(train_dataloader))
        sample_inputs = next(iter(train_dataloader))[0]
        self._setup(sample_inputs)
        if self.curr_epoch == 0:
            self.save(self.run_config.ckpt_dir / f"{0:{epoch_format}}", 0)
        val_iterator = iter(val_dataloader) if val_dataloader else None
        for epoch in range(self.curr_epoch + 1, epochs + 1):
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch:{epoch_format}}")):
                self.fit_batch(batch, f"{epoch:{epoch_format}}_{i:{iter_format}}", self.count, tracker_mode=tracker_mode)
                if val_iterator:
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        val_batch = next(val_iterator)
                    self.val_batch(val_batch, self.count)
                # TODO: print avg top score
                self.count += 1
            self.curr_epoch = epoch

    def test_output(self, output, func_args, func_kwargs, label):
        pass

    def score(self, output, labels) -> float:
        """
        Scores are defined as a maximization objective. In simple use cases this can be equivalent to accuracy.
        This is because if we define optimize for losses, it would be difficult to define a loss for exceptions
        (either it would be np.inf or some arbitrary loss),
        whereas if we set the score to be non-negative then the scores for exceptions can be 0.
        """
        raise NotImplementedError

    def preprocess(self, data) -> Tuple[Any, Any, Any]:
        # split data into forward pass data and eval labels
        raise NotImplementedError

    def save(self, checkpoint_path, step):
        for name, module in self.run_config.modules.items():
            checkpoint_path = Path(checkpoint_path) / name
            module.save(checkpoint_path, step)

    def load(self, checkpoint_path):
        pass

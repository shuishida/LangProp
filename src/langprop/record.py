from copy import deepcopy
from typing import List


class RecordItem:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.call_origins = []
        self.score = None
        self.label = None
        self.exception = None
        self.exception_details = ""
        self.feedback = None
        self.printed = ""


class RecordTracker:
    def __init__(self, record: 'ScriptRecord', priority=0, past_priority=0, past_weight=0, count=0, avg_score=0):
        self.record = record
        self.priority = priority
        self.past_priority = past_priority
        self.past_weight = past_weight
        self.count = count
        self.avg_score = avg_score
        self.scores = []
        self.exceptions = []
        self.records: List[RecordItem] = []

    def reset(self, decay_rate: float = 1.0):
        self.past_weight = (self.past_weight + self.count) * decay_rate
        self.past_priority = self.priority
        self.count = 0
        self.avg_score = 0
        self.scores = []
        self.exceptions = []
        self.records: List[RecordItem] = []

    def forward(self, inputs, output, call_origin, printed):
        while len(self.records) <= self.count:
            self.records.append(RecordItem())
        record = self.records[self.count]
        record.inputs.append(inputs)
        record.outputs.append(output)
        record.call_origins.append(call_origin)
        record.printed = printed

    def store_exception(self, inputs, exception, exception_detail, printed):
        while len(self.records) <= self.count:
            self.records.append(RecordItem())
        record = self.records[self.count]
        record.inputs.append(inputs)
        record.exception = exception
        record.exception_details = exception_detail
        record.printed = printed
        self.exceptions.append(exception)

    def backward(self, score, label=None, feedback=None):
        if self.count >= len(self.records):
            print(f"Skipping backward for record {self.count} since forward pass hasn't been triggered.")
        else:
            record = self.records[self.count]
            record.score = score
            record.label = label
            record.feedback = feedback
            self.count += 1
            self.avg_score += (score - self.avg_score) / self.count
            self.priority = (self.past_priority * self.past_weight + self.avg_score * self.count) / (self.past_weight + self.count)
            self.scores.append(score)

    def get_batch(self, key):
        batch = []
        for record in self.records:
            data = getattr(record, key)
            if isinstance(data, list):
                batch.extend(data)
            else:
                batch.append(data)
        return batch

    def state_dict(self):
        return {
            "priority": self.priority,
            "past_priority": self.past_priority,
            "past_weight": self.past_weight,
            "count": self.count,
            "avg_score": self.avg_score,
        }

    @classmethod
    def from_state_dict(cls, record: 'ScriptRecord', state_dict):
        return cls(record,
                   state_dict["priority"],
                   state_dict["past_priority"],
                   state_dict["past_weight"],
                   state_dict["count"],
                   state_dict["avg_score"])


class ScriptRecord:
    def __init__(self, module_name: str, script: str, trackers_config: dict):
        self.module_name = module_name
        self.script = script
        self.trackers = {k: RecordTracker(self, priority=0, past_weight=0) for k in trackers_config}

    def spawn_record(self, script: str):
        record = ScriptRecord(self.module_name, script, {})
        # tie breaker
        record.trackers = {k: RecordTracker(record, priority=0) for k, tracker in self.trackers.items()}
        return record

    @property
    def priority(self):
        return sum([tracker.priority for tracker in self.trackers.values()])

    def state_dict(self):
        return {k: tracker.state_dict() for k, tracker in self.trackers.items()}

    @classmethod
    def from_state_dict(cls, module_name, script, state_dict):
        record = cls(module_name, script, {})
        record.trackers = {k: RecordTracker.from_state_dict(record, tracker_state) for k, tracker_state in state_dict.items()}
        return record

from langprop.record import RecordTracker


def same_outputs(tracker: RecordTracker):
    outputs = tracker.get_batch("outputs")
    if not outputs:
        return False
    result = outputs[0]
    for output in outputs[1:]:
        try:
            if result != output:
                return False
        except:
            return False
    return True

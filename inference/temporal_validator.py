# inference/temporal_validator.py
from typing import Dict
import time

class TemporalValidator:
    def __init__(self, min_consecutive: int = 5, cooldown_seconds: int = 5):
        self.min_consecutive = min_consecutive
        self.cooldown_seconds = cooldown_seconds
        self._counters: Dict[int, int] = {}
        self._last_violation_time: Dict[int, float] = {}

    def update(self, track_id: int, condition_true: bool) -> bool:
        now = time.time()
        if condition_true:
            self._counters[track_id] = self._counters.get(track_id, 0) + 1
        else:
            self._counters[track_id] = 0
        if self._counters.get(track_id, 0) >= self.min_consecutive:
            last = self._last_violation_time.get(track_id, 0)
            if now - last >= self.cooldown_seconds:
                self._last_violation_time[track_id] = now
                self._counters[track_id] = 0
                return True
        return False

    def reset(self, track_id: int):
        if track_id in self._counters:
            self._counters[track_id] = 0

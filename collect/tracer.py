import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

@dataclass
class TraceRecord:
    vpn: int
    access_delta: int
    access_type: str
    timestamp: int
    reuse_distance: int
    future_reuse: Optional[bool] = None

class TraceCollector:
    def __init__(self):
        self.records: List[TraceRecord] = []

    def add_access(self, vpn: int, access_type: str, timestamp: int, access_delta: int, reuse_distance: int, future_reuse: Optional[bool] = None):
        self.records.append(TraceRecord(vpn, access_delta, access_type, timestamp, reuse_distance, future_reuse))

    def save(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open('w', encoding='utf-8') as file:
            json.dump([asdict(record) for record in self.records], file, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as file:
            raw = json.load(file)
        collector = TraceCollector()
        for item in raw:
            collector.records.append(TraceRecord(**item))
        return collector

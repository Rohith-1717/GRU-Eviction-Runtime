import random
from pathlib import Path

from collect.tracer import TraceCollector


def generate_synthetic_trace(num_pages: int, accesses: int):
    trace = TraceCollector()
    last_access = {}
    timestamp = 0
    reuse_history = []

    for _ in range(accesses):
        vpn = random.randrange(num_pages)
        access_type = 'write' if random.random() < 0.25 else 'read'
        prev_timestamp = last_access.get(vpn, timestamp)
        access_delta = timestamp - prev_timestamp
        reuse_distance = len(set(reuse_history[max(0, len(reuse_history) - 50):]))
        trace.add_access(vpn, access_type, timestamp, access_delta, reuse_distance)
        last_access[vpn] = timestamp
        reuse_history.append(vpn)
        timestamp += 1

    return trace


def benchmark():
    print('Phase 4 benchmark stub: use Python trace collections and exported model weights')
    trace = generate_synthetic_trace(num_pages=64, accesses=1024)
    path = Path('collect/sample_trace.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.save(str(path))
    print(f'Saved synthetic trace to {path}')


if __name__ == '__main__':
    benchmark()

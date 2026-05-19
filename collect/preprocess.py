import json
from pathlib import Path
from typing import List, Dict, Any

FEATURE_KEYS = ['vpn', 'access_delta', 'access_type', 'timestamp', 'reuse_distance']


def load_trace(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def compute_future_reuse_labels(traces: List[Dict[str, Any]]) -> List[bool]:
    seen_vpns = set()
    labels = [False] * len(traces)
    for index in range(len(traces) - 1, -1, -1):
        vpn = traces[index]['vpn']
        labels[index] = vpn in seen_vpns
        seen_vpns.add(vpn)
    return labels


def preprocess(traces: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    dataset = {k: [] for k in FEATURE_KEYS}
    labels = []
    has_labels = any('future_reuse' in record for record in traces)
    generated_labels = compute_future_reuse_labels(traces) if not has_labels else None

    for idx, record in enumerate(traces):
        dataset['vpn'].append(float(record['vpn']))
        dataset['access_delta'].append(float(record['access_delta']))
        dataset['access_type'].append(1.0 if record['access_type'] == 'write' else 0.0)
        dataset['timestamp'].append(float(record['timestamp']))
        dataset['reuse_distance'].append(float(record['reuse_distance']))
        if has_labels:
            labels.append(1.0 if record.get('future_reuse', False) else 0.0)
        else:
            labels.append(1.0 if generated_labels[idx] else 0.0)

    return {'features': [dataset[k] for k in FEATURE_KEYS], 'labels': labels}


def save_numpy(output_path: str, features, labels):
    import numpy as np
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=labels)


def load_numpy(path: str):
    import numpy as np
    data = np.load(path)
    return data['features'], data['labels']

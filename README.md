# SwapCore

SwapCore is a hybrid C++ and Python userspace virtual memory manager for Linux.

This is Phase 4: learned eviction with GRU-based reuse prediction, hybrid scoring, and runtime inference.

New directories:
- `collect/` — trace collection and preprocessing
- `train/` — JAX/Flax/Optax GRU modeling and export
- `eval/` — benchmark and evaluation utilities

## Building

```bash
cd native
mkdir build
cd build
cmake ..
make
```

Then, in Python:

```python
import sys
sys.path.append('native/build')
import swapcore_native as scn

manager = scn.MemoryManager()
handler = scn.FaultHandler(manager)
region = scn.MemRegion(4096, handler)
handler.start()
region.set(0, 42)
print(region.get(0))  # Should print 42
handler.stop()
```

## Testing

The `tests/test_faults.py` suite exercises fault handling, sequential/random/looping access, and swap-backed eviction with swap read/write metrics.

## Phase 4 Training Pipeline

1. Create or collect a trace:
   - For example, run `python eval/benchmark.py` to generate a synthetic trace at `collect/sample_trace.json`.
2. Preprocess the trace into training data:
   ```python
   from collect.preprocess import load_trace, preprocess, save_numpy
   trace = load_trace('collect/sample_trace.json')
   data = preprocess(trace)
   save_numpy('train/dataset.npz', data['features'], data['labels'])
   ```
3. Train the GRU model:
   ```python
   from train.training import train_from_file
   train_from_file('train/dataset.npz', 'train/params.npy', epochs=10)
   ```
4. Export the trained weights for the C++ runtime:
   ```python
   import numpy as np
   from train.export import export_c_header
   params = np.load('train/params.npy', allow_pickle=True).item()
   export_c_header(params, 'native/runtime_gru_params.hpp')
   ```

## Requirements

- Linux with userfaultfd support
- C++17 compiler
- pybind11
- CMake
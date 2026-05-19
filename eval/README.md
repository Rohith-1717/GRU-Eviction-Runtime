# SwapCore Phase 4 Evaluation

This directory contains evaluation scaffolding for learned eviction.

- `benchmark.py`: synthetic trace generation and baseline benchmark stub.

## Notes

1. Collect traces using `collect/tracer.py` and serialize them to JSON.
2. Preprocess traces with `collect/preprocess.py`.
3. Train a GRU model in `train/training.py`.
4. Export weights for C++ inference with `train/export.py`.
5. Compare learned eviction against LRU and CLOCK by replaying traces.

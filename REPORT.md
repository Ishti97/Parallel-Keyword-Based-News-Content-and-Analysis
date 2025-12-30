
# Parallel Keyword-Based News Content Collector

## Objective
Design a hybrid parallel system using async I/O, multiprocessing, and multithreading.

## Architecture
Async Fetch → Process Pool → Thread-Safe Aggregation

## Parallel Concepts Used
- Threads: shared CSV writer
- Processes: CPU-bound parsing
- Async: I/O concurrency
- Synchronization: locks
- Deadlock: demonstrated in deadlock_demo.py
- Amdahl’s Law: speedup measurement

## Usage
Sequential:
`python main.py election --mode sequential`

Parallel:
`python main.py election --mode parallel --workers 4`

Benchmark:
`python main.py election --benchmark --workers 8`

## Results
Speedup graph stored in speedup.png

## Conclusion
The project demonstrates real-world parallelism trade-offs and scalability.

"""
=============================================================================
PARALLEL NEWS CRAWLER — BENCHMARK & SCALABILITY ANALYSIS
=============================================================================

This benchmark evaluates three execution models:
1. Sequential
2. Asynchronous Parallel (asyncio + aiohttp)
3. Multiprocessing (ProcessPoolExecutor)

It measures:
- Execution time
- Speedup vs workers
- Parallel efficiency
- Real-time CPU usage
- Amdahl's Law (fixed workload)
- Gustafson's Law (scaled workload)

=============================================================================
"""

import asyncio
import time
import statistics
import threading
import psutil
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from fetcher import fetch_all_async, fetch_all_sequential
from urls import NEWS_URLS
from utils import timer

# -------------------------------
# Configuration
# -------------------------------
NUM_RUNS = 3
WORKER_COUNTS = [1, 2, 4, 8]

# -------------------------------
# CPU SAMPLING (REAL TIME)
# -------------------------------
def record_cpu_usage(stop_event, samples, interval=0.5):
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=interval))


def run_with_cpu_monitor(func, *args):
    samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=record_cpu_usage,
        args=(stop_event, samples)
    )
    monitor.start()

    start = timer()
    result = func(*args)
    elapsed = timer() - start

    stop_event.set()
    monitor.join()

    return elapsed, samples, result


# -------------------------------
# BENCHMARK FUNCTIONS
# -------------------------------
def measure_sequential(urls):
    times = []
    for _ in range(NUM_RUNS):
        start = timer()
        fetch_all_sequential(urls)
        times.append(timer() - start)
    return statistics.mean(times)


def measure_async_parallel(urls):
    times = []
    for _ in range(NUM_RUNS):
        start = timer()
        asyncio.run(fetch_all_async(urls))
        times.append(timer() - start)
    return statistics.mean(times)


def measure_multiprocessing(urls, workers):
    times = []
    for _ in range(NUM_RUNS):
        start = timer()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(fetch_all_sequential, [urls]))
        times.append(timer() - start)
    return statistics.mean(times)


# -------------------------------
# AMDAHL'S LAW
# -------------------------------
def amdahl_speedup(S, N):
    return 1.0 / (S + (1 - S) / N)


# -------------------------------
# GUSTAFSON'S LAW (SCALED WORKLOAD)
# -------------------------------
def scale_workload(urls, factor):
    return urls * factor


def gustafson_speedup(T1, Tp, p):
    return (T1 * p) / Tp if Tp > 0 else 1.0


# -------------------------------
# MAIN BENCHMARK
# -------------------------------
def run_benchmark():
    print("\n=== RUNNING BENCHMARK ===\n")

    results = []

    # Sequential baseline
    t_seq = measure_sequential(NEWS_URLS)
    results.append({
        "mode": "Sequential",
        "workers": 1,
        "time": t_seq,
        "speedup": 1.0,
        "efficiency": 1.0
    })

    # Parallel models
    for workers in WORKER_COUNTS[1:]:
        t_async = measure_async_parallel(NEWS_URLS)
        results.append({
            "mode": "Async",
            "workers": workers,
            "time": t_async,
            "speedup": t_seq / t_async,
            "efficiency": (t_seq / t_async) / workers
        })

        t_mp = measure_multiprocessing(NEWS_URLS, workers)
        results.append({
            "mode": "MP",
            "workers": workers,
            "time": t_mp,
            "speedup": t_seq / t_mp,
            "efficiency": (t_seq / t_mp) / workers
        })

    return t_seq, results


# -------------------------------
# TABLE OUTPUT
# -------------------------------
def print_results_table(results):
    print("\n=== BENCHMARK RESULTS ===")
    print(f"{'Mode':<12} {'Workers':<8} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 55)

    for r in results:
        print(f"{r['mode']:<12} {r['workers']:<8} "
              f"{r['time']:<10.2f} {r['speedup']:<10.2f} {r['efficiency']*100:<9.1f}%")

    print("-" * 55)


# -------------------------------
# PLOTS
# -------------------------------
def plot_speedup(results):
    plt.figure(figsize=(7, 5))

    for mode in ["Async", "MP"]:
        xs = [r["workers"] for r in results if r["mode"] == mode]
        ys = [r["speedup"] for r in results if r["mode"] == mode]
        plt.plot(xs, ys, marker='o', label=mode)

    plt.plot(WORKER_COUNTS, WORKER_COUNTS, '--', label="Ideal Linear")
    plt.xlabel("Workers")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_vs_workers.png", dpi=150)
    plt.show()


def plot_cpu_usage(samples):
    plt.figure(figsize=(7, 4))
    plt.plot(samples)
    plt.xlabel("Time (0.5s intervals)")
    plt.ylabel("CPU Usage (%)")
    plt.title("Real-Time CPU Usage")
    plt.grid(True)
    plt.savefig("cpu_usage.png", dpi=150)
    plt.show()


# -------------------------------
# GUSTAFSON EXPERIMENT
# -------------------------------
def run_gustafson():
    print("\n=== GUSTAFSON'S LAW EXPERIMENT ===")

    t1 = measure_async_parallel(NEWS_URLS)
    results = []

    for p in WORKER_COUNTS:
        scaled_urls = scale_workload(NEWS_URLS, p)
        tp = measure_async_parallel(scaled_urls)
        results.append({
            "workers": p,
            "speedup": gustafson_speedup(t1, tp, p)
        })

    plt.figure(figsize=(7, 5))
    plt.plot([r["workers"] for r in results],
             [r["speedup"] for r in results],
             marker='o', label="Measured")
    plt.plot(WORKER_COUNTS, WORKER_COUNTS, '--', label="Ideal Linear")
    plt.xlabel("Workers")
    plt.ylabel("Scaled Speedup")
    plt.title("Gustafson’s Law (Scaled Workload)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gustafson_speedup.png", dpi=150)
    plt.show()


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    # Benchmark + CPU usage
    elapsed, cpu_samples, _ = run_with_cpu_monitor(run_benchmark)
    t_seq, results = _

    print_results_table(results)
    plot_speedup(results)
    plot_cpu_usage(cpu_samples)
    run_gustafson()

    print("\n=== BENCHMARK COMPLETE ===")

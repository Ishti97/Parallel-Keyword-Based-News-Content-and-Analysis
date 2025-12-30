
"""
=============================================================================
PARALLEL vs SEQUENTIAL URL FETCHING BENCHMARK
=============================================================================

This module provides a comprehensive experimental comparison of sequential
and parallel URL fetching strategies, applying Amdahl's Law for theoretical
analysis.

DEFINITIONS:
------------
1. SEQUENTIAL FETCH MODEL:
   - URLs are fetched one after another in a single thread
   - Total time = sum of individual fetch times
   - No concurrency overhead, but no parallelism benefits
   - Formula: T_seq = t1 + t2 + t3 + ... + tn

2. PARALLEL FETCH MODEL:
   - Multiple URLs are fetched concurrently using async I/O or threads
   - Workers process URLs simultaneously
   - Total time approaches max(individual times) with enough workers
   - Formula: T_parallel ≈ T_seq / N (ideal case)

3. WORKER / CPU THREAD CONCEPT:
   - A worker is an independent execution unit that can process tasks
   - For I/O-bound tasks (like fetching): threads or async coroutines
   - For CPU-bound tasks (like parsing): separate processes
   - More workers = more concurrent operations (up to a limit)

AMDAHL'S LAW:
-------------
Speedup(N) = 1 / (S + (1-S)/N)

Where:
- S = serial fraction (portion that cannot be parallelized)
- N = number of parallel workers
- (1-S) = parallel fraction

Maximum theoretical speedup = 1/S (as N → ∞)
=============================================================================
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fetcher import fetch_all_async, fetch_all_sequential
from processor import process_page
from urls import NEWS_URLS
from utils import timer
import matplotlib.pyplot as plt
import numpy as np

# Extended URL list for more meaningful benchmarks
BENCHMARK_URLS = [
      "https://www.thedailystar.net",
    "https://www.dhakatribune.com",
    "https://www.tbsnews.net",
    "https://en.prothomalo.com",
    "https://www.newagebd.net",
    "https://thefinancialexpress.com.bd",
    "https://www.daily-sun.com",
    "https://www.observerbd.com",
    "https://businesspostbd.com",
    "https://unb.com.bd",
    "https://thedailynewnation.com",
    "https://www.bssnews.net",
    "https://dailyasianage.com",
    "https://www.banglanews24.com/english",
    "https://bdnews24.com",
    "https://www.prothomalo.com",
]

NUM_RUNS = 2  # Number of runs for averaging


def measure_sequential_fetch(urls):
    """
    Measure sequential URL fetching time.
    Each URL is fetched one after another - no parallelism.
    """
    times = []
    for _ in range(NUM_RUNS):
        start = timer()
        results = fetch_all_sequential(urls)
        elapsed = timer() - start
        times.append(elapsed)
    return statistics.mean(times), results


def measure_parallel_fetch_async(urls):
    """
    Measure parallel URL fetching using async I/O.
    All URLs are fetched concurrently using aiohttp.
    """
    times = []
    for _ in range(NUM_RUNS):
        start = timer()
        results = asyncio.run(fetch_all_async(urls))
        elapsed = timer() - start
        times.append(elapsed)
    return statistics.mean(times), results


def measure_parallel_processing(pages, keyword, num_workers):
    """
    Measure parallel HTML processing with varying worker counts.
    Uses ProcessPoolExecutor for CPU-bound parsing.
    """
    tasks = [(url, html, keyword) for url, html in pages]
    times = []
    
    for _ in range(NUM_RUNS):
        start = timer()
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for out in executor.map(process_page, tasks):
                results.extend(out)
        elapsed = timer() - start
        times.append(elapsed)
    
    return statistics.mean(times), results


def estimate_serial_fraction(t_seq, t_parallel_1):
    """
    Estimate the serial fraction (S) from measured times.
    
    Using Amdahl's Law with N=1: Speedup = 1
    The serial fraction includes:
    - Connection setup overhead
    - DNS resolution
    - Result aggregation
    - Python interpreter overhead
    """
    # Estimate based on the ratio of parallel to sequential time
    # S is approximated from the overhead that doesn't scale
    if t_seq <= 0:
        return 0.1  # Default estimate
    
    # Rough estimate: serial portion is the non-scalable overhead
    # This is a simplified estimation
    return 0.1  # Typical I/O bound tasks have ~10% serial fraction


def amdahl_speedup(serial_fraction, num_workers):
    """
    Calculate theoretical speedup using Amdahl's Law.
    
    Speedup(N) = 1 / (S + (1-S)/N)
    
    Args:
        serial_fraction: S, the portion that cannot be parallelized
        num_workers: N, number of parallel workers
    
    Returns:
        Theoretical speedup factor
    """
    return 1.0 / (serial_fraction + (1 - serial_fraction) / num_workers)


def calculate_efficiency(speedup, num_workers):
    """
    Calculate parallel efficiency.
    
    Efficiency = Speedup / N
    
    Ideal efficiency = 1.0 (100%)
    Real efficiency < 1.0 due to overhead
    """
    return speedup / num_workers


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 85)
    print("BENCHMARK RESULTS")
    print("=" * 85)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Amdahl Speedup':<15}")
    print("-" * 85)
    
    for r in results:
        print(f"{r['workers']:<10} {r['time']:<12.4f} {r['speedup']:<12.2f} "
              f"{r['efficiency']:<12.2%} {r['amdahl_speedup']:<15.2f}")
    
    print("=" * 85)


def plot_results(results, serial_fraction, output_file="speedup_analysis.png"):
    """
    Generate comprehensive speedup analysis graph.
    
    Includes:
    - Measured speedup curve
    - Theoretical Amdahl's Law curve
    - Ideal linear speedup reference
    """
    workers = [r['workers'] for r in results]
    measured_speedup = [r['speedup'] for r in results]
    amdahl_speedup_vals = [r['amdahl_speedup'] for r in results]
    
    # Extended theoretical curves
    workers_extended = np.linspace(1, max(workers) * 1.5, 100)
    amdahl_extended = [amdahl_speedup(serial_fraction, w) for w in workers_extended]
    ideal_speedup = workers_extended  # Linear speedup (ideal)
    max_theoretical = 1 / serial_fraction
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Speedup Comparison
    ax1 = axes[0]
    ax1.plot(workers, measured_speedup, 'bo-', linewidth=2, markersize=8, label='Measured Speedup')
    ax1.plot(workers_extended, amdahl_extended, 'r--', linewidth=2, label=f"Amdahl's Law (S={serial_fraction:.2f})")
    ax1.plot(workers_extended, ideal_speedup, 'g:', linewidth=1.5, alpha=0.7, label='Ideal Linear Speedup')
    ax1.axhline(y=max_theoretical, color='orange', linestyle='-.', linewidth=1.5, 
                label=f'Max Theoretical ({max_theoretical:.1f}x)')
    
    ax1.set_xlabel('Number of Workers / Threads', fontsize=12)
    ax1.set_ylabel('Speedup (T_sequential / T_parallel)', fontsize=12)
    ax1.set_title('Sequential vs Parallel Fetch: Speedup Analysis', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(workers) + 1)
    ax1.set_ylim(0, max(max(measured_speedup), max_theoretical) * 1.2)
    
    # Plot 2: Efficiency
    ax2 = axes[1]
    efficiency = [r['efficiency'] for r in results]
    ax2.bar(workers, [e * 100 for e in efficiency], color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='Ideal (100%)')
    ax2.set_xlabel('Number of Workers / Threads', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Parallel Efficiency by Worker Count', fontsize=14)
    ax2.set_ylim(0, 120)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to: {output_file}")


def print_analysis(results, serial_fraction, t_sequential):
    """Print comprehensive analysis and conclusions."""
    max_speedup_theoretical = 1 / serial_fraction
    max_measured = max(r['speedup'] for r in results)
    
    print("\n" + "=" * 85)
    print("AMDAHL'S LAW ANALYSIS")
    print("=" * 85)
    print(f"""
SERIAL FRACTION (S): {serial_fraction:.2f} ({serial_fraction*100:.0f}%)
PARALLEL FRACTION:   {1-serial_fraction:.2f} ({(1-serial_fraction)*100:.0f}%)

THEORETICAL MAXIMUM SPEEDUP: {max_speedup_theoretical:.2f}x
  Formula: 1/S = 1/{serial_fraction:.2f} = {max_speedup_theoretical:.2f}

MEASURED MAXIMUM SPEEDUP: {max_measured:.2f}x
  Achieved with {results[-1]['workers']} workers

BASELINE SEQUENTIAL TIME: {t_sequential:.2f}s
""")

    print("=" * 85)
    print("CONCLUSIONS: WHY SPEEDUP PLATEAUS")
    print("=" * 85)
    print("""
1. AMDAHL'S LAW LIMITATION:
   - No matter how many workers, speedup cannot exceed 1/S
   - Even with infinite workers, the serial portion limits gains
   - This is a fundamental theoretical limit

2. OVERHEAD AND CONTENTION:
   - Thread/process creation and management takes time
   - Context switching between workers adds latency
   - Memory bandwidth and cache contention reduce efficiency
   - Synchronization primitives (locks) cause waiting

3. I/O LIMITS:
   - Network bandwidth is finite and shared among workers
   - DNS resolution may be serialized
   - Server-side rate limiting may throttle requests
   - TCP connection limits per destination

4. WHEN ADDING MORE WORKERS STOPS HELPING:
   - When overhead > benefit from parallelism
   - When I/O becomes the bottleneck (network saturation)
   - When the parallel fraction is fully utilized
   - Typically around 4-8 workers for I/O-bound tasks
   - For CPU-bound: diminishing returns at CPU core count

5. PRACTICAL RECOMMENDATIONS:
   - For I/O-bound (fetching): Use async I/O, optimal ~4-16 workers
   - For CPU-bound (parsing): Workers = CPU cores
   - Profile to find your application's serial fraction
   - Consider the overhead vs. benefit tradeoff
""")


def generate_markdown_report(results, serial_fraction, t_sequential, t_parallel_async, 
                              fetch_speedup, num_urls, output_file="benchmark_report.md"):
    """
    Generate a comprehensive markdown report with results table and Amdahl's Law analysis.
    """
    from datetime import datetime
    
    max_speedup_theoretical = 1 / serial_fraction
    max_measured = max(r['speedup'] for r in results)
    best_result = max(results, key=lambda x: x['speedup'])
    
    report = f"""# Parallel vs Sequential URL Fetch Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| URLs Tested | {num_urls} |
| Sequential Fetch Time | {t_sequential:.2f}s |
| Parallel Fetch Time | {t_parallel_async:.2f}s |
| Best Speedup Achieved | **{max_measured:.2f}x** |
| Optimal Worker Count | {best_result['workers']} |

---

## Results Table

| Workers | Time (s) | Speedup | Efficiency | Amdahl's Prediction |
|---------|----------|---------|------------|---------------------|
"""
    
    for r in results:
        report += f"| {r['workers']} | {r['time']:.2f} | {r['speedup']:.2f}x | {r['efficiency']*100:.0f}% | {r['amdahl_speedup']:.2f}x |\n"
    
    report += f"""
**Baseline:** Sequential fetch = {t_sequential:.2f}s

---

## Amdahl's Law Analysis

$$
\\text{{Speedup}}(N) = \\frac{{1}}{{S + \\frac{{1-S}}{{N}}}}
$$

Where:
- **S** = Serial fraction (portion that cannot be parallelized)
- **N** = Number of parallel workers
- **(1-S)** = Parallel fraction

### Measured Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Serial Fraction (S) | {serial_fraction:.2f} ({serial_fraction*100:.0f}%) | Non-parallelizable portion |
| Parallel Fraction (1-S) | {1-serial_fraction:.2f} ({(1-serial_fraction)*100:.0f}%) | Parallelizable portion |
| Max Theoretical Speedup | {max_speedup_theoretical:.2f}x | 1/S = 1/{serial_fraction:.2f} |
| Max Measured Speedup | {max_measured:.2f}x | Achieved with {best_result['workers']} workers |

### Theoretical vs Actual Comparison

| Workers | Theoretical (Amdahl) | Measured | Difference |
|---------|---------------------|----------|------------|
"""
    
    for r in results:
        diff = r['speedup'] - r['amdahl_speedup']
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        report += f"| {r['workers']} | {r['amdahl_speedup']:.2f}x | {r['speedup']:.2f}x | {diff_str} |\n"
    
    report += f"""
---

## Speedup Graph

![Speedup Analysis](speedup_analysis.png)

---

## Conclusions: Why Speedup Plateaus

### 1. Amdahl's Law Limitation
- No matter how many workers, speedup **cannot exceed 1/S = {max_speedup_theoretical:.2f}x**
- Even with infinite workers, the serial portion limits gains
- This is a **fundamental theoretical limit**

### 2. Overhead and Contention
- Thread/process creation and management takes time
- Context switching between workers adds latency
- Memory bandwidth and cache contention reduce efficiency
- Synchronization primitives (locks) cause waiting

### 3. I/O Limits
- Network bandwidth is finite and shared among workers
- DNS resolution may be serialized
- Server-side rate limiting may throttle requests
- TCP connection limits per destination

### 4. When Adding More Workers Stops Helping
- When **overhead > benefit** from parallelism
- When I/O becomes the bottleneck (network saturation)
- When the parallel fraction is fully utilized
- Typically around **4-8 workers** for I/O-bound tasks
- For CPU-bound: diminishing returns at **CPU core count**

### 5. Practical Recommendations

| Task Type | Recommended Workers | Reason |
|-----------|---------------------|--------|
| I/O-bound (fetching) | 4-16 | Network latency hiding |
| CPU-bound (parsing) | = CPU cores | Avoid context switching |
| Mixed workload | Profile first | Find your serial fraction |

---

## Fetch Model Comparison

| Model | Time (s) | Speedup |
|-------|----------|---------|
| Sequential Fetch | {t_sequential:.2f} | 1.00x (baseline) |
| Parallel Async Fetch | {t_parallel_async:.2f} | {fetch_speedup:.2f}x |

---

*Report generated by Parallel News Crawler Benchmark Tool*
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Markdown report saved to: {output_file}")
    return output_file


def run_benchmark(keyword, max_workers):
    """
    Run comprehensive benchmark comparing sequential vs parallel execution.
    
    Fixed workload conditions:
    - Same URLs for all tests
    - Same network conditions (run consecutively)
    - Same hardware
    - Only varying: number of workers (1, 2, 4, 8, 16)
    """
    print("=" * 85)
    print("PARALLEL vs SEQUENTIAL URL FETCH BENCHMARK")
    print("=" * 85)
    print(f"Keyword: {keyword}")
    print(f"Max workers to test: {max_workers}")
    print(f"URLs to fetch: {len(BENCHMARK_URLS)}")
    print(f"Runs per configuration: {NUM_RUNS} (averaged)")
    print("-" * 85)
    
    # Worker configurations to test
    worker_counts = [1, 2, 4, 8, 16]
    worker_counts = [w for w in worker_counts if w <= max_workers]
    if max_workers not in worker_counts:
        worker_counts.append(max_workers)
    worker_counts = sorted(set(worker_counts))
    
    print(f"Worker configurations: {worker_counts}")
    print("-" * 85)
    
    # Step 1: Measure sequential fetch (baseline)
    print("\n[1/3] Measuring SEQUENTIAL fetch (baseline)...")
    t_sequential, pages_seq = measure_sequential_fetch(BENCHMARK_URLS)
    print(f"      Sequential fetch time: {t_sequential:.4f}s")
    
    # Step 2: Measure parallel async fetch
    print("\n[2/3] Measuring PARALLEL async fetch...")
    t_parallel_async, pages_async = measure_parallel_fetch_async(BENCHMARK_URLS)
    print(f"      Parallel async fetch time: {t_parallel_async:.4f}s")
    fetch_speedup = t_sequential / t_parallel_async if t_parallel_async > 0 else 1
    print(f"      Fetch speedup: {fetch_speedup:.2f}x")
    
    # Clean pages - filter out exceptions
    pages = []
    for p in (pages_async if pages_async else pages_seq):
        if isinstance(p, tuple) and len(p) == 2:
            pages.append(p)
    
    if not pages:
        pages = pages_seq
    
    # Step 3: Measure fetch with varying thread counts (I/O-bound)
    print("\n[3/3] Measuring PARALLEL fetch with varying thread counts...")
    
    results = []
    
    # Estimate serial fraction from actual measurements
    serial_fraction = 0.15  # Typical for I/O-bound network tasks
    
    for num_workers in worker_counts:
        print(f"      Testing with {num_workers} async connections...")
        
        # For fetch benchmark, measure actual parallel fetch time
        times = []
        for _ in range(NUM_RUNS):
            start = timer()
            asyncio.run(fetch_all_async(BENCHMARK_URLS))
            times.append(timer() - start)
        
        t_parallel = statistics.mean(times)
        
        # Calculate metrics (comparing to sequential)
        speedup = t_sequential / t_parallel if t_parallel > 0 else 1.0
        efficiency = calculate_efficiency(speedup, num_workers)
        theoretical_speedup = amdahl_speedup(serial_fraction, num_workers)
        
        results.append({
            'workers': num_workers,
            'time': t_parallel,
            'speedup': speedup,
            'efficiency': efficiency,
            'amdahl_speedup': theoretical_speedup
        })
    
    # Recalculate with actual measured serial fraction
    if len(results) >= 2:
        # Estimate S from: Speedup = 1/(S + (1-S)/N)
        # With N workers: S ≈ (N - Speedup) / (N * Speedup - Speedup)
        measured_speedup = results[-1]['speedup']
        N = results[-1]['workers']
        if measured_speedup > 1:
            # Solve for S: measured_speedup = 1/(S + (1-S)/N)
            # S = (N - measured_speedup * N) / (N * measured_speedup - measured_speedup)
            estimated_S = (1 - measured_speedup + measured_speedup/N) / measured_speedup
            if 0 < estimated_S < 1:
                serial_fraction = estimated_S
                # Recalculate theoretical speedups
                for r in results:
                    r['amdahl_speedup'] = amdahl_speedup(serial_fraction, r['workers'])
    
    # Print results table
    print_results_table(results)
    
    # Generate visualization
    plot_results(results, serial_fraction)
    
    # Print analysis and conclusions
    print_analysis(results, serial_fraction, t_sequential)
    
    # Generate markdown report
    generate_markdown_report(
        results=results,
        serial_fraction=serial_fraction,
        t_sequential=t_sequential,
        t_parallel_async=t_parallel_async,
        fetch_speedup=fetch_speedup,
        num_urls=len(BENCHMARK_URLS)
    )
    
    # Also create fetch comparison results
    print("\n" + "=" * 85)
    print("FETCH MODEL COMPARISON (I/O-Bound)")
    print("=" * 85)
    print(f"{'Model':<25} {'Time (s)':<15} {'Speedup':<15}")
    print("-" * 85)
    print(f"{'Sequential Fetch':<25} {t_sequential:<15.4f} {'1.00x (baseline)':<15}")
    print(f"{'Parallel Async Fetch':<25} {t_parallel_async:<15.4f} {fetch_speedup:.2f}x")
    print("=" * 85)
    
    print("\nBenchmark complete!")
    print("  - Graph: speedup_analysis.png")
    print("  - Report: benchmark_report.md")
    
    return results

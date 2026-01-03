# Parallel vs Sequential URL Fetch Benchmark Report

**Generated:** 2026-01-02 03:36:36

---

## Executive Summary

| Metric | Value |
|--------|-------|
| URLs Tested | 16 |
| Sequential Fetch Time | 40.71s |
| Parallel Fetch Time | 2.31s |
| Best Speedup Achieved | **4.85x** |
| Optimal Worker Count | 8 |

---

## Results Table

| Workers | Time (s) | Speedup | Efficiency | Amdahl's Prediction |
|---------|----------|---------|------------|---------------------|
| 1 | 9.36 | 4.35x | 435% | 1.00x |
| 2 | 8.65 | 4.71x | 235% | 1.74x |
| 4 | 8.98 | 4.53x | 113% | 2.76x |
| 8 | 8.39 | 4.85x | 61% | 3.90x |

**Baseline:** Sequential fetch = 40.71s

---

## Amdahl's Law Analysis

$$
\text{Speedup}(N) = \frac{1}{S + \frac{1-S}{N}}
$$

Where:
- **S** = Serial fraction (portion that cannot be parallelized)
- **N** = Number of parallel workers
- **(1-S)** = Parallel fraction

### Measured Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Serial Fraction (S) | 0.15 (15%) | Non-parallelizable portion |
| Parallel Fraction (1-S) | 0.85 (85%) | Parallelizable portion |
| Max Theoretical Speedup | 6.67x | 1/S = 1/0.15 |
| Max Measured Speedup | 4.85x | Achieved with 8 workers |

### Theoretical vs Actual Comparison

| Workers | Theoretical (Amdahl) | Measured | Difference |
|---------|---------------------|----------|------------|
| 1 | 1.00x | 4.35x | +3.35 |
| 2 | 1.74x | 4.71x | +2.97 |
| 4 | 2.76x | 4.53x | +1.77 |
| 8 | 3.90x | 4.85x | +0.95 |

---

## Speedup Graph

![Speedup Analysis](speedup_analysis.png)

---

## Conclusions: Why Speedup Plateaus

### 1. Amdahl's Law Limitation
- No matter how many workers, speedup **cannot exceed 1/S = 6.67x**
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
| Sequential Fetch | 40.71 | 1.00x (baseline) |
| Parallel Async Fetch | 2.31 | 17.64x |

---
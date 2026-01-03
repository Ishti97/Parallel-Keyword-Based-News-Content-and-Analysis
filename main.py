
import sys, os
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor
from fetcher import fetch_all_async, fetch_all_sequential, run_multiprocessing
from processor import process_page
from aggregator import write_csv
from urls import NEWS_URLS
from utils import timer, measure_cpu
from benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword")
    parser.add_argument("--mode", choices=["sequential", "parallel", "multiprocessing"], default="parallel")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.keyword, args.workers)
        return

    if args.mode == "parallel" and args.workers == 0:
        args.workers = os.cpu_count()
        print(f"Dynamic scaling: Using {args.workers} workers.")

    start = timer()
    results = []

    if args.mode == "sequential":
        pages = fetch_all_sequential(NEWS_URLS)
        for url, html in pages:
            results.extend(process_page((url, html, args.keyword)))

    elif args.mode == "multiprocessing":
        pages = run_multiprocessing(NEWS_URLS, cores=args.workers)
        for url, html in pages:
            results.extend(process_page((url, html, args.keyword)))

    else:  # parallel
        pages = asyncio.run(fetch_all_async(NEWS_URLS))
        tasks = [(url, html, args.keyword) for url, html in pages]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for out in executor.map(process_page, tasks):
                results.extend(out)

    write_csv(results)
    end = timer()

    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers}")
    print(f"Matches: {len(results)}")
    print(f"Time: {end - start:.2f}s")
    print(f"CPU usage: {measure_cpu()}%")

    # # --- 4. OUTPUT FOR YOUR REPORT ---
    # print("\n--- BENCHMARK RESULTS ---")
    # for mode, duration in results.items():
    #     speedup = results['Sequential'] / duration
    #     print(f"{mode:20} | Time: {duration:.2f}s | Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
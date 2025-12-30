
import csv, threading
lock = threading.Lock()

def write_csv(results, filename="output.csv"):
    with lock:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["site", "headline", "link"])
            for r in results:
                w.writerow(r)

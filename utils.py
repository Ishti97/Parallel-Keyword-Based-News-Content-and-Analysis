
import time, psutil
def timer(): return time.perf_counter()
def measure_cpu(): return psutil.cpu_percent(interval=1)

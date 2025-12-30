
# FOR REPORT ONLY – DO NOT USE IN MAIN RUN
import threading, time

lockA = threading.Lock()
lockB = threading.Lock()

def task1():
    with lockA:
        time.sleep(1)
        with lockB:
            pass

def task2():
    with lockB:
        time.sleep(1)
        with lockA:
            pass

t1 = threading.Thread(target=task1)
t2 = threading.Thread(target=task2)
t1.start()
t2.start()
t1.join()
t2.join()

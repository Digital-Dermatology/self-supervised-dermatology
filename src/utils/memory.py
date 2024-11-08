import gc
import os

import psutil


def init_memory_monitor():
    # process to monitor the memory
    process = psutil.Process(os.getpid())
    return process


def start_memory_monitor(process):
    mem_start = process.memory_percent()
    return mem_start


def end_memory_monitor(process, mem_start):
    mem_end = process.memory_percent()
    # tell gc to collect
    gc.collect()
    mem_gc = process.memory_percent()
    mem_temp = (
        "MEMORY MGMT - epoch start: {0:.2f}%, epoch end: {1:.2f}%, after gc: {2:.2f}%"
    )
    print(mem_temp.format(mem_start, mem_end, mem_gc))
    print(
        "MEMORY MGMT - overall memory usage: {0:.2f} Mb ({1:.2f}%)".format(
            psutil.virtual_memory().used / 1024 / 1024,
            psutil.virtual_memory().percent,
        )
    )

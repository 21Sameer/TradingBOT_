import time
import functools

def timer(func):
    """A decorator to measure the execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"⏱️ [Execution Time] {func.__name__}: {run_time:.4f} seconds")
        return result
    return wrapper

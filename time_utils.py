import time


def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        print(f"\nStarting execution of {method.__name__}.")
        result = method(*args, **kw)
        end_time = time.time()
        n_seconds = round(end_time - start_time, 3)
        if n_seconds < 60:
            print(f"\n{method.__name__} : {n_seconds}s to execute")
        elif 60 < n_seconds < 3600:
            print(
                f"\n{method.__name__} : {n_seconds // 60}min {n_seconds % 60}s to execute"
            )
        else:
            print(
                f"\n{method.__name__} : {n_seconds // 3600}h {n_seconds % 3600 // 60}min {n_seconds // 3600 % 60}s to execute"
            )
        return result

    return timed

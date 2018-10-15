import time
from functools import wraps


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("- Function {} tooks {}'s".format(func.__name__, end - start))
        return result

    return wrapper

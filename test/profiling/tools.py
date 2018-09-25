import time


def tick_toc_x1(func):
    def wrapper():
        start = time.time()
        for _ in range(1):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 1.0))

    return wrapper


def tick_toc_x10(func):
    def wrapper():
        start = time.time()
        for _ in range(10):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 10.0))

    return wrapper


def tick_toc_x20(func):
    def wrapper():
        start = time.time()
        for _ in range(10):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff / 20.0))

    return wrapper

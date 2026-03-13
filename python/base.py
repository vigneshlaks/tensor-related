import sys                                                                                                               
                
# guards.py:1895 
def type_guard(x):                                                                                                       
    t = type(x)                                                                                                          
    return lambda y: type(y) == t

# guards.py:2121
def value_guard(x):
    val = x
    return lambda y: y == val

# guards.py:2698
def shape_guard(x):
    if hasattr(x, 'shape'):
        shape = x.shape
        return lambda y: hasattr(y, 'shape') and y.shape == shape
    return lambda y: True  # no-op if not array-like

def dtype_guard(x):
    if hasattr(x, 'dtype'):
        dtype = x.dtype
        return lambda y: hasattr(y, 'dtype') and y.dtype == dtype
    return lambda y: True

class Guard:
    def __init__(self, predicates: list):
        # predicates: list of callables (x) -> bool
        self.predicates = predicates

    def check(self, x) -> bool:
        return all(p(x) for p in self.predicates)

def build_guard(x) -> Guard:
    predicates = [type_guard(x)]

    if isinstance(x, bool):
        predicates.append(value_guard(x))
    elif isinstance(x, int) and abs(x) < 100:
        predicates.append(value_guard(x))
    elif isinstance(x, str):
        predicates.append(value_guard(x))

    if hasattr(x, 'shape'):
        predicates.append(shape_guard(x))
    if hasattr(x, 'dtype'):
        predicates.append(dtype_guard(x))

    return Guard(predicates)

class OptimizeContext:
    def __init__(self, cache_size_limit=8):
        self.cache: list[tuple[Guard, object]] = []
        self.cache_size_limit = cache_size_limit

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            x = args[0]

            for guard, compiled in self.cache:
                if guard.check(x):
                    print(f"  Guard passed for {x!r}")
                    return compiled(x)

            if len(self.cache) >= self.cache_size_limit:
                print(f"  Cache full — falling back to eager for {x!r}")
                return fn(x)

            print(f"  Compiling for {x!r} ...")
            guard = build_guard(x)
            compiled = fn
            self.cache.append((guard, compiled))
            return compiled(x)

        return wrapper


def optimize():
    return OptimizeContext()


@optimize()
def my_func(x):
    return x * 2

cases = [5, 5, 10, "hi", "hi", "bye", True, False, 5, 200]
for v in cases:
    print(f"my_func({v!r}) = {my_func(v)}")
    print()
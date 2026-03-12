import sys                                                                       

class Guard:
    def __init__(self, x):
        self.type = type(x)

    def check(self, x):
        return type(x) == self.type

class OptimizeContext:
    def __init__(self):
        self.compiled = None
        self.guard = None

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            x = args[0]

            if self.guard and self.guard.check(x):
                print("Guard passed - running compiled version")
                return self.compiled(x)

            print("Compiling...")
            self.guard = Guard(x)
            # just set the compiled version
            # to the actual function for now
            self.compiled = fn

            return self.compiled(x)
        return wrapper

def optimize():
    return OptimizeContext()

@optimize()
def my_func(x):
    return x * 2

print(my_func(5))    # compiles
print(my_func(10))   # guard passes, uses compiled
print(my_func("hi")) # guard fails, recompiles
import dis

def simple_add(a, b):
    return a + b

def matrix_mul(x, y):
    result = x @ y
    return result

def neural_net_forward(input, w1):
    pre_activated = input @ w1
    activated = max(0, pre_activated)
    return activated

class SimpleModel:
    def __init__(self, weights):
        self.weights = weights

    def forward(self, x):
        return x @ self.weights

if __name__ == "__main__":
    print("=" * 60)
    print("Simple Addition Function")
    print("=" * 60)
    dis.dis(simple_add)

    print("\n" + "=" * 60)
    print("Matrix Multiplication Function")
    print("=" * 60)
    dis.dis(matrix_mul)

    print("\n" + "=" * 60)
    print("Neural Network Forward Pass")
    print("=" * 60)
    dis.dis(neural_net_forward)

    print("\n" + "=" * 60)
    print("Class Method")
    print("=" * 60)
    dis.dis(SimpleModel.forward)

    print("\n" + "=" * 60)
    print("Lambda Function")
    print("=" * 60)
    relu = lambda x: max(0, x)
    dis.dis(relu)

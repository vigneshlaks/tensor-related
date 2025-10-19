import json
import numpy as np

GROUND_TRUTH = np.array([5, 2])

# store the computations we need to perform
class Executor:
    def __init__(self):
        self.computations = {
            "relu": self.relu_exec,
            "matmul": self.matmul_exec,
            "fused__matmul__relu": self.fused_matmul_relu_exec,
            "mse_loss": self.mse_loss_exec,
        }
    
    def relu_exec(self, x):
        return np.maximum(0, x)

    def matmul_exec(self, a, b):
        return np.matmul(a, b)

    def fused_matmul_relu_exec(self, a, b):
        return np.maximum(0, np.matmul(a, b))

    def mse_loss_exec(self, input):
        return (input - GROUND_TRUTH) ** 2
    
    def launch(self, op, args):
        # probably unideal wtv for now
        args = [arg if isinstance(arg, np.ndarray) else np.array(arg) for arg in args]
        return self.computations[op](*args)

def forward(instrs):
    values = {}
    exec = Executor()
    for instr in instrs:
        # check if it's a computation
        if instr["op"] in exec.computations.keys():
            # get the computed result for our inputs
            inputs = [values[arg] for arg in instr["args"]]
            # replace with computed vals for forward
            instr["args"] = inputs
            res = exec.launch(instr["op"], inputs)
            # save for rest of forward
            values[instr["var_name"]] = res
            # place in json itself to reference during backward
            instr["output"] = res
        elif instr["op"] == "const":
            values[instr["var_name"]] = np.array(instr["value"])
        else:
            raise ValueError(f"{instr['op']} is not a supported Operation")
    return instrs

def iterative_backprop(instrs, ground_truth):
    for i in range(len(instrs) - 1, -1, -1):
        if instrs[i]["op"] == "const":
            continue
        if instrs[i]["op"] == "mse_loss":
            # pred - ground_truth for mse
            instrs[i]["error"] = instrs[i]["args"][0] - ground_truth
        elif instrs[i]["op"] == "relu":
            # apply deriv to send back w mask
            input = instrs[i]["args"][0]
            instrs[i]["error"] = instrs[i+1]["error"] * (input > 0).astype(int)
        elif instrs[i]["op"] == "matmul":
            # input or prev nonlinearity
            input = instrs[i]["args"][0]
            w = instrs[i]["args"][1]
            delta = instrs[i+1]["error"]
            # grad acc
            instrs[i]["grad"] = np.outer(delta, input)
            # also send back error
            instrs[i]["error"] = w.T @ delta
        else:
            raise ValueError(f"{instr['op']} is not a supported Operation")

def descent_step(instrs, lr):
    for instr in instrs:
        if "grad" in instr:
            # matmul (the weights) is the only parameter to update right now
            if instr["op"] == "matmul":
                instr["args"][1] -= instr["grad"].T * lr
            else:
                raise ValueError(f"{instr['op']} is not a supported Operation")

def backward(instrs):
    # random y
    return iterative_backprop(instrs, np.array([5, 2]))

if __name__ == "__main__":
    # get instrs
    with open("./autograd/1-layer.json", "r") as ir:
        instrs = json.load(ir)

    forward(instrs)
    
    for instr in instrs:
        print(instr)
    
    print()

    backward(instrs)

    for instr in instrs:
        print(instr)
    
    # learning rate of 0.01
    descent_step(instrs, 0.01)

import json
import numpy as np

GROUND_TRUTH = np.array([5, 2])

class Executor:
    def __init__(self):
        # store the computations to perform
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
        args = [arg if isinstance(arg, np.ndarray) else np.array(arg, dtype=np.float64) for arg in args]
        return self.computations[op](*args)

def forward(instrs):
    values = {}
    exec = Executor()
    for instr in instrs:
        # check if it's a computation
        if instr["op"] in exec.computations.keys():
            # get the computed result for our inputs
            inputs = [values[arg] for arg in instr["args"]]
            # execute the operation
            res = exec.launch(instr["op"], inputs)
            # save for rest of forward
            values[instr["var_name"]] = res
            # place in json itself to reference during backward
            instr["output"] = res
            instr["input_values"] = inputs
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
            instrs[i]["error"] = 2 * (instrs[i]["input_values"][0] - ground_truth)
        elif instrs[i]["op"] == "relu":
            # apply deriv to send back w mask
            input = instrs[i]["input_values"][0]
            instrs[i]["error"] = instrs[i+1]["error"] * (input > 0).astype(int)
        elif instrs[i]["op"] == "matmul":
            # input or prev nonlinearity
            input = instrs[i]["input_values"][0]
            w = instrs[i]["input_values"][1]
            delta = instrs[i+1]["error"]
            # grad acc
            instrs[i]["grad"] = np.outer(input, delta)
            # also send back error
            instrs[i]["error"] = delta @ w.T
        else:
            raise ValueError(f"{instrs[i]['op']} is not a supported Operation")

def descent_step(instrs, lr):
    # build a map of variable names to their const instructions
    const_map = {}
    for instr in instrs:
        if instr["op"] == "const":
            const_map[instr["var_name"]] = instr

    for instr in instrs:
        if "grad" in instr:
            if instr["op"] == "matmul":
                # update the const value for the weight parameter
                weight_var = instr["args"][1]
                const_map[weight_var]["value"] = (np.array(const_map[weight_var]["value"]) - instr["grad"] * lr).tolist()
            else:
                raise ValueError(f"{instr['op']} is not a supported Operation")

def backward(instrs):
    return iterative_backprop(instrs, GROUND_TRUTH)

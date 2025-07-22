import json
# temporary to get end to end will replace with triton mlir in end of day
import torch
from dotenv import load_dotenv
import os

GROUND_TRUTH = torch.tensor([5,2])

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
        return torch.relu(x)
    
    def matmul_exec(self, a, b):
        return torch.matmul(a, b)
    
    def fused_matmul_relu_exec(self, a, b):
        return torch.relu(torch.matmul(a, b))

    def mse_loss_exec(self, input):
        return (input - GROUND_TRUTH) ** 2
    
    def launch(self, op, args):
        # probably unideal wtv for now
        args = [arg if torch.is_tensor(arg) else torch.tensor(arg) for arg in args]
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
            # place in json itself  to reference during backward
            instr["output"] = res
        elif instr["op"] == "const":
            values[instr["var_name"]] = instr["value"]
        else:
            raise ValueError(f"{instr['op']} is not a supported Operation")
    return instrs

def iterative_backprop(instrs, ground_truth):
    for i in range(len(instrs) - 1, -1, -1):
        if instrs[i]["op"] == "const":
            continue
        if instrs[i]["op"] == "mse_loss":
            instrs[i]["backprop"] = ground_truth - instrs[i]["output"]
        elif instrs[i]["op"] == "relu":
            # if in bounds and next is loss node
            if i+1 < len(instrs):
                if instrs[i+1]["op"] == "mse_loss":
                    # send back backprop from loss node
                    instrs[i]["backprop"] = instrs[i+1]["backprop"]
                else:
                    # assume matmul (preactivation of following layer) in front
                    matmul_instr = instrs[i+1]
                    for arg in matmul_instr["args"]:
                        # find matrix op
                        if arg.dim() == 2:
                            # send back through the weight matrix to where we are now
                            # arg is the computed tensor btw
                            instrs[i]["backprop"] = arg.T * matmul_instr["backprop"]
        elif instrs[i]["op"] == "matmul":
            relu_instr = instrs[i+1]
            deriv = instrs[i]["output"] * (instrs[i]["output"] > 0).int()
            instrs[i]["backprop"] = relu_instr["backprop"] * deriv
        else:
            raise ValueError("Unsupported Op")

def backward(instrs):
    # assume arb. output
    #layer_backprop(instrs, 0, torch.tensor([5,2]))
    return iterative_backprop(instrs, torch.tensor([5,2]))

def clean_output(instrs):
    for instr in instrs:
        if "output" in instr:
            print(instr)
            instr["output"] = torch.Tensor.tolist(instr["output"])

if __name__ == "__main__":
    # get instrs
    load_dotenv()
    autograd_loc = os.getenv('autograd_loc')
    with open(f"{autograd_loc}/1-layer.json", "r") as ir:
        instrs = json.load(ir)

    forward_dir = forward(instrs)
    for instr in instrs:
        print(instr)
    
    print()

    back = backward(forward_dir)

    for instr in instrs:
        print(instr)


    #backward_dir = backward(forward_dir)

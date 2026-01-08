import json
import numpy as np

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

    def mse_loss_exec(self, input, ground_truth):
        return (input - ground_truth) ** 2

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
            # for mse_loss, add ground truth
            if instr["op"] == "mse_loss":
                inputs.append(np.array(instr["value"]))
                instr["ground_truth"] = np.array(instr["value"])
            # execute the operation
            res = exec.launch(instr["op"], inputs)
            # save for rest of forward
            values[instr["id"]] = res
            # place in json itself to reference during backward
            instr["input_values"] = inputs
        elif instr["op"] == "const":
            values[instr["id"]] = np.array(instr["value"])
        else:
            raise ValueError(f"{instr['op']} is not a supported Operation")
    return instrs

def iterative_backprop(instrs):
    # Initialize gradient accumulator for all tensors
    tensor_grads = {}

    for i in range(len(instrs) - 1, -1, -1):
        if instrs[i]["op"] == "const":
            continue
        if instrs[i]["op"] == "mse_loss":
            # Initialize mse output gradient to 1 (like C++)
            instrs[i]["grad"] = np.ones_like(instrs[i]["input_values"][0])
            # pred - ground_truth for mse, multiplied by output gradient
            instrs[i]["error"] = instrs[i]["grad"] * 2 * (instrs[i]["input_values"][0] - instrs[i]["ground_truth"])
        elif instrs[i]["op"] == "relu":
            # apply deriv to send back w mask
            input = instrs[i]["input_values"][0]
            instrs[i]["error"] = instrs[i+1]["error"] * (input > 0).astype(int)
        elif instrs[i]["op"] == "matmul":
            # input or prev nonlinearity
            input_id = instrs[i]["args"][0]
            weight_id = instrs[i]["args"][1]
            input = instrs[i]["input_values"][0]
            w = instrs[i]["input_values"][1]
            delta = instrs[i+1]["error"]
            # grad for weights (accumulate on weight tensor)
            instrs[i]["weight_grad"] = np.outer(input, delta)
            tensor_grads[weight_id] = np.outer(input, delta)
            # error sent back to input (accumulate on input tensor)
            instrs[i]["error"] = delta @ w.T
            tensor_grads[input_id] = delta @ w.T
        else:
            raise ValueError(f"{instrs[i]['op']} is not a supported Operation")

    return tensor_grads
 
def recursive_backprop(instrs):
    def backprop(index, error):
        instr = instrs[index]

        if instr["op"] == "const":
            return
        if instr["op"] == "mse_loss":
            # pred - ground_truth for mse
            backprop(index - 1, 2 * (instr["input_values"][0] - instr["ground_truth"]))
        elif instr["op"] == "relu":
            # apply deriv to send back w mask
            input = instr["input_values"][0]
            new_error = error * (input > 0).astype(int)
            backprop(index - 1, new_error)
        elif instr["op"] == "matmul":
            # input or prev nonlinearity
            input = instr["input_values"][0]
            w = instr["input_values"][1]
            delta = error
            # grad acc
            instr["grad"] = np.outer(input, delta)
            # also send back error
            backprop(index - 1, delta @ w.T)
        else:
            raise ValueError(f"{instr['op']} is not a supported Operation")
    
    # dummy error value
    backprop(len(instrs) - 1, 1)

def descent_step(instrs, lr):
    # build a map of variable names to their const instructions
    const_map = {}
    for instr in instrs:
        if instr["op"] == "const":
            const_map[instr["id"]] = instr

    for instr in instrs:
        if "grad" in instr:
            if instr["op"] == "matmul":
                # update the const value for the weight parameter
                weight_var = instr["args"][1]
                const_map[weight_var]["value"] = (np.array(const_map[weight_var]["value"]) - instr["grad"] * lr).tolist()
            else:
                raise ValueError(f"{instr['op']} is not a supported Operation")

def backward(instrs):
    return iterative_backprop(instrs)

if __name__ == "__main__":
    with open("1-layer.json", "r") as f:
        data = json.load(f)

    forward(data)
    # recursive_backprop(data)
    tensor_grads = iterative_backprop(data)

    print("\nForward Pass Results:")
    print("input:", data[0]["value"])
    print("w1:", data[1]["value"])
    print("pre_activated:", np.matmul(np.array(data[0]["value"]), np.array(data[1]["value"])))
    print("y_hat:", np.maximum(0, np.matmul(np.array(data[0]["value"]), np.array(data[1]["value"]))))
    y_hat = np.maximum(0, np.matmul(np.array(data[0]["value"]), np.array(data[1]["value"])))
    print("mse:", (y_hat - np.array(data[4]["value"])) ** 2)

    print("\nTensor Gradients:")
    for instr in data:
        if instr["op"] == "const":
            if instr["id"] in tensor_grads:
                print(f"{instr['id']} grad: {tensor_grads[instr['id']].flatten()}")
            else:
                print(f"{instr['id']} grad: (no gradient - leaf)")

    for instr in data:
        if instr["op"] != "const":
            if "error" in instr:
                print(f"{instr['id']} grad: {instr['error']}")
            if "grad" in instr:
                print(f"{instr['id']} output grad: {instr['grad']}")

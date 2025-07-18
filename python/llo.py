import json
# temporary to get end to end will replace with triton mlir in end of day
import torch

class KernelExecutor:
    def __init__(self):
        self.kernels = {
            "relu": self.relu_kernel,
            "matmul": self.matmul_kernel,
            "fused__matmul__relu": self.fused_matmul_relu_kernel,
        }
    
    def relu_kernel(self, x):
        print(f"Launching ReLU kernel on {x.shape}")
        return torch.relu(x)
    
    def matmul_kernel(self, a, b):
        print(f"Launching MatMul kernel: {a.shape} @ {b.shape}")
        return torch.matmul(a, b)
    
    def fused_matmul_relu_kernel(self, a, b):
        print(f"Launching Fused MatMul+ReLU kernel")
        print(a)
        print(b)
        print(torch.relu(torch.matmul(a, b)))
        return torch.relu(torch.matmul(a, b))
    
    def launch(self, op, args):
        args = [torch.tensor(arg) for arg in args]
        return self.kernels[op](*args)

def execute_ir(instrs):
    values = {}
    exec = KernelExecutor()
    output = None
    for instr in instrs:
        # check if it's a kernel
        if instr["op"] in exec.kernels.keys():
            inputs = [values[arg] for arg in instr["args"]]
            values[instr["dest"]] = exec.launch(instr["op"], inputs)
        elif instr["op"] == "const":
            values[instr["dest"]] = instr["value"]
        elif instr["op"] == "print":
            print(f"Print Op - {instr['args'][0]}: {values[instr['args'][0]]}")
        else:
            raise ValueError(f"{instr['op']} is not a support Operation")
        output = instr
    return output

        

if __name__ == "__main__":
    with open("hlo_op.json", "r") as file:
        instrs = json.load(file)

    execute_ir(instrs)

    #print(instrs)

    # for instr in instrs:
    #     print(instr)

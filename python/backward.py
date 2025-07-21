import json 
from typing import List, Dict
from llo import forward
'''
I want to create a backward graph
find loss node grad
then find grad between layers

1. find loss node grad
2. find grad between layers

Keep index positions within instrs for other graph
'''

# change to something better
COMPUTE_NODES = {
    "matmul",
    "relu"
}

def layer_backprop(instrs, index, loss_grad):
    # grab loss grad computed in other func
    # assume last ind is loss
    if index == len(index) - 1:
        return loss_grad
    # place next instr first
    next_instr = layer_backprop(instrs, index + 1, loss_grad)
    curr_instr = next_instr["grad"]
    return curr_instr + next_instr

def loss_grad(pred_instr):
    pred_instr[""]
    pass

def backward(instrs):
    lg = loss_grad(instrs[-1])
    layer_backprop(instrs, 0, lg)

if __name__ == "__main__":
    with open('./irs/clean_nn.json') as file:
        nn = json.load(file)

    instrs = nn["functions"][0]["instrs"]
    forward_dir = forward(instrs)
    # assume for now we get a cleaned nn coming in
    backward(forward_dir)

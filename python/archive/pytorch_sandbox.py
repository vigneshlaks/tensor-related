import torch
from torchvision import datasets, transforms

# define neural net
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # defined with one layer
        self.flatten = torch.nn.Flatten()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(784,10),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.sequential(self.flatten(x))

import torch.fx
import json

# Convert to IR function
def pytorch_to_ir(model, sample_input):
    traced = torch.fx.symbolic_trace(model)
    
    instructions = []
    var_counter = 0
    
    # Add parameters as constants
    for name, param in model.named_parameters():
        instructions.append({
            "dest": name.replace(".", "_"),
            "op": "const", 
            "value": param.detach().tolist()
        })
    
    # Convert graph nodes
    for node in traced.graph.nodes:
        if node.op == 'call_function':
            if node.target == torch.flatten:
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "flatten",
                    "args": [str(node.args[0])]
                })
            elif 'linear' in str(node.target).lower():
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "linear", 
                    "args": [str(node.args[0]), str(node.args[1]), str(node.args[2])]
                })
            elif 'relu' in str(node.target).lower():
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "relu",
                    "args": [str(node.args[0])]
                })
            var_counter += 1
    
    return {"instructions": instructions}

# Use it
nn = NN()

sample_input = torch.randn(1, 1, 28, 28)
ir = pytorch_to_ir(nn, sample_input)

with open('model_ir', 'w') as f:
    json.dump(ir, f, indent=2)

# put dataset into dataloader
t = transforms.ToTensor()
train_set = datasets.FashionMNIST('data', train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_set)
test_loader = torch.utils.data.DataLoader(test_set)

# 100 epochs
# store the loss here
test_losses = []

nn = NN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nn.parameters(), lr=0.01) 

for _ in range(100):
    # train
    for img, label in train_loader:
        guess = nn.forward(img)
        # compute loss
        loss = criterion(guess, label)
        # clear gradient
        optimizer.zero_grad()
        # acc gradient
        loss.backward()
        # update params
        optimizer.step()

    # test
    total_loss = 0
    with torch.no_grad():
        for img, label in test_loader:
            guess = nn.forward(img)
            loss = criterion(guess, label)
            total_loss += loss.item()
    

    avg_loss = total_loss / len(test_loader)

    print(avg_loss)
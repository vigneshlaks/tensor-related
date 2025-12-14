# To see what the graph output looks like
import torch
import torch.nn as nn
import torch.fx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Create model and trace it
model = SimpleModel()
traced = torch.fx.symbolic_trace(model)

# Print the graph
print("=" * 60)
print("FX Graph Structure")
print("=" * 60)
print(traced.graph)

print("\n" + "=" * 60)
print("Graph Code (Readable)")
print("=" * 60)
print(traced.code)

print("\n" + "=" * 60)
print("Graph Nodes (Tabular)")
print("=" * 60)
traced.graph.print_tabular()

print("\n" + "=" * 60)
print("Individual Node Information")
print("=" * 60)
for node in traced.graph.nodes:
    print(f"Node: {node.name}")
    print(f"  Op: {node.op}")
    print(f"  Target: {node.target}")
    print(f"  Args: {node.args}")
    print(f"  Kwargs: {node.kwargs}")
    print()

# Test with sample input
print("=" * 60)
print("Test Execution")
print("=" * 60)
x = torch.randn(1, 10)
output = traced(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")
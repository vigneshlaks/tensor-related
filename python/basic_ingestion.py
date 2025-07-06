import torch
import json

def pytorch_to_ir(model, sample_input):
    """
    Convert a PyTorch model to custom IR format.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor for tracing
    
    Returns:
        dict: IR representation with instructions
    """
    # Trace the model to get computation graph
    traced_model = torch.fx.symbolic_trace(model)
    
    instructions = []
    var_counter = 0

    # Step 1: Extract model parameters as constants
    for param_name, param_tensor in model.named_parameters():
        print(param_name)
        # Convert parameter name to valid variable name
        clean_name = param_name.replace(".", "_")
        
        instructions.append({
            "dest": clean_name,
            "op": "const",
            "value": param_tensor.detach().tolist()
        })

    # Step 2: Convert computation graph nodes to IR
    for node in traced_model.graph.nodes:
        if node.op == 'call_function':
            # Handle different operation types
            op_name = str(node.target).lower()
            
            if 'flatten' in op_name:
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "flatten",
                    "args": [str(node.args[0])]
                })
            elif 'linear' in op_name or 'addmm' in op_name:
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "linear",
                    "args": [str(arg) for arg in node.args]
                })
            elif 'relu' in op_name:
                instructions.append({
                    "dest": f"v{var_counter}",
                    "op": "relu", 
                    "args": [str(node.args[0])]
                })
            
            var_counter += 1
    
    return {"instructions": instructions}

def save_ir_to_file(ir_dict, filename="model_ir.json"):
    """Save IR dictionary to JSON file."""
    with open(filename, 'w') as file_handle:
        json.dump(ir_dict, file_handle, indent=2)
    print(f"IR saved to {filename}")

def load_ir_from_file(filename="model_ir.json"):
    """Load IR dictionary from JSON file."""
    with open(filename, 'r') as file_handle:
        ir_dict = json.load(file_handle)
    print(f"IR loaded from {filename}")
    return ir_dict

# Example usage:
if __name__ == "__main__":
    # Define a tiny model for easy viewing
    class TinyNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)  # 4 inputs -> 2 outputs
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x
    
    # Create model and sample input
    model = TinyNN()
    sample_input = torch.randn(1, 4)  # Batch=1, Features=4
    
    # Convert to IR
    ir = pytorch_to_ir(model, sample_input)
    
    # Save to file
    save_ir_to_file(ir, "my_model.json")
    
    # Load from file (example)
    loaded_ir = load_ir_from_file("my_model.json")
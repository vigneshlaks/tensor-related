import json

def ir_to_json():
   pass

def json_to_ir(json):
   lines = []
   for instr in json["instructions"]:
       string = None
       if instr["op"] == "const":
           string = f"{instr['dest']} = {instr['op']} {instr['value']};"
       else:
           string = f"{instr['dest']} = {instr['op']} {instr['args']};"
       lines.append(string)

   with open("output.txt", "w") as file:
       file.write("\n".join(lines))

def fuse_operations(instructions):
   """Transform separate operations into fused operations"""
   result = []
   i = 0
   
   while i < len(instructions):
       instr = instructions[i]
       
       # Look for fusion opportunity: current op feeds into next op
       if (i + 1 < len(instructions) and 
           can_fuse(instr, instructions[i + 1])):
           
           # Create fused operation
           fused = create_fused_op(instr, instructions[i + 1])
           result.append(fused)
           i += 2  # Skip both instructions
       else:
           # Keep instruction as-is
           result.append(instr)
           i += 1
   
   return result

def is_pointwise_op(op):
    """Check if operation is pointwise (element-by-element)"""
    pointwise_ops = {"add", "multiply", "subtract", "divide", "relu", "sigmoid", "tanh", "exp", "log"}
    return op in pointwise_ops

def can_fuse(op1, op2):
    """Check if op1 output feeds directly into op2 AND both are pointwise"""
    return (op1["dest"] in op2.get("args", []) and 
            is_pointwise_op(op2["op"]))

def create_fused_op(producer, consumer):
    """Create a fused operation from producer-consumer pair"""
    return {
        "dest": consumer["dest"],
        "op": "fused",
        "steps": [
            f"{producer['op']}({', '.join(producer['args'])})",
            f"{consumer['op']}($prev, {consumer['args'][1]})"
        ]
    }

'''
Then we need to write dead code elimination

Then need to write ssa?

Need to also write operator fusion.
'''

if __name__ == "__main__":
   with open("data.json", 'r') as file:
       data = json.load(file)

   # Apply fusion transformation
   data["instructions"] = fuse_operations(data["instructions"])
   
   # Output the fused JSON
   with open("fused_output.json", "w") as file:
       json.dump(data, file, indent=2)
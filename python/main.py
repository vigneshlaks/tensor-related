import json

def ir_to_json():
   pass

def json_to_ir(json_data):
   lines = []
   for instr in json_data["instructions"]:
       string = None
       if instr["op"] == "const":
           string = f"{instr['dest']} = {instr['op']} {instr['value']};"
       else:
           string = f"{instr['dest']} = {instr['op']} {instr['args']};"
       lines.append(string)

   with open("output.txt", "w") as file:
       file.write("\n".join(lines))

def trivial_dce_pass(instructions):
    """Remove instructions that are never used as arguments
    to any other instruction. Return a bool indicating whether we deleted
    anything.
    """
    # Find all the variables used as an argument to any instruction
    used = set()
    for instr in instructions:
        # Mark all the variable arguments as used
        used.update(instr.get("args", []))

    # Delete the instructions that write to unused variables
    # Keep effect instructions that don't produce a result
    new_instructions = [i for i in instructions if "dest" not in i or i["dest"] in used]

    # Record whether we deleted anything
    changed = len(new_instructions) != len(instructions)

    return new_instructions, changed

def trivial_dce(instructions):
    """Iteratively remove dead instructions, stopping when nothing
    remains to remove.
    """
    while True:
        instructions, changed = trivial_dce_pass(instructions)
        if not changed:
            break
    return instructions

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
            is_pointwise_op(op1["op"]) and
            is_pointwise_op(op2["op"]))

def create_fused_op(producer, consumer):
    """Create a fused operation from producer-consumer pair"""
    # Filter out the intermediate variable from consumer args
    consumer_args = [arg for arg in consumer.get("args", []) if arg != producer["dest"]]
    
    return {
        "dest": consumer["dest"],
        "op": "fused",
        "steps": [
            f"{producer['op']}({', '.join(producer.get('args', []))})",
            f"{consumer['op']}($prev, {', '.join(consumer_args)})"
        ]
    }

def test_dead_code_elimination():
    """Test cases for DCE"""
    print("=== Testing Dead Code Elimination ===")
    
    # Test case 1: Simple dead code
    test1 = {
        "instructions": [
            {"dest": "a", "op": "const", "value": 5},
            {"dest": "b", "op": "const", "value": 10},
            {"dest": "c", "op": "add", "args": ["a", "b"]},  # c is used
            {"dest": "d", "op": "multiply", "args": ["a", "b"]},  # d is NOT used
            {"dest": "result", "op": "subtract", "args": ["c", "a"]}
        ]
    }
    
    print("Before DCE:")
    for instr in test1["instructions"]:
        print(f"  {instr}")
    
    optimized = trivial_dce(test1["instructions"])
    print("\nAfter DCE:")
    for instr in optimized:
        print(f"  {instr}")
    print(f"Removed {len(test1['instructions']) - len(optimized)} instructions\n")
    
    # Test case 2: Chain of dead code
    test2 = {
        "instructions": [
            {"dest": "x", "op": "const", "value": 1},
            {"dest": "y", "op": "const", "value": 2},
            {"dest": "dead1", "op": "add", "args": ["x", "y"]},
            {"dest": "dead2", "op": "multiply", "args": ["dead1", "x"]},
            {"dest": "dead3", "op": "subtract", "args": ["dead2", "y"]},
            {"dest": "final", "op": "exp", "args": ["x"]}  # Only this chain survives
        ]
    }
    
    print("Chain of dead code test:")
    print("Before DCE:")
    for instr in test2["instructions"]:
        print(f"  {instr}")
    
    optimized2 = trivial_dce(test2["instructions"])
    print("\nAfter DCE:")
    for instr in optimized2:
        print(f"  {instr}")
    print(f"Removed {len(test2['instructions']) - len(optimized2)} instructions\n")

def test_operator_fusion():
    """Test cases for operator fusion"""
    print("=== Testing Operator Fusion ===")
    
    # Test case 1: Simple fusion opportunity
    test1 = {
        "instructions": [
            {"dest": "a", "op": "const", "value": 5},
            {"dest": "b", "op": "const", "value": 10},
            {"dest": "temp", "op": "add", "args": ["a", "b"]},
            {"dest": "result", "op": "relu", "args": ["temp"]}
        ]
    }
    
    print("Simple fusion test:")
    print("Before fusion:")
    for instr in test1["instructions"]:
        print(f"  {instr}")
    
    fused = fuse_operations(test1["instructions"])
    print("\nAfter fusion:")
    for instr in fused:
        print(f"  {instr}")
    print()
    
    # Test case 2: Multiple fusion opportunities
    test2 = {
        "instructions": [
            {"dest": "x", "op": "const", "value": 3},
            {"dest": "y", "op": "const", "value": 4},
            {"dest": "z", "op": "const", "value": 2},
            {"dest": "temp1", "op": "add", "args": ["x", "y"]},
            {"dest": "temp2", "op": "multiply", "args": ["temp1", "z"]},
            {"dest": "final", "op": "tanh", "args": ["temp2"]}
        ]
    }
    
    print("Multiple fusion test:")
    print("Before fusion:")
    for instr in test2["instructions"]:
        print(f"  {instr}")
    
    fused2 = fuse_operations(test2["instructions"])
    print("\nAfter fusion:")
    for instr in fused2:
        print(f"  {instr}")
    print()
    
    # Test case 3: No fusion possible (non-pointwise ops)
    test3 = {
        "instructions": [
            {"dest": "a", "op": "const", "value": 5},
            {"dest": "b", "op": "const", "value": 10},
            {"dest": "temp", "op": "matmul", "args": ["a", "b"]},  # Not pointwise
            {"dest": "result", "op": "add", "args": ["temp", "a"]}
        ]
    }
    
    print("No fusion possible test:")
    print("Before fusion:")
    for instr in test3["instructions"]:
        print(f"  {instr}")
    
    fused3 = fuse_operations(test3["instructions"])
    print("\nAfter fusion:")
    for instr in fused3:
        print(f"  {instr}")
    print("(Should be unchanged)")
    print()

def test_combined_optimizations():
    """Test DCE + Fusion together"""
    print("=== Testing Combined Optimizations ===")
    
    test_data = {
        "instructions": [
            {"dest": "x", "op": "const", "value": 1},
            {"dest": "y", "op": "const", "value": 2},
            {"dest": "z", "op": "const", "value": 3},
            {"dest": "dead_var", "op": "add", "args": ["x", "y"]},  # Dead code
            {"dest": "temp1", "op": "multiply", "args": ["x", "z"]},
            {"dest": "temp2", "op": "add", "args": ["temp1", "y"]},  # Can fuse with temp1
            {"dest": "another_dead", "op": "subtract", "args": ["y", "z"]},  # Dead code
            {"dest": "final", "op": "sigmoid", "args": ["temp2"]}  # Can fuse with temp2
        ]
    }
    
    print("Original code:")
    for instr in test_data["instructions"]:
        print(f"  {instr}")
    
    # Apply DCE first
    after_dce = trivial_dce(test_data["instructions"])
    print(f"\nAfter DCE (removed {len(test_data['instructions']) - len(after_dce)} instructions):")
    for instr in after_dce:
        print(f"  {instr}")
    
    # Apply fusion
    final_optimized = fuse_operations(after_dce)
    print(f"\nAfter fusion (combined {len(after_dce) - len(final_optimized)} pairs):")
    for instr in final_optimized:
        print(f"  {instr}")


if __name__ == "__main__":
    test_dead_code_elimination()
    test_operator_fusion()
    test_combined_optimizations()
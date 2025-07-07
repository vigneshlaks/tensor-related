import json
import torch
import triton
import triton.language as tl

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

def fuse_operations_pass(instructions):
    """Single pass of fusion operation. Return new instructions and whether anything changed."""
    result = []
    used = set()
    changed = False
    
    for i, instr in enumerate(instructions):
        if i in used:
            continue
            
        # Skip non-pointwise operations
        if not is_pointwise_op(instr.get("op", "")):
            result.append(instr)
            continue
            
        # Build fusion chain starting from this instruction
        chain = [instr]
        used.add(i)
        
        # Keep extending the chain
        while True:
            last_op = chain[-1]
            found_next = False
            
            # Look for the next instruction that can be fused
            for j in range(len(instructions)):
                if j in used:
                    continue
                    
                candidate = instructions[j]
                
                # Check if candidate uses the output of last_op and is pointwise
                if (is_pointwise_op(candidate.get("op", "")) and
                    last_op["dest"] in candidate.get("args", [])):
                    
                    # Make sure last_op output is only used by this candidate
                    usage_count = sum(1 for k, inst in enumerate(instructions) 
                                    if k not in used and last_op["dest"] in inst.get("args", []))
                    
                    if usage_count == 1:
                        chain.append(candidate)
                        used.add(j)
                        found_next = True
                        break
            
            if not found_next:
                break
        
        # Add the result (fused or single operation)
        if len(chain) > 1:
            result.append(create_fused_chain(chain))
            changed = True
        else:
            result.append(chain[0])
    
    return result, changed

def fuse_operations(instructions):
    """Iteratively fuse operations until no more changes occur"""
    while True:
        instructions, changed = fuse_operations_pass(instructions)
        if not changed:
            break
    return instructions

def can_extend_fusion_chain(output_var, next_instr, all_instructions, used_indices):
    """Check if we can safely add next_instr to the fusion chain"""
    # Make sure the output variable is only used by this next instruction
    # (among unused instructions)
    uses_count = 0
    for k, instr in enumerate(all_instructions):
        if k in used_indices:
            continue
        if output_var in instr.get("args", []):
            uses_count += 1
    
    return uses_count == 1

def create_fused_chain(fusion_chain):
    """Create a fused operation from a chain of operations"""
    steps = []
    
    # Track which variables are intermediate (produced and consumed within the chain)
    intermediate_vars = set()
    for i in range(len(fusion_chain) - 1):
        intermediate_vars.add(fusion_chain[i]["dest"])
    
    # Collect all external arguments (not intermediate variables)
    external_args = []
    for op in fusion_chain:
        for arg in op.get("args", []):
            if arg not in intermediate_vars and arg not in external_args:
                external_args.append(arg)
    
    # Build the computation steps
    for i, op in enumerate(fusion_chain):
        if i == 0:
            # First operation: use its original arguments
            args_str = ", ".join(op.get("args", []))
            steps.append(f"{op['op']}({args_str})")
        else:
            # Subsequent operations: use $prev for the chained input, plus any external args
            prev_output = fusion_chain[i-1]["dest"]
            other_args = [arg for arg in op.get("args", []) if arg != prev_output]
            
            if other_args:
                args_str = "$prev, " + ", ".join(other_args)
            else:
                args_str = "$prev"
                
            steps.append(f"{op['op']}({args_str})")
    
    # Create fused operation with metadata from the last operation
    last_op = fusion_chain[-1]
    fused_op = {
        "dest": last_op["dest"],
        "op": "fused",
        "steps": steps
    }
    
    # Only include external args if there are any
    if external_args:
        fused_op["args"] = external_args
    
    # Preserve metadata from the last operation
    for key in ["shape", "dtype"]:
        if key in last_op:
            fused_op[key] = last_op[key]
    
    return fused_op

def is_pointwise_op(op):
    """Check if operation is pointwise (element-by-element)"""
    pointwise_ops = {"add", "multiply", "subtract", "divide", "relu", "sigmoid", "tanh", "exp", "log"}
    return op in pointwise_ops

# Legacy functions kept for compatibility
def can_fuse(op1, op2):
    """Check if op1 output feeds directly into op2 AND both are pointwise"""
    return (op1["dest"] in op2.get("args", []) and 
            is_pointwise_op(op1["op"]) and
            is_pointwise_op(op2["op"]))

def create_fused_op(producer, consumer):
    """Create a fused operation from producer-consumer pair"""
    return create_fused_chain([producer, consumer])

def run_pipeline(instructions):
    """Run the complete optimization pipeline: DCE followed by operator fusion"""
    # Save original instructions
    original_data = {"instructions": instructions}
    with open("00_original.json", "w") as f:
        json.dump(original_data, f, indent=2)
    
    # Step 1: Apply dead code elimination
    optimized_instructions = trivial_dce(instructions)
    after_dce_data = {"instructions": optimized_instructions}
    with open("01_after_dce.json", "w") as f:
        json.dump(after_dce_data, f, indent=2)
    
    # Step 2: Apply operator fusion
    final_instructions = fuse_operations(optimized_instructions)
    final_data = {"instructions": final_instructions}
    with open("02_final_optimized.json", "w") as f:
        json.dump(final_data, f, indent=2)
    
    return final_instructions

@triton.jit
def fused_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple fused kernel for pointwise operations"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Placeholder for fused operations
    result = x
    
    tl.store(output_ptr + offsets, result, mask=mask)

def execute_kernel(instruction):
    """Execute a single instruction using Triton kernel"""
    if instruction["op"] == "fused":
        # Create input/output tensors
        input_tensor = torch.randn(1024, device='cuda')
        output_tensor = torch.empty_like(input_tensor)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(input_tensor.numel(), meta['BLOCK_SIZE']),)
        fused_kernel[grid](input_tensor, output_tensor, input_tensor.numel(), BLOCK_SIZE=256)
        
        return output_tensor
    else:
        # Fallback for non-fused ops
        return torch.randn(1024, device='cuda')
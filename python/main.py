import json
from typing import Any

FUSABLE_OPS = {"matmul", "relu"}

def trivial_dce_pass(instructions):
    """Remove instructions that are never used as arguments"""
    used = set()
    for instr in instructions:
        used.update(instr.get("args", []))
    
    # include if not a variable or a used variable
    new_instructions = [i for i in instructions if "dest" not in i or i["dest"] in used]
    changed = len(new_instructions) != len(instructions)
    return new_instructions, changed

def trivial_dce(instructions):
    """Iteratively remove dead instructions"""
    while True:
        instructions, changed = trivial_dce_pass(instructions)
        if not changed:
            break
    return instructions

def is_fusable_op(op:str) -> bool:
    fusable_ops = {"matmul", "relu"}
    return op in fusable_ops

def create_fused_op(chain: list[Any]) -> dict:
    # initial input are the new arguments
    args = [chain[0]["args"]]
    dests = []
    op = []
    
    for instr in chain:
        assert "args" in instr, "args not in instr"
        assert "dest" in instr, "dest not in instr"
        dests.append(instr["dest"])
        op.append(instr["op"])

    fused_op = {}
    # concat the names
    fused_op["dest"] = f"__".join(dests)
    fused_op["args"] = args
    fused_op["op"] = f"fused__{"__".join(op)}"

    return fused_op

def fuse_operations(instructions):
    result = []
    curr_chain = []
    for instr in instructions:
        if instr["op"] in FUSABLE_OPS:
            curr_chain.append(instr)
        else:
            curr_args = instr["args"] if "args" in instr else []
            
            if curr_chain != []:
                new_args = []
                fused_op = create_fused_op(curr_chain)
                curr_chain_dests = [op['dest'] for op in curr_chain]
                need_fused = False
                for arg in curr_args:
                    # if the argument is one that has been fused
                    if arg in curr_chain_dests:
                        need_fused = True
                        continue
                    new_args.append(arg)
                if need_fused:
                    new_args.append(fused_op["dest"])

                result.append(fused_op)
                instr["args"] = new_args
                curr_chain = []

            result.append(instr)
    
    # take care of last chain if needed
    if curr_chain != []:
        result.append(create_fused_op(curr_chain))
    
    return result

def run_pipeline(instructions):
    """Run the complete optimization pipeline"""
    print("Original instructions:")
    for i, instr in enumerate(instructions):
        print(f"  {i}: {instr}")
    
    # Step 1: Dead code elimination
    optimized_instructions = trivial_dce(instructions)
    print(f"\nAfter DCE: {len(optimized_instructions)} instructions")
    
    # Step 2: Operator fusion
    final_instructions = fuse_operations(optimized_instructions)
    print(f"After fusion: {len(final_instructions)} instructions")
    
    print("\nFinal optimized instructions:")
    for i, instr in enumerate(final_instructions):
        print(f"  {i}: {instr}")
    
    return final_instructions

if __name__ == "__main__":
    print("Running matmul optimization pipeline...")
    with open('1_layer_nn_forward.json', 'r') as file:
        data = json.load(file)
    instrs = run_pipeline(data["functions"][0]["instrs"])
    print()

    # for instr in data["functions"][0]["instrs"]:
    #     print(instr)
    
    # Run optimization pipelinea
    #optimized_instructions = run_pipeline(MATMUL_JSON["instructions"])
    
    print("\nOptimization pipeline completed!") 
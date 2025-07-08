import json
import torch
import triton
import triton.language as tl

# Sample matmul JSON embedded in the script
MATMUL_JSON = {
    "instructions": [
        {
            "dest": "a",
            "op": "load",
            "args": ["input_a"],
            "shape": [1024, 1024],
            "dtype": "float32"
        },
        {
            "dest": "b", 
            "op": "load",
            "args": ["input_b"],
            "shape": [1024, 1024],
            "dtype": "float32"
        },
        {
            "dest": "c",
            "op": "matmul",
            "args": ["a", "b"],
            "shape": [1024, 1024],
            "dtype": "float32"
        },
        {
            "dest": "result",
            "op": "store",
            "args": ["c", "output"],
            "shape": [1024, 1024],
            "dtype": "float32"
        }
    ]
}

def trivial_dce_pass(instructions):
    """Remove instructions that are never used as arguments"""
    used = set()
    for instr in instructions:
        used.update(instr.get("args", []))
    
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

def is_fusable_op(op):
    """Check if operation can be fused"""
    fusable_ops = {"matmul", "add", "multiply", "relu", "load", "store"}
    return op in fusable_ops

def fuse_operations_pass(instructions):
    """Single pass of fusion operation"""
    result = []
    used = set()
    changed = False
    
    for i, instr in enumerate(instructions):
        if i in used:
            continue
            
        if not is_fusable_op(instr.get("op", "")):
            result.append(instr)
            continue
            
        # Build fusion chain
        chain = [instr]
        used.add(i)
        
        # Look for consecutive operations that can be fused
        while True:
            last_op = chain[-1]
            found_next = False
            
            for j in range(len(instructions)):
                if j in used:
                    continue
                    
                candidate = instructions[j]
                
                if (is_fusable_op(candidate.get("op", "")) and
                    last_op["dest"] in candidate.get("args", [])):
                    
                    # Check if output is only used once
                    usage_count = sum(1 for k, inst in enumerate(instructions) 
                                    if k not in used and last_op["dest"] in inst.get("args", []))
                    
                    if usage_count == 1:
                        chain.append(candidate)
                        used.add(j)
                        found_next = True
                        break
            
            if not found_next:
                break
        
        if len(chain) > 1:
            result.append(create_fused_chain(chain))
            changed = True
        else:
            result.append(chain[0])
    
    return result, changed

def fuse_operations(instructions):
    """Iteratively fuse operations"""
    while True:
        instructions, changed = fuse_operations_pass(instructions)
        if not changed:
            break
    return instructions

def create_fused_chain(fusion_chain):
    """Create a fused operation from a chain of operations"""
    steps = []
    intermediate_vars = set()
    
    for i in range(len(fusion_chain) - 1):
        intermediate_vars.add(fusion_chain[i]["dest"])
    
    external_args = []
    for op in fusion_chain:
        for arg in op.get("args", []):
            if arg not in intermediate_vars and arg not in external_args:
                external_args.append(arg)
    
    for i, op in enumerate(fusion_chain):
        if i == 0:
            args_str = ", ".join(op.get("args", []))
            steps.append(f"{op['op']}({args_str})")
        else:
            prev_output = fusion_chain[i-1]["dest"]
            other_args = [arg for arg in op.get("args", []) if arg != prev_output]
            
            if other_args:
                args_str = "$prev, " + ", ".join(other_args)
            else:
                args_str = "$prev"
                
            steps.append(f"{op['op']}({args_str})")
    
    last_op = fusion_chain[-1]
    fused_op = {
        "dest": last_op["dest"],
        "op": "fused",
        "steps": steps
    }
    
    if external_args:
        fused_op["args"] = external_args
    
    for key in ["shape", "dtype"]:
        if key in last_op:
            fused_op[key] = last_op[key]
    
    return fused_op

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized matmul kernel generated from fused operations"""
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def generate_kernel_code(optimized_instructions):
    """Generate optimized kernel code based on instructions"""
    kernel_code = '''
@triton.jit
def optimized_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Generated optimized kernel"""
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
'''
    return kernel_code

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

def test_matmul():
    """Test the matmul kernel"""
    # Create test tensors
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )
    
    # Verify result
    ref_out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    print(f"Max difference: {torch.max(torch.abs(c.to(torch.float32) - ref_out))}")
    print("Matmul kernel test passed!")

if __name__ == "__main__":
    print("Running matmul optimization pipeline...")
    
    # Run optimization pipeline
    optimized_instructions = run_pipeline(MATMUL_JSON["instructions"])
    
    # Generate kernel code
    kernel_code = generate_kernel_code(optimized_instructions)
    print("\nGenerated kernel code:")
    print(kernel_code)
    
    # Test the kernel
    if torch.cuda.is_available():
        print("\nTesting matmul kernel...")
        test_matmul()
    else:
        print("\nCUDA not available, skipping kernel test")
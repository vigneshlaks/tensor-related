import numpy as np

# Your example from 1_layer.json
input = np.array([1, 2])  # Shape [2] (or [1, 2] flattened)
delta = np.array([5, 4])  # Shape [2] (or [1, 2] flattened)

print("Batch size = 1 case:")
print(f"input shape: {input.shape}, values: {input}")
print(f"delta shape: {delta.shape}, values: {delta}")
print()

# Your Python code (line 70)
grad_outer = np.outer(input, delta)
print("Using np.outer(input, delta):")
print(grad_outer)
print(f"Shape: {grad_outer.shape}")
print()

# General formula
input_2d = input.reshape(1, 2)  # [1, 2]
delta_2d = delta.reshape(1, 2)  # [1, 2]
grad_matmul = input_2d.T @ delta_2d  # [2, 1] @ [1, 2] = [2, 2]
print("Using input.T @ delta:")
print(grad_matmul)
print(f"Shape: {grad_matmul.shape}")
print()

print("Are they equal?", np.allclose(grad_outer, grad_matmul))
print()

print("="*60)
print("Batch size = 2 case (why outer product breaks):")
print("="*60)

input_batch = np.array([[1, 2], [3, 4]])  # [2, 3] - 2 samples
delta_batch = np.array([[5, 4], [7, 6]])  # [2, 2] - 2 samples

print(f"input shape: {input_batch.shape}")
print(f"delta shape: {delta_batch.shape}")
print()

# Correct way
grad_correct = input_batch.T @ delta_batch  # [3, 2] @ [2, 2] = [3, 2]
print("Using input.T @ delta (correct):")
print(grad_correct)
print(f"Shape: {grad_correct.shape}")
print()

# Outer product (wrong for batches)
grad_outer_wrong = np.outer(input_batch, delta_batch)
print("Using np.outer(input, delta) (WRONG for batches):")
print(grad_outer_wrong)
print(f"Shape: {grad_outer_wrong.shape}")
print("This is wrong! Shape should be [2, 2], not [4, 4]")

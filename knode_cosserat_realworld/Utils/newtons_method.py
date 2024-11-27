import torch

# Ensure that the gradients are retained after the backward pass
torch.set_grad_enabled(True)

# Define the function for which you want to find the root
# Example function: f(x) = x^2 - 1
def f(x):
    return x**2 - 1

# Starting point for the Newton's method (initial guess)
x = torch.tensor([0.5], requires_grad=True)

# Set the convergence criteria
tolerance = 1e-5
max_iterations = 100

for iteration in range(max_iterations):
    # Forward pass: compute the function value at the current estimate
    y = f(x)
    
    # Backward pass: compute the gradient
    y.backward(retain_graph=True)
    
    # Check for convergence
    if abs(y.item()) < tolerance:
        print(f"Converged to {x.item()} in {iteration+1} iterations")
        break

    # Compute the update (Newton's method step)
    # For this simple case, we can use the gradient directly as the Hessian is 1
    x.data -= y.data / x.grad.data
    
    # Zero the gradients for the next iteration
    x.grad.data.zero_()
else:
    print(f"Did not converge within {max_iterations} iterations")

# At this point, x contains the root of the function

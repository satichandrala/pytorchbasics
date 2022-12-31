import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2 # THIS operations makes a computational graph -for each operation, we have imput and output and backpropagation - Forward and backward pass

print(y) # result : tensor([3.2630, 0.4205, 2.1874], grad_fn=<AddBackward0>)

z = y*y*2
#z=z.mean() normaly the last operation is  a scalar so we need to make a vector.
print(z)

#we need to multiply with a vector value but here z is a scalar
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
#z.backward() # CALCULATES THE GRADIENTS wrt x dz/dx  if no z.mean is applied, then z has also 1*3 vector
z.backward(v)
print(x.grad)


#    wHEN WE APPLY GRAD TO A vector, it gives error which means we apply grad only to scalars
# Traceback (most recent call last):
#  File "C:\Users\satic\miniconda3\envs\masterthesis\lib\runpy.py", line 197, in _run_module_as_main
#   return _run_code(code, main_globals, None,
#   File "C:\Users\satic\miniconda3\envs\masterthesis\lib\runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy\__main__.py", line 39, in <module>
#     cli.main()
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy/..\debugpy\server\cli.py", line 430, in main
#     run()
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy/..\debugpy\server\cli.py", line 284, in run_file
#     runpy.run_path(target, run_name="__main__")
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 321, in run_path
#     return _run_module_code(code, init_globals, run_name,
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 135, in _run_module_code
#     _run_code(code, mod_globals, init_globals,
#   File "c:\Users\satic\.vscode\extensions\ms-python.python-2022.20.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 124, in _run_code
#     exec(code, run_globals)
#   File "c:\Users\satic\Desktop\Uni_Bonn\Master_Thesis-UniBonn\code\pytorchbasics\usingautograd.py", line 16, in <module>
#     z.backward(v)
#   File "C:\Users\satic\miniconda3\envs\masterthesis\lib\site-packages\torch\_tensor.py", line 487, in backward
#     torch.autograd.backward(
#   File "C:\Users\satic\miniconda3\envs\masterthesis\lib\site-packages\torch\autograd\__init__.py", line 190, in backward
#     grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
#   File "C:\Users\satic\miniconda3\envs\masterthesis\lib\site-packages\torch\autograd\__init__.py", line 68, in _make_grads
#     raise RuntimeError("Mismatch in shape: grad_output["
#    RuntimeError: Mismatch in shape: grad_output[0] has a shape of torch.Size([3]) and output[0] has a shape of torch.Size([]).

# how to make pytorch not track the history of grad(n attribute), in training, updating weights should not be possible.

# x.requires_grad_(False)
# x.detach() # NEW TENSOR THAT DOESN#T RQUIRE GRADIENT

# or wrap in a with statement

# with torch.no_grad():
x.requires_grad_(False) # IN pytorch a trailing _(underscore) means the function will modify the variable in place
print(x)
#requires_grad=True
# first x value: tensor([-0.0436, -0.3668, -0.0522], requires_grad=True)
# after we set requires_grad to False; output: tensor([-0.0436, -0.3668, -0.0522])
y = x.detach() # WILL CREATE A new or same vector with old values but doesn't require grad
print(y)

with torch.no_grad():
    y = x+2
    print(y)
    # THE result has no gradient and thus preventing from tracking history in our computational graph
    # Whenever we call the backward function for a variable the gradient will accumulate to the .grad attributes and values could be summed up

    # dUMMY TRAINING EXAMPLE

weights= torch.ones(4, requires_grad=True)
#TRAINING LOOP
# Dummy operation which will simulate te model output
for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward() # GRADIENT

    print(weights.grad)

# EMPTY THE gradients  before starting to optimize

weights.grad.zero_()

# USING TORCH OPTIM and build

# optimizer = torch.optim.SGD(weights, lr = 0.01)
# optimizer.step()
# optimizer.zero_grad()

# z.backward()

# weights.grad.zero_()
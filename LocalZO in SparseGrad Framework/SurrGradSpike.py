import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    raise ('Sparse spike backpropagation requires CUDA available!')


# class SurrGradSpike(torch.autograd.Function):
#     scale = 100.0  # controls steepness of surrogate gradient

#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.zeros_like(input)
#         out[input > 0] = 1.0
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
#         return grad
        
        
# class Local2PtZO(torch.autograd.Function):
#     delta = 0.1
#     @staticmethod
#     def forward(ctx, input_):
#         ctx.save_for_backward(input_)
#         out = (input_ > 0).float()
#         return out


#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         abs_z = torch.abs(torch.randn(input_.size(), device=device, dtype=torch.float))
#         grad = grad_input*(torch.abs(input_)< abs_z*Local2PtZO.delta)*(abs_z/(2*Local2PtZO.delta))
        
#         return grad, None
        

class normal(torch.autograd.Function):
    delta = 0.05
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*torch.exp(-(input_**2)/(2*(normal.delta**2)))/(normal.delta*torch.sqrt(2*torch.tensor(np.pi, device=device, dtype=torch.float)))
        return grad, None    

class uniform(torch.autograd.Function):
    delta = 0.05
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp=(3-input_**2/uniform.delta**2)
        grad = grad_input*(temp>0)*temp/(4*np.sqrt(3)*uniform.delta)
        return grad, None   

class laplace(torch.autograd.Function):
    delta = 0.05
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*(torch.abs(input_)/laplace.delta + 1/np.sqrt(2))*torch.exp(-np.sqrt(2)*torch.abs(input_)/laplace.delta)/(2*laplace.delta)
        return grad, None   

class sigmoid(torch.autograd.Function):
    delta = 0.05
    k = np.sqrt(1/0.4262)/delta
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*sigmoid.k*torch.exp(-sigmoid.k*input_)/(1+torch.exp(-sigmoid.k*input_))**2
        return grad, None

class fsigmoid(torch.autograd.Function):
    k = 100.0
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(1+fsigmoid.k*torch.abs(input_))**2
        return grad, None        
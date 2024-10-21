import torch


def sgd(parameters, lr, momentum=0, dampening=0, weight_decay=0):
    for param in parameters:
        if param.grad is not None:
            param_t = param.data
            grad_t = param.grad.data

            if momentum != 0:
                if not hasattr(param, 'momentum_buffer'):
                    param.momentum_buffer = torch.zeros_like(param_t)
                buf = param.momentum_buffer
                buf.mul_(momentum).add_(grad_t, alpha=1 - dampening)
                grad_t = buf

            if weight_decay != 0:
                grad_t.add_(param_t, alpha=weight_decay)

            param_t.add_(grad_t, alpha=-lr)
import torch


def momentum_sgd(parameters, lr, momentum=0.9):
    for param in parameters:
        if param.grad is not None:
            param_t = param.data
            grad_t = param.grad.data

            if param.grad.is_sparse:
                raise RuntimeError('Momentum SGD does not support sparse gradients, please consider SparseAdam instead')

            state = param.state['momentum_buffer']
            if ((s := state) is None):
                s = param.state['momentum_buffer'] = torch.zeros_like(param_t)
                s.mul_(momentum).add_(grad_t, alpha=1 - momentum)
            else:
                s.mul_(momentum).add_(grad_t, alpha=1 - momentum)
            param_t.add_(s, alpha=-lr)
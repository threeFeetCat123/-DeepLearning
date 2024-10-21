import torch


def rmsprop(parameters, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.data
            if grad.is_sparse:
                raise RuntimeError('RMSprop does not support sparse gradients, please consider SparseAdam instead')

            state = param.state
            if len(state) == 0:
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(param.data)
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(param.data)
                if centered:
                    state['grad_avg'] = torch.zeros_like(param.data)

            state['step'] += 1

            square_avg = state['square_avg']
            square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
            avg = torch.sqrt(square_avg + eps)

            if weight_decay != 0:
                grad.add_(param.data, alpha=weight_decay)

            if momentum > 0:
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                param.data.add_(buf, alpha=-lr / avg)
            else:
                param.data.addcdiv_(grad, avg, value=-lr)
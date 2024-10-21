import torch


def adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

        state = param.state
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(param.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = betas
        state['step'] += 1

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
            # Use the max. for normalizing running avg. of gradient
            denom = state['max_exp_avg_sq'].sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        if weight_decay != 0:
            grad.add_(param.data, alpha=weight_decay)
        param.data.addcdiv_(exp_avg, denom, value=-step_size)
def sgd_adagrad(parameters, lr=0.01, eps=1e-10):
    for param in parameters:
        if param.grad is not None:
            # 计算梯度的平方
            sqr_grad = param.grad.data ** 2
            # 如果是第一次更新，初始化sqr，否则累加
            if not hasattr(param, 'sqr'):
                param.sqr = torch.zeros_like(param.data)
            param.sqr.add_(sqr_grad)

            # 计算自适应学习率
            adaptive_lr = lr / (torch.sqrt(param.sqr) + eps)
            # 更新参数
            param.data.add_(-adaptive_lr * param.grad.data)


# 下面是使用pytorch的示例使用
import torch
# 假设我们有一个简单的模型和一些模拟的数据
model = torch.nn.Linear(5, 3)  # 一个简单的线性模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用SGD作为基础优化器
loss_fn = torch.nn.MSELoss()

# 模拟一些数据
inputs = torch.randn(10, 5)
targets = torch.randn(10, 3)

# 前向传播
outputs = model(inputs)
loss = loss_fn(outputs, targets)

# 反向传播
loss.backward()

# 使用自定义的SGD和Adagrad结合的更新函数
sgd_adagrad(model.parameters(), lr=0.01)

# 注意：在实际使用中，通常不需要手动实现优化器的更新步骤，
# 因为PyTorch已经提供了优化器的实现。这里仅作为示例展示如何结合SGD和Adagrad。
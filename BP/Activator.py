import numpy as np
class ReLUActivator(object):
    def forward(self, weighted_input):
        return np.maximum(0, weighted_input)

    def backward(self, output):
        return (output > 0).astype(int)

# Sigmoid 激活函数
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

# tanh 激活函数
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
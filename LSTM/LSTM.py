import numpy as np

np.random.seed(42)
# ReLu 激活函数
class ReluActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0

# y = x 激活函数
class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

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


class LstmLayer(object):
    def __init__(self, input_width, state_width,
                 output_width, learning_rate):
        # 目前仅支持 state_width == output_width
        if(state_width != output_width):
            raise ValueError("state_width must be equal to output_width")
        self.input_width = input_width
        self.state_width = state_width
        self.output_width = output_width
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec(state_width)
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec(output_width)
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec(state_width)
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec(state_width)
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec(state_width)
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec(state_width)
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = (
            self.init_weight_mat())
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi = (
            self.init_weight_mat())
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo = (
            self.init_weight_mat())
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc = (
            self.init_weight_mat())

    # 初始化保存状态的向量
    def init_state_vec(self, dimension):
        state_vec_list = [np.zeros(
            (dimension, 1))]
        return state_vec_list

    # 初始化权重矩阵
    def init_weight_mat(self):
        Wh = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.output_width))
        Wx = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    # 计算门
    def calc_gate(self, x, Wx, Wh, b, activator):
        h = self.h_list[self.times - 1]  # 上次的LSTM输出
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate

    def forward(self, x):
        """
        H(t-1) 前一次输出结果的状态
        C(t-1) 前一次记忆单元的状态
        fg = sigmoid(Wf · [H(t-1), x] + bf )
        ig = sigmoid(Wi · [H(t-1), x] + bi)
        og = sigmoid(Wo · [H(t-1), x] + bo)
        ct = tanh(Wc · [H(t-1), x] + bc)
        C(t) = fg * C(t-1) + ig * ct
        H(t) = og * tanh[C(t)]
        根据式1-式6进行前向计算
        """
        self.times += 1
        # 遗忘门
        fg = self.calc_gate(x, self.Wfx, self.Wfh,
                            self.bf, self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wix, self.Wih,
                            self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Wox, self.Woh,
                            self.bo, self.gate_activator)
        self.o_list.append(og)
        # 输入的即时状态
        ct = self.calc_gate(x, self.Wcx, self.Wch,
                            self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct

        self.c_list.append(c)
        # 输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)

    def backward(self, x, expected_h):
        """
        实现LSTM训练算法
        """
        delta_h = self.h_list[self.times] - expected_h
        self.calc_delta(delta_h)
        self.calc_gradient(x)
        self.update()

    def update(self):
        """
        按照梯度下降，更新权重
        """
        self.Wfh -= self.learning_rate * self.Wfh_grad
        self.Wfx -= self.learning_rate * self.Wfx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Wih_grad
        self.Wix -= self.learning_rate * self.Wix_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Woh_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wch_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad

    def calc_delta(self, delta_h):
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta(self.output_width)  # 输出误差项
        self.delta_o_list = self.init_delta(self.state_width)  # 输出门误差项
        self.delta_i_list = self.init_delta(self.state_width)  # 输入门误差项
        self.delta_f_list = self.init_delta(self.state_width)  # 遗忘门误差项
        self.delta_ct_list = self.init_delta(self.state_width)  # 即时输出误差项

        # 保存预测值与实际值的误差
        self.delta_h_list[-1] = delta_h

        # 迭代计算每个时刻的误差项
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)

    def init_delta(self, dimension):
        """
        初始化误差项
        """
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros(
                (dimension, 1)))
        return delta_list

    def calc_delta_k(self, k):
        """
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        """
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k - 1]
        tanh_c = self.output_activator.forward(c)
        delta_h = self.delta_h_list[k]
        # 根据链式求导
        delta_o = (delta_h * tanh_c *
                   self.gate_activator.backward(og))
        delta_f = (delta_h * og *
                   (1 - tanh_c * tanh_c) * c_prev *
                   self.gate_activator.backward(fg))
        delta_i = (delta_h * og *
                   (1 - tanh_c * tanh_c) * ct *
                   self.gate_activator.backward(ig))
        delta_ct = (delta_h * og *
                    (1 - tanh_c * tanh_c) * ig *
                    self.output_activator.backward(ct))
        delta_h_prev = (
                np.dot(delta_o.transpose(), self.Woh) +
                np.dot(delta_i.transpose(), self.Wih) +
                np.dot(delta_f.transpose(), self.Wfh) +
                np.dot(delta_ct.transpose(), self.Wch)
        ).transpose()
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def init_weight_gradient_mat(self):
        """
        初始化权重矩阵
        """
        Wh_grad = np.zeros((self.state_width,
                            self.output_width))
        Wx_grad = np.zeros((self.state_width,
                            self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad

    def calc_gradient(self, x):
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad = (
            self.init_weight_gradient_mat())

        # 计算对上一次输出h的权重梯度
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Wfh_grad, bf_grad,
             Wih_grad, bi_grad,
             Woh_grad, bo_grad,
             Wch_grad, bc_grad) = (
                self.calc_gradient_t(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += np.dot(Wfh_grad, self.h_list[t - 1].transpose())
            self.bf_grad += bf_grad
            self.Wih_grad += np.dot(Wih_grad, self.h_list[t - 1].transpose())
            self.bi_grad += bi_grad
            self.Woh_grad += np.dot(Woh_grad, self.h_list[t - 1].transpose())
            self.bo_grad += bo_grad
            self.Wch_grad += np.dot(Wch_grad, self.h_list[t - 1].transpose())
            self.bc_grad += bc_grad

        # 计算对本次输入x的权重梯度
        xt = x.transpose()
        self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
        self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
        self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)

    # 计算每个时刻t权重的梯度
    # 源代码这里有误，以下代码已经修改了
    def calc_gradient_t(self, t):
        h_prev_trans = self.h_list[t - 1].transpose()
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev_trans)
        bf_grad = self.delta_f_list[t]
        Wih_grad = np.dot(self.delta_i_list[t], h_prev_trans)
        bi_grad = self.delta_i_list[t]
        Woh_grad = np.dot(self.delta_o_list[t], h_prev_trans)
        bo_grad = self.delta_o_list[t]
        Wch_grad = np.dot(self.delta_ct_list[t], h_prev_trans)
        bc_grad = self.delta_ct_list[t]
        return Wfh_grad, bf_grad, Wih_grad, bi_grad, \
            Woh_grad, bo_grad, Wch_grad, bc_grad

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec(self.state_width)
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec(self.output_width)
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec(self.state_width)
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec(self.state_width)
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec(self.state_width)
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec(self.state_width)


import numpy as np

def generate_data_set(sequence_length, num_sequences):
    """
    生成一组序列数据，用于测试 LSTM 网络。
    数据可以根据正弦函数生成，用于模拟连续时间序列。
    """
    x = []
    d = []
    for _ in range(num_sequences):
        sequence = np.sin(np.linspace(0, 10, sequence_length)) + np.random.normal(0, 0.01, sequence_length)
        x.append([np.array([[s]]) for s in sequence[:-1]])  # 输入是序列中前 n-1 项
        d.append(np.array([[sequence[-1]]]))  # 目标输出是序列最后一项
    return x, d

def improved_test_lstm():
    # 初始化 LSTM 层
    lstm = LstmLayer(1, 1, 1, 0.01)  # 输入宽度1，状态宽度1，学习率0.01

    # 生成模拟训练数据集（假设每个序列长度为10，总共训练1000个序列）
    sequence_length = 10
    num_sequences = 1000
    x_train, y_train = generate_data_set(sequence_length, num_sequences)

    # 训练参数
    epochs = 100  # 训练轮数
    print_every = 10  # 每50轮打印一次训练状态

    # 训练过程
    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_sequences):
            lstm.reset_state()

            # 前向传播
            for t in range(sequence_length - 1):
                lstm.forward(x_train[i][t])

            # 反向传播
            lstm.backward(x_train[i][-1], y_train[i])

            # 计算损失
            total_loss += np.mean(np.abs(y_train[i] - lstm.h_list[-1]))

        # 打印训练状态
        if epoch % print_every == 0:
            print(f'Epoch {epoch}, average loss: {total_loss / num_sequences}')

    # 评估模型
    test_x, test_y = generate_data_set(sequence_length, 1)
    lstm.reset_state()
    for t in range(sequence_length - 1):
        lstm.forward(test_x[0][t])
    predicted = lstm.h_list[-1]
    print(f'Predicted: {predicted}')
    print(f'Actual: {test_y[0][0]}')

    return lstm

# 调用改进的测试函数
improved_test_lstm()

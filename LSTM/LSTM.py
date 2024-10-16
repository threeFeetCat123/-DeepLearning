import numpy as np
from Activator import SigmoidActivator, TanhActivator


np.random.seed(42)
# ReLu 激活函数


class LstmLayer(object):
    def __init__(self, input_width, state_width, output_width, learning_rate):
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
        Wh = np.random.uniform(-5e-1, 5e-1,
                               (self.state_width, self.output_width))
        Wx = np.random.uniform(-5e-1, 5e-1,
                               (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    # 计算门
    def calc_gate(self, x, Wx, h, Wh, b, activator):
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
        fg = self.calc_gate(x, self.Wfx, self.h_list[self.times - 1], self.Wfh,
                            self.bf, self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wix, self.h_list[self.times - 1], self.Wih,
                            self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Wox, self.h_list[self.times - 1], self.Woh,
                            self.bo, self.gate_activator)
        self.o_list.append(og)
        # 输入的即时状态
        ct = self.calc_gate(x, self.Wcx, self.h_list[self.times - 1], self.Wch,
                            self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        # 输出
        h = og * self.output_activator.forward(c)
        assert h.shape == self.bc.shape, f"Shape mismatch: h.shape = {h.shape}, self.bc.shape = {self.bc.shape}"
        self.h_list.append(h)

    def backward(self, x, expected_h):
        """
        实现LSTM训练算法
        """
        # 确保在当前时间步的输出、期望输出和输入的形状相同
        assert self.h_list[self.times].shape == expected_h.shape, \
            f"Shape mismatch: self.h_list[self.times].shape = {self.h_list[self.times].shape}, " \
            f"expected_h.shape = {expected_h.shape},"

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
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
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


def generate_data_set(sequence_length, num_sequences, noise_level=0.02):
    """
    生成一组基于正弦波的三维序列数据，用于测试 LSTM 网络。
    输入是时间点，输出是这些时间点处的三个正弦波值。

    参数:
    - sequence_length: 每个序列的长度。
    - num_sequences: 序列的数量。
    - noise_level: 添加到输出的噪声水平。
    """
    x = np.zeros((num_sequences, sequence_length, 1))  # 输入数据，时间点
    d = np.zeros((num_sequences, sequence_length, 5))  # 目标输出，三个正弦波值
    t = np.linspace(0, 1, sequence_length * 100 + num_sequences).reshape(-1, 1)
    for i in range(num_sequences):
        # 为每个序列生成不同的间隔时间点, 生成1到100的整数
        random_int = np.random.randint(1, 100)
        # 生成三个正弦波信号，并添加噪声
        for j in range(sequence_length):
            id = i + j * random_int
            sine_wave1 = np.sin(2 * np.pi * t[id])  # 频率为1的正弦波
            sine_wave2 = np.sin(2 * np.pi * t[id])  # 频率为2的正弦波
            sine_wave3 = np.sin(2 * np.pi * t[id])  # 频率为3的正弦波
            sine_wave4 = np.sin(2 * np.pi * t[id])  # 频率为4的正弦波
            sine_wave5 = np.sin(2 * np.pi * t[id])  # 频率为5的正弦波
            # print("now data: ")
            # print(sine_wave1, sine_wave2, sine_wave3)
            # 将三个正弦波值作为输出向量的一个维度
            d[i, j, :] = np.array([
                sine_wave1 + np.random.normal(0, noise_level),
                sine_wave2 + np.random.normal(0, noise_level),
                sine_wave3 + np.random.normal(0, noise_level),
                sine_wave4 + np.random.normal(0, noise_level),
                sine_wave5 + np.random.normal(0, noise_level)
            ]).flatten()

            x[i, j, 0] = 2 * np.pi * t[id]

    return x, d

def test_lstmlayer():
    # 初始化 LSTM 层
    lstm = LstmLayer(1, 5, 5, 0.1)  # 输入宽度1，状态宽度1，学习率0.01

    # 生成模拟训练数据集（假设每个序列长度为10，总共训练1000个序列）
    sequence_length = 5
    num_sequences = 1000
    x_train, y_train = generate_data_set(sequence_length, num_sequences)
    print(x_train.shape)
    print(y_train.shape)
    # 训练参数
    epochs = 40  # 训练轮数
    print_every = 5  # 每10轮打印一次训练状态

    # 训练过程
    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_sequences):
            lstm.reset_state()

            # 前向传播
            for t in range(sequence_length - 1):
                input_x = np.array(x_train[i][t]).reshape(-1, 1)
                # print("x : ", np.array(x_train[i][t]).reshape(-1, 1))
                lstm.forward(input_x)
            # print("y : ", np.array(y_train[i][-1]).reshape(-1, 1))

            # 反向传播
            input_x = np.array(x_train[i][-1]).reshape(-1, 1)

            expected_y = np.array(y_train[i][-1]).reshape(-1, 1)
            lstm.backward(input_x, expected_y)

            # 计算损失
            total_loss += np.mean(np.abs(y_train[i][-1] - lstm.h_list[-1].T))

        # 打印训练状态
        if epoch % print_every == 0:
            print(f'Epoch {epoch}, average loss: {total_loss / num_sequences}\n')

    # 评估模型
    test_x, test_y = generate_data_set(sequence_length, num_sequences)
    lstm.reset_state()
    for i in range(1 , num_sequences, num_sequences // 5):
        lstm.reset_state()
        for t in range(sequence_length - 1):
            input_x = np.array(test_x[i][t]).reshape(-1, 1)
            lstm.forward(input_x)
        predicted = lstm.h_list[-1]
        print(f'Predicted: {predicted.T}')
        print(f'Actual: {test_y[i][-1]}\n\n')

    return lstm

# 调用改进的测试函数
test_lstmlayer()

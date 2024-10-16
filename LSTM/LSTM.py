import numpy as np
from LSTM_Layer import LstmLayer
import matplotlib.pyplot as plt

np.random.seed(42)

class LSTM(object):
    def __init__(self, input_width, state_width, output_width, num_layers, learning_rate):
        self.layers = []
        for _ in range(num_layers):
            # 创建指定宽度的 LSTM 层
            lstm_layer = LstmLayer(input_width, state_width, output_width, learning_rate)
            self.layers.append(lstm_layer)
            # 对于后续层，输入宽度将等于上一层的输出宽度
            input_width = output_width

    def forward(self, x):
        for layer in self.layers:
            # 使用当前层的输出作为下一层的输入
            layer.forward(x)
            x = layer.h_list[-1]

    def backward(self, x, expected_h):
        # 从最后一层的 delta 计算开始
        delta_h = self.layers[-1].h_list[-1] - expected_h
        self.layers[-1].calc_delta(delta_h)

        # 反向传播所有层
        for i in reversed(range(len(self.layers) - 1)):
            # 取当前层的 delta_h (这里的计算可能存在一定问题，需要数学推导证明一下)
            delta_h = np.dot(self.layers[i + 1].delta_h_list[-1].T, self.layers[i + 1].Wfh).T
            self.layers[i].calc_delta(delta_h)

        # Perform gradient calculations and update for all layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.calc_gradient(x)
            else:
                layer.calc_gradient(self.layers[i - 1].h_list[-1])
            layer.update()

    def reset_state(self):
        # 重置所有图层的状态
        for layer in self.layers:
            layer.reset_state()

    def train(self, x_train, y_train, epochs, print_every):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(x_train)):
                self.reset_state()
                # 每个序列的正向传播
                # print("new test: ")
                for t in range(len(x_train[i]) - 1):
                    input_x = np.array(x_train[i][t]).reshape(-1, 1)
                    # print(input_x)
                    self.forward(input_x)
                input_x = np.array(x_train[i][-1]).reshape(-1, 1)
                expected_y = np.array(y_train[i][-1]).reshape(-1, 1)
                # 最后预期输出的向后传播
                # print("expected_y:", expected_y)
                self.backward(input_x, expected_y)

                # 计算损失 （平均绝对误差）
                total_loss += np.mean(np.abs(y_train[i][-1] - self.layers[-1].h_list[-1].T))

            # 打印训练状态
            if epoch % print_every == 0:
                print(f'Epoch {epoch}, average loss: {total_loss / len(x_train)}')

    def predict(self, x):
        self.reset_state()
        for t in range(len(x)):
            self.forward(x[t])
        return self.layers[-1].h_list[-1]




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

def lstm_test():
    # 初始化 LSTM 网络
    input_width = 1
    state_width = 5
    output_width = 5
    num_layers = 5
    learning_rate = 0.02
    lstm = LSTM(input_width, state_width, output_width, num_layers, learning_rate)

    # 生成模拟训练数据集
    sequence_length = 5
    num_sequences = 1000
    x_train, y_train = generate_data_set(sequence_length, num_sequences)

    # 训练参数
    epochs = 100  # 训练轮数
    print_every = 10  # 每10轮打印一次训练状态

    # 训练过程
    lstm.train(x_train, y_train, epochs, print_every)

    test_num_sequences = 4
    # 生成测试数据集
    test_x, test_y = generate_data_set(sequence_length, num_sequences)
    for i in range(1, num_sequences, num_sequences // 5):
        lstm.reset_state()
        for t in range(sequence_length - 1):
            input_x = np.array(test_x[i][t]).reshape(-1, 1)
            lstm.forward(input_x)
        predicted = lstm.layers[-1].h_list[-1].T
        print(f'Predicted: {predicted}')
        print(f'Actual: {test_y[i][-1]}\n\n')

lstm_test()
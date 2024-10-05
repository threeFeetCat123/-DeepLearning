import numpy as np
import random


def create_dataset(num_samples=20, img_size=(28, 28)):
    # 创建一个空的数组来存储图像和标签
    X = np.zeros((num_samples * 10, img_size[0], img_size[1]))
    Y = np.zeros(num_samples * 10, dtype=int)

    # 为每个数字生成图像和标签
    for i in range(10):
        images = []
        for _ in range(num_samples):
            img = np.zeros(img_size)
            # 将数字写在图像中间
            start = img_size[0] // 2 - 4 + i
            for x in range(max(0, img_size[1] // 2 - 3), min(img_size[1], img_size[1] // 2 + 4)):
                img[start - 1, x:x + 4] = 1
                img[start - 2, x:x + 4] = 1
                img[start - 3, x - 1:x + 3] = 1
                img[start - 4, x] = 1
            images.append(img)
        X[i * num_samples: (i + 1) * num_samples, :, :] = np.array(images)
        Y[i * num_samples: (i + 1) * num_samples] = i

    return X, Y


# 创建数据集
X_train, Y_train = create_dataset()
X_test, Y_test = create_dataset()

# 将数据集保存为npz文件
np.savez('mnist_train.npz', X_train=X_train, Y_train=Y_train)
np.savez('mnist_test.npz', X_test=X_test, Y_test=Y_test)
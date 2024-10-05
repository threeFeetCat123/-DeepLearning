import numpy as np

def convolve2d(I, K):
    (h, w) = K.shape
    oH = I.shape[0] - h + 1
    oW = I.shape[1] - w + 1
    O = np.zeros((oH, oW))
    for i in range(oH):
        for j in range(oW):
            O[i, j] = np.sum(I[i:i + h, j:j + w] * K)
    return O

def relu(Z):
    return np.maximum(0, Z)

def maxpool(A, size):
    (h, w) = A.shape
    (sH, sW) = (h // size[0], w // size[1])
    O = np.zeros((sH, sW))
    for i in range(sH):
        for j in range(sW):
            O[i, j] = np.max(A[i * size[0]:(i + 1) * size[0], j * size[1]:(j + 1) * size[1]])
    return O

def softmax(Z):
    eZ = np.exp(Z - np.max(Z))
    return eZ / eZ.sum(axis=1, keepdims=True)

def fully_connected(A, W, b):
    return np.dot(A, W) + b

# Parameters
input_size = (28, 28)
conv_kernel_size = (5, 5)
num_filters = 10
hidden_dim = 500  # This can be used for a hidden layer if needed
output_dim = 10
learning_rate = 0.01
num_epochs = 10

# Initialize weights
conv_W = np.random.randn(num_filters, *conv_kernel_size) / 9
fc_W = np.random.randn(num_filters * 12 * 12, output_dim) / 1000  # Adjusted here
fc_b = np.zeros((output_dim,))  # Kept as output_dim
conv_b = np.zeros((num_filters,))

# Load data
data = np.load('mnist_train.npz')
X_train = data['X_train']
Y_train = data['Y_train']

data = np.load('mnist_test.npz')
X_test = data['X_test']
Y_test = data['Y_test']

X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Convert Y_batch to one-hot encoding
Y_train_one_hot = np.zeros((Y_train.size, output_dim))
Y_train_one_hot[np.arange(Y_train.size), Y_train] = 1

Y_test_one_hot = np.zeros((Y_test.size, output_dim))
Y_test_one_hot[np.arange(Y_test.size), Y_test] = 1

# Training
for epoch in range(num_epochs):
    for i in range(len(X_train) // 32):
        X_batch = X_train[i * 32:(i + 1) * 32]
        Y_batch = Y_train_one_hot[i * 32:(i + 1) * 32]

        # Forward pass
        conv_out = np.zeros((X_batch.shape[0], num_filters, 24, 24))
        for k in range(num_filters):
            for b in range(X_batch.shape[0]):
                conv_out[b, k] = relu(convolve2d(X_batch[b], conv_W[k]) + conv_b[k])

        # Max pooling across each filter
        pool_out = np.zeros((X_batch.shape[0], num_filters, 12, 12))
        for b in range(X_batch.shape[0]):
            for k in range(num_filters):
                pool_out[b, k] = maxpool(conv_out[b, k], (2, 2))

        pool_out_flat = pool_out.reshape(pool_out.shape[0], -1)  # Shape will be (32, num_filters * 12 * 12)

        fc_out = fully_connected(pool_out_flat, fc_W, fc_b)
        Z = relu(fc_out)
        output = softmax(Z)

        # Compute loss
        loss = -np.sum(Y_batch * np.log(output + 1e-8))  # Keep the epsilon for numerical stability

        # Backward pass
        dZ = output - Y_batch
        dW = np.dot(pool_out_flat.T, dZ)
        db = np.sum(dZ, axis=0)

        # Update weights
        fc_W -= learning_rate * dW
        fc_b -= learning_rate * db

    print(f'Epoch {epoch + 1}, Loss: {loss}')

# Evaluation
correct = 0
total = 0
for i in range(len(X_test) // 32):
    X_batch = X_test[i * 32:(i + 1) * 32]
    Y_batch = Y_test[i * 32:(i + 1) * 32]

    # Forward pass
    conv_out = np.zeros((X_batch.shape[0], num_filters, 24, 24))
    for k in range(num_filters):
        for b in range(X_batch.shape[0]):
            conv_out[b, k] = relu(convolve2d(X_batch[b], conv_W[k]) + conv_b[k])

    # Max pooling across each filter
    pool_out = np.zeros((X_batch.shape[0], num_filters, 12, 12))
    for b in range(X_batch.shape[0]):
        for k in range(num_filters):
            pool_out[b, k] = maxpool(conv_out[b, k], (2, 2))

    pool_out_flat = pool_out.reshape(pool_out.shape[0], -1)

    fc_out = fully_connected(pool_out_flat, fc_W, fc_b)
    Z = relu(fc_out)
    output = softmax(Z)

    predictions = np.argmax(output, axis=1)
    total += Y_batch.shape[0]
    correct += np.sum(predictions == Y_batch)

print(f'Accuracy: {100 * correct / total}%')

import numpy as np
import os
import cv2
import random

# Đọc dữ liệu từ thư mục Train
train_dir = "522H0148_Traffic_detection/Train"
train_images = []
train_labels = []

# Lấy tất cả các ClassId từ tên thư mục
class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_ids = {cls: idx for idx, cls in enumerate(class_dirs)}  # Tạo ánh xạ ClassId -> index

# Đọc ảnh từ tập Train
for cls in class_dirs:
    path = os.path.join(train_dir, cls)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path)
        if img_array is None:
            continue
        img_resized = cv2.resize(img_array, (64, 64))
        train_images.append(img_resized)
        train_labels.append(class_ids[cls])  # Lấy index thay vì ClassId gốc

# Chuyển dữ liệu thành numpy array và chuẩn hóa
train_images = np.array(train_images).astype('float32') / 255.0
train_labels = np.array(train_labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
def train_test_split(images, labels, test_size=0.25):
    total_samples = len(images)
    test_size = int(total_samples * test_size)
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    x_train = images[train_indices]
    x_test = images[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels)

# Chuyển nhãn sang dạng one-hot
def to_categorical(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train = to_categorical(y_train, len(class_ids))
y_test = to_categorical(y_test, len(class_ids))

# Kiểm tra thông tin
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", y_test.shape)
print("Shape of y_test:", y_test.shape)


# Các lớp CNN, MaxPooling, và Dense
class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding='same'):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(output_channels)

    def forward(self, input):
        self.input = input
        pad_h, pad_w = 0, 0
        if self.padding == 'same':
            # Tính padding cho chiều cao và chiều rộng
            pad_h = (self.kernel_size - 1) // 2
            pad_w = (self.kernel_size - 1) // 2
            if input.ndim == 3:
                input = np.expand_dims(input, axis=0)  # Add batch dimension
            input = np.pad(input, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                           mode='constant', constant_values=0)

        out_h = (input.shape[1] - self.kernel_size + 2 * pad_h) // self.stride + 1
        out_w = (input.shape[2] - self.kernel_size + 2 * pad_w) // self.stride + 1
        output = np.zeros((input.shape[0], out_h, out_w, self.output_channels))

        for i in range(out_h):
            for j in range(out_w):
                for c in range(self.output_channels):
                    # Lấy region của ảnh đầu vào
                    region = input[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]

                    # Kiểm tra lại kích thước của region
                    region_h = region.shape[1]
                    region_w = region.shape[2]

                    # Nếu chiều của region không khớp với kernel_size, ta bỏ qua hoặc điều chỉnh
                    if region_h != self.kernel_size or region_w != self.kernel_size:
                        print(f"Skipping region due to incompatible size: {region.shape}")
                        continue  # Bỏ qua region này nếu không có kích thước đúng

                    # Làm phẳng region và weights
                    region_flat = region.reshape(-1, self.input_channels * self.kernel_size * self.kernel_size)
                    weights_flat = self.weights[c, :, :, :].reshape(-1)

                    # Tính toán giá trị trong output
                    output[:, i, j, c] = np.dot(region_flat, weights_flat) + self.biases[c]
        
        return output



class Dense:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        self.activation = activation

    def forward(self, input):
        if input.ndim > 1:
            self.input = input.reshape(input.shape[0], -1)
        else:
            self.input = input.flatten()
        output = np.dot(self.input, self.weights) + self.biases
        if self.activation == 'relu':
            return np.maximum(0, output)
        elif self.activation == 'softmax':
            exp_values = np.exp(output - np.max(output))
            return exp_values / np.sum(exp_values)
        else:
            return output

    def backpropagate(self, d_output, learning_rate=0.01):
        d_input = np.dot(d_output, self.weights.T).reshape(self.input.shape)
        d_weights = np.outer(self.input, d_output)
        d_biases = d_output

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input


class MaxPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        pool_h, pool_w = self.pool_size
        out_h = (input.shape[0] - pool_h) // self.stride + 1
        out_w = (input.shape[1] - pool_w) // self.stride + 1
        output = np.zeros((out_h, out_w, input.shape[2]))

        for i in range(out_h):
            for j in range(out_w):
                for c in range(input.shape[2]):
                    region = input[i*self.stride:i*self.stride+pool_h, j*self.stride:j*self.stride+pool_w, c]
                    output[i, j, c] = np.max(region)
        return output


# Định nghĩa mô hình CNN
def define_model():
    model = [
        Conv2D(3, 64, 3),
        MaxPooling2D((2, 2), stride=2),
        Conv2D(64, 128, 3),
        MaxPooling2D((2, 2), stride=2),
        Conv2D(128, 256, 3),
        MaxPooling2D((2, 2), stride=2),
        Dense(256*6*6, 128, activation='relu'),
        Dense(128, len(class_ids), activation='softmax')
    ]
    return model

def predict(model, image):
    out = image
    for layer in model:
        out = layer.forward(out)
    return np.argmax(out)

# Hàm tính Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

# Hàm huấn luyện mô hình
def train(model, x_train, y_train, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(x_train)):
            x_batch = x_train[i:i+1]
            y_batch = y_train[i:i+1]

            out = x_batch
            for layer in model:
                out = layer.forward(out)

            loss = cross_entropy_loss(y_batch, out)
            total_loss += loss

            d_output = out - y_batch
            for layer in reversed(model):
                d_output = layer.backpropagate(d_output, learning_rate)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(x_train)}")

# Huấn luyện mô hình
model = define_model()
train(model, x_train, y_train, epochs=20, learning_rate=0.01)

# Hàm tính độ chính xác
def accuracy(model, x_test, y_test):
    correct_predictions = 0
    total_samples = len(x_test)
    for i in range(total_samples):
        x_sample = x_test[i:i+1]
        y_true = y_test[i:i+1]

        out = x_sample
        for layer in model:
            out = layer.forward(out)

        predicted_label = np.argmax(out)
        true_label = np.argmax(y_true)

        if predicted_label == true_label:
            correct_predictions += 1

    return correct_predictions / total_samples

# Đánh giá độ chính xác
acc = accuracy(model, x_test, y_test)
print(f"Accuracy: {acc * 100:.2f}%")

# MNIST Dataset - Digit Classification
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist

(train_x_3D, train_y), (test_x_3D, test_y) = mnist.load_data()

train_x = train_x_3D.reshape(60000, 784)
train_x = train_x.T / 255
train_y = train_y.T

test_x = test_x_3D.reshape(10000, 784)
test_x = test_x.T / 255
test_y = test_y.T

def init_params():
	W1 = np.random.rand(10, 784) - 0.5
	b1 = np.random.rand(10, 1) - 0.5
	W2 = np.random.rand(10, 10) - 0.5
	b2 = np.random.rand(10, 1) - 0.5
	return W1, b1, W2, b2

def ReLU(Z):
	return np.maximum(0, Z)

def softmax(Z):
	A = np.exp(Z) / sum(np.exp(Z))
	return A

def forward_prop(W1, b1, W2, b2, X):
	Z1 = W1.dot(X) + b1
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = softmax(Z2)
	return Z1, A1, Z2, A2

def one_hot(Y):
	one_hot_Y = np.zeros((Y.size, Y.max() + 1))
	one_hot_Y[np.arange(Y.size), Y] = 1
	one_hot_Y = one_hot_Y.T
	return one_hot_Y

def ReLUPrime(Z):
	return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
	m = Y.size
	one_hot_Y = one_hot(Y)
	dZ2 = A2 - one_hot_Y
	dW2 = (1 / m) * dZ2.dot(A1.T)
	db2 = (1 / m) * np.sum(dZ2)
	dZ1 = W2.T.dot(dZ2) * ReLUPrime(Z1)
	dW1 = (1 / m) * dZ1.dot(X.T)
	db1 = (1 / m) * np.sum(dZ1)
	return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	W1 = W1 - alpha * dW1
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2
	return W1, b1, W2, b2

def get_predictions(A2):
	return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
	print(predictions, Y)
	return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
	W1, b1, W2, b2 = init_params()
	for i in range(iterations):
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
		W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		if i % 10 == 0:
			print("Iteration: ", i)
			print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
	return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_x, train_y, 200, 0.1)

# Compare Result to Expected

def make_predictions(X, W1, b1, W2, b2):
	_, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
	predictions = get_predictions(A2)
	return predictions

def test_prediction(W1, b1, W2, b2):
	index = np.random.randint(0, train_y.size)
	current_image = train_x[:, index, None]
	prediction = make_predictions(train_x[:, index, None], W1, b1, W2, b2)
	label = train_y[index]
	print("Prediction: ", prediction)
	print("Label: ", label)
	
	current_image = current_image.reshape((28,28)) * 255
	plt.gray()
	plt.imshow(current_image, interpolation='nearest')
	plt.show()

# Test Predictions

test_prediction(W1, b1, W2, b2)
test_prediction(W1, b1, W2, b2)
test_prediction(W1, b1, W2, b2)

print('\n')

# Accuracy in Test Set

test_predictions = make_predictions(test_x, W1, b1, W2, b2)
print("Accuracy: ", get_accuracy(test_predictions, test_y))

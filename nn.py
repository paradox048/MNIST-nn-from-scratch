import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
row, col = data.shape
np.random.shuffle(data)

# Splitting data into train and test
# Testing Data
data_dev = data[0:1000].T
Y_dev = data_dev[0] # First row is labels
X_dev = data_dev[1:col]/255.0 # Other rows are data

#Training Data
data_train = data[1000:row].T
Y_train = data_train[0]
X_train = data_train[1:col]/255.0
_,m_train = X_train.shape # m_train is number of examples in training set

def main():
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, .1)
    test_predictions(0, W1, b1, W2, b2)
    test_predictions(1, W1, b1, W2, b2)
    test_predictions(2, W1, b1, W2, b2)
    test_predictions(3, W1, b1, W2, b2)
    
    
    
def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    
    return W1, b1, W2, b2

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = np.maximum(Z1,0) # ReLU function
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

# One hot encoding
def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max()+1)) # Y.max() + 1 is number of classes
    one_hot_Y[np.arange(Y.size), Y] = 1 # Set the correct column to 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y 

def deriv_ReLU(Z):
    return Z > 0

# Cost function
# backward propagation
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    
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
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_predictions(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()
import mnist_handwritten_digits
import numpy as np

# each input graphic has 28x28 pixels and the output is a number between 0
# and 9, i.e. one node for each number in the output layer
nn = mnist_handwritten_digits.neuralNet(28**2, 28**2, 10)

# read in training set
data_file = open("mnist_train_100.csv", "r")
data_training = data_file.readlines()
data_file.close()

# read in test set
data_file = open("mnist_test_10.csv", "r")
data_test = data_file.readlines()
data_file.close()

# define targets
targets = {}
onodes = 10
for node in range(onodes):
    target = np.zeros(onodes) + 0.01
    target[node] = 0.99
    targets[str(node)] = target

# train the neural net with training set
for data in data_training:
    all_values = data.split(',')
    # optional for printing: resize the input array to a 28x28 matrix
    #image_array = np.asfarray(all_values[1:]).reshape((28,28))
    scaled_input = nn.scale_input(all_values)
    nn.train(scaled_input, targets[all_values[0]])

# test the neural net with test set and print the calculated output
for data in data_test:
    test_values = data.split(',')
    scaled_input = nn.scale_input(test_values)
    print("Tatsächlicher Wert: " + test_values[0])
    print("Vorhergesagter Wert:")
    print(np.around(nn.query(scaled_input), 3))

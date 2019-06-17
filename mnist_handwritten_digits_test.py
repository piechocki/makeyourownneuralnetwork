import mnist_handwritten_digits
import numpy as np
import scipy.ndimage
import os

# - each input graphic has 28x28 pixels that equals the number of nodes in the
# input layer
# - the hidden layer should have a number of nodes between the input and
# output layer that the network is able to reduce and compress the information
# given as output by the previous layer, but it must not be to low because
# then there is not enough capacity for learning, on the other side the
# calculation lasts to long if the number of nodes in the hidden layer is high
# - the output is a number between 0 and 9, i.e. one node for each valid output
# number in the output layer
# - learning rate (alpha) can be a number between 0 and 1, but the smaller the
# alpha, the less (slower) the current weights are corrected by the delta
# (correction of the weights by the gradient descent), high values for alpha
# can force an 'overshooting' effect, low values slow down the step range of
# the gradient descent when finding the minimum of the error function (a good
# alpha can be found by 'trial and error')
nn = mnist_handwritten_digits.neuralNet(28**2, 200, 10, 0.01)

# optional: define a path where data files are located
path_to_data = ".\\mnist_dataset"
epochs = 10
small = False
rotate_train_images = True
rotation_angle = 10
filename_training = "mnist_train_100.csv" if small else "mnist_train.csv"
filename_test = "mnist_test_10.csv" if small else "mnist_test.csv"

# read in training set
data_file = open(os.path.join(path_to_data, filename_training), "r")
data_training = data_file.readlines()
data_file.close()

# read in test set
data_file = open(os.path.join(path_to_data, filename_test), "r")
data_test = data_file.readlines()
data_file.close()

# define target vectors
targets = {}
onodes = 10
for node in range(onodes):
    target = np.zeros(onodes) + 0.01
    target[node] = 0.99
    targets[str(node)] = target

# train the neural net with training set with mutiple epochs
# the more epochs the more the machine can learn but that holds the danger
# of overfitting (machine is used to training data and can't handle unknown
# test data anymore)
# the higher the number of epochs is, the lower the learning rate can be set
# (if we go much more steps within the gradient descent, the range per step
# can be shorter therefore)
for _ in range(epochs):
    for record in data_training:
        training_data = record.split(',')
        # optional for printing: resize the input array to a 28x28 matrix
        #image_array = np.asfarray(training_data[1:]).reshape((28,28))
        scaled_input = nn.scale_input(training_data)
        nn.train(scaled_input, targets[training_data[0]])
        # optional: train the net with copies of the input images that are
        # rotated by 10 degrees in both directions
        if rotate_train_images:
            # rotation by 10 degrees anticlockwise
            inputs_plus10_img = scipy.ndimage.interpolation.rotate(
                scaled_input.reshape(28,28), rotation_angle, cval=0.01,
                reshape=False)
            # rotation by 10 degrees clockwise
            inputs_minus10_img = scipy.ndimage.interpolation.rotate(
                scaled_input.reshape(28,28), -rotation_angle, cval=0.01,
                reshape=False)
            nn.train(inputs_plus10_img.reshape(784),
                targets[training_data[0]])
            nn.train(inputs_minus10_img.reshape(784),
                targets[training_data[0]])

# test the neural net with test set and print the calculated output
scorecard = []
for record in data_test:
    test_data = record.split(',')
    scaled_input = nn.scale_input(test_data)
    predicted_label = np.argmax(nn.query(scaled_input))
    true_label = int(test_data[0])
    scorecard.append(1 if predicted_label == true_label else 0)

print("Accuracy of " + str(sum(scorecard) / len(scorecard)))

# save weights for later use
nn.save_weights()

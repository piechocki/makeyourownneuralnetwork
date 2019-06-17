import numpy as np
import scipy.special
import os
from tempfile import TemporaryFile

class neuralNet:

    def __init__(self, inputnodes, hiddennodes, outputnodes,
    learningrate = 0.1):
        # save number of nodes and the learning rate in the class
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate # aka alpha

        # define a class wide sigmoid function, here a logistical function
        # with y=1/(1+e^(-x))
        self.activation_function = lambda x: scipy.special.expit(x)

        # get the weight matrices with initial values
        # theory says:
        # choose a random number within +-1/sq.root(number of outgoing nodes)
        self.w_ih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                     (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, pow(self.onodes, -0.5),
                                     (self.onodes, self.hnodes))
        # alternatively you can choose a random number within +-0.5
        # self.w_ih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.w_ho = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.scale_input = lambda x: (np.asfarray(x[1:]) / 255.0 * 0.99) + 0.01
    
    def train(self, inputs_list, targets_list):
        # convert input and target lists into column vectors
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate final outputs forwards layer by layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # calculate errors backwards layer by layer
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        # update weights of each weight matrix
        self.w_ho += self.lr * np.dot((output_errors * final_outputs *
            (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.w_ih += self.lr * np.dot((hidden_errors * hidden_outputs *
            (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        # convert input list into a column vector
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate final outputs forwards layer by layer
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_ouputs = self.activation_function(final_inputs)

        return final_ouputs

    def save_weights(self, path = ""):
        np.save(os.path.join(path, "w_ih.npy"), self.w_ih)
        np.save(os.path.join(path, "w_ho.npy"), self.w_ho)

    def load_weights(self, path = ""):
        w_ih = np.load(os.path.join(path, "w_ih.npy"))
        w_ho = np.load(os.path.join(path, "w_ho.npy"))
        if w_ih.shape == self.w_ih.shape and \
           w_ho.shape == self.w_ho.shape:
            self.w_ih = w_ih
            self.w_ho = w_ho
        else:
            raise ValueError("Dimensions of neuralNet object don't equal to" \
                " dimensions of loaded weights. Please check the number of" \
                " nodes in each layer of the created object.")

import mnist_handwritten_digits
import numpy as np
import scipy.ndimage
import os
from random import randint

nn = mnist_handwritten_digits.neuralNet(9, 6, 2, 0.1)

def scale_input(record):
    scaled_record = []
    # age
    scaled_record.append(((int(record[0]) - 18) / (75.0 - 18) * 0.99) + 0.01)
    # sex
    scaled_record.append(0.99 if record[1] == "male" else 0.01)
    # job
    scaled_record.append((int(record[2]) / 3.0 * 0.99) + 0.01)
    # housing
    scaled_record.append({"free": 0.01, "own": 0.5, "rent": 0.99}[record[3]])
    # saving accounts
    scaled_record.append({"NA": 0.01, "little": 0.25, "moderate": 0.5,
        "quite rich": 0.75, "rich": 0.99}[record[4]])
    # checking account
    scaled_record.append({"NA": 0.01, "little": 0.33, "moderate": 0.66,
        "rich": 0.99}[record[5]])
    # credit amount
    scaled_record.append((int(record[6]) / 20000.0 * 0.99) + 0.01)
    # duration
    scaled_record.append((int(record[7]) / 75.0 * 0.99) + 0.01)
    # purpose
    scaled_record.append({"radio/TV": 0.01, "education": 0.14,
        "furniture/equipment": 0.28, "car": 0.42, "business": 0.56,
        "domestic appliances": 0.7, "repairs": 0.84,
        "vacation/others": 0.99}[record[8]])

    return scaled_record

# optional: define a path where data files are located
path_to_data = ".\\Kaggle"
train_ratio = 0.7 # ratio of train data in relation to the whole dataset
epochs = 50
filename = "german_credit_data.csv"

# read all data (train and test)
data_file = open(os.path.join(path_to_data, filename), "r")
data_raw = data_file.readlines()[1:] # skip first row that is a header
data_file.close()

# split data into train and test sets randomly
number_tests = round(len(data_raw) * (1 - train_ratio))
data_test = []
data_training = []
indexes = []
for _ in range(number_tests):
    while True:
        index = randint(0, len(data_raw) - 1)
        if index not in indexes:
            data_test.append(data_raw[index])
            indexes.append(index)
            break
for index in range(len(data_raw)):
    if index not in indexes:
        data_training.append(data_raw[index])

# define target vectors
targets = {}
for node in ["good", "bad"]:
    target = np.zeros(2) + 0.01
    target[0 if node == "good" else 1] = 0.99
    targets[node] = target

# load weights if net is already trained
nn.load_weights(path_to_data)

# # train the neural net with training set with mutiple epochs
# for _ in range(epochs):
#     for record in data_training:
#         training_data = record.split(';')
#         scaled_input = scale_input(training_data)
#         nn.train(scaled_input, targets[training_data[9][:-1]])

# test the neural net with test set and print the calculated output
scorecard = []
for record in data_test:
    test_data = record.split(';')
    scaled_input = scale_input(test_data)
    predicted_label = np.argmax(nn.query(scaled_input)) # predict credit risk
    true_label = 0 if test_data[9][:-1] == "good" else 1
    scorecard.append(1 if predicted_label == true_label else 0)

accuracy = sum(scorecard) / len(scorecard)
print("Accuracy of " + str(accuracy))

# save weights for later use if the accuracy is sufficient
if accuracy > 0.75:
    nn.save_weights(path_to_data)

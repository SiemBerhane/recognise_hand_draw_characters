# read and sort data
# apply ml algorithm 
import numpy as np

from mlxtend.data import loadlocal_mnist
from neural_network import Network

def train_model(train_data_filename, train_label_filename, epochs):
    x_train, y_train = loadlocal_mnist(train_data_filename, train_label_filename)
    #y_train = vectorise_labels(y_train) 
    x_train = x_train/255
    network = Network(x_train, y_train, 0, num_output_neurons=10, num_hidden=30, learning_rate=3, batch_size=10)
    for i in range(epochs):
        network.begin_training(input_data=x_train, training_labels=y_train, num_output_neurons=10, num_hidden=30, learning_rate=3, batch_size=10, flatten=True)
    network.save_model('recognise_hand_draw_characters/git/models/weights_test.pickle', 'recognise_hand_draw_characters/git/models/biases_test.pickle')

def vectorise_labels(labels):
    vect_labels = []
    for label in labels:
        vect = np.zeros((10,1))
        vect[label] = 1
        vect_labels.append(vect)

    return vect_labels

def test_model(test_data_filename, test_label_filename, w_filename, b_filename):
    x_test, y_test = loadlocal_mnist(test_data_filename, test_label_filename)
    n = Network(x_test, y_test, 1, batch_size=10, w_filename=w_filename, b_filename=b_filename)



train_model('recognise_hand_draw_characters/git/data/train-images.idx3-ubyte', 
'recognise_hand_draw_characters/git/data/train-labels.idx1-ubyte', 10)

test_model('recognise_hand_draw_characters/git/data/t10k-images.idx3-ubyte', 
'recognise_hand_draw_characters/git/data/t10k-labels.idx1-ubyte', 
'recognise_hand_draw_characters/git/models/weights_test.pickle',
'recognise_hand_draw_characters/git/models/biases_test.pickle')




# read and sort data
# apply ml algorithm 
import numpy as np

from mlxtend.data import loadlocal_mnist
from neural_network import Network

def load_data(train_data_filename, train_label_filename, test_data_filename, test_label_filename):
    x_train, y_train = loadlocal_mnist(train_data_filename, train_label_filename)
    x_test, y_test = loadlocal_mnist(test_data_filename, test_label_filename)
    
    y_train = vectorise_labels(y_train)
    y_test = vectorise_labels(y_test)

    print(np.shape(x_train), np.shape(y_train))

    network = Network(x_train, 10, y_train, num_hidden=30, learning_rate=3, batch_size=10)

    network.save_model('recognise_hand_draw_characters/git/models/weights.pickle', 'recognise_hand_draw_characters/git/models/biases.pickle')

def vectorise_labels(labels):
    vect_labels = []
    for label in labels:
        vect = np.zeros((10,1))
        vect[label] = 1
        vect_labels.append(vect)

    return vect_labels

load_data('recognise_hand_draw_characters/git/data/train-images.idx3-ubyte', 
'recognise_hand_draw_characters/git/data/train-labels.idx1-ubyte', 
'recognise_hand_draw_characters/git/data/t10k-images.idx3-ubyte', 
'recognise_hand_draw_characters/git/data/t10k-labels.idx1-ubyte')

import numpy as np
import pickle
import random

class Network:

    # input should be list of matrices
        # flat_input is a 2d array. An array of each input 
    def __init__(self, input_data, num_output_neurons, training_labels, num_hidden=None, flatten=False, batch_size=16, learning_rate=0.5):
        # if train:
        #     self.__begin_training(input_data, num_output_neurons, training_labels, num_hidden, flatten, batch_size, learning_rate)
        if flatten:
            input_data = self.__flatten_input(input_data) # Each item in list is an individual image

        batched_training_data = self.__create_batches(input_data, batch_size)

        batched_labels = self.__create_batches(training_labels, batch_size)

        if num_hidden == None:
            num_hidden = self.__calc_hidden_neurons(batched_training_data[0][0].shape[0], num_output_neurons)

        __w1 = self.__weight_shape(batched_training_data[0][0].shape[0], num_hidden)
        __w2 = self.__weight_shape(num_hidden, num_output_neurons)
        self.__weights = [__w1, __w2]

        b1 = self.__bias_shape(__w1)
        b2 = self.__bias_shape(__w2)
        self.__biases = [b1, b2]

        # Tweak to work w batched data 
        # For loop and pass each batch as input
        for i in range(len(batched_training_data)):
            # 1st item is list of dC/dw, 2nd item is list of dC/db
            output_derivatives = self.__calc_w_b_derivatives(batched_training_data[i], self.__weights, self.__biases, batched_labels[i])
            self.__weights = self.__stochastic_grad_descent(self.__weights, learning_rate, output_derivatives[0])
            self.__biases = self.__stochastic_grad_descent(self.__biases, learning_rate, output_derivatives[1])
            print(f"Batch {i}/{len(batched_training_data)} is done ({i/len(batched_training_data) * 100}%)")
        

        # save weights & biases


    def save_model(self, w_filename, b_filename):
        with open(w_filename, 'wb') as file: 
            pickle.dump(self.__weights, file)

        with open(b_filename, 'wb') as file: 
            pickle.dump(self.__biases, file)

    #def __begin_training(self, input_data, num_output_neurons, training_labels, num_hidden=None, flatten=False, batch_size=16, learning_rate=0.5):
    

    def test_model(self, test_data, labels, w_filename, b_filename, batch_size):
        with open(w_filename, 'rb') as file: 
            weights = pickle.load(file)

        with open(b_filename, 'rb') as file: 
            biases = pickle.load(file)

        batched_test_data = self.__create_batches(test_data, batch_size)
        batched_labels = self.__create_batches(labels, batch_size)

        #for i in range(len(batched_test_data)):


    def __create_batches(self, input, batch_size):
        '''Returns a 3D list of batches. \n
        First layer is used to access each batch,
        2nd layer used to access each input in a batch,
        3rd layer used to access each piece of data in an input e.g. individual pixels of an img'''

        #num_of_batches = len(input) // batch_size
        b = []
        list_of_b = []
        for i in range(len(input)):
            b.append(input[i])

            # if the loop reaches the last item in the list 
            # or the number of items in a batch has been reached
            if (i % batch_size == 0 and i != 0) or i == len(input) - 1:
                list_of_b.append(b)
                b = []
            
        return list_of_b



    def __calc_w_b_derivatives(self, input, w, b, input_labels):
        # looping through each input in a given batch
        w_der = [] # indexing is back to front compared to w & b
        b_der = []

        for i in range(len(input)):
            w_der_batch = []
            b_der_batch = []
            input[i] = input[i].reshape(len(input[i]), 1) # Shape of input w/o this = (len, ). This line changes it to: (len, 1)
            # Initial shape caused problems with the shape of the outputs

            a = [input[i]] # each item will contain a list of activitions for each neuron in a specific layer

            # feeds forward
            for j in range(len(w)): # Loops over each layer l and calculates the activation of layer l+1
                a.append(self.__sigmoid_func(w[j], a[j], b[j]))

            # calculates derivatives for output layer
            cost_der = self.__calc_cost_derivative(input_labels[i], a[-1])
            e_l = self.__calc_output_error(cost_der, w[-1], b[-1], a[-2])
            w_der_batch.append(self.__w_jk_derivative(w[-1], a[-2], e_l))
            b_der_batch.append(e_l)

            # backpropagation
            for k in range(len(w)):
                # -2-k is the index for the current layer
                index = len(w) - k - 1
                if index == 0:
                    break
                
                e_l = self.__calc_error_for_layer(e_l, w[-1-k], w[-2-k], b[-2-k], a[-3-k])
                w_der_batch.append(self.__w_jk_derivative(w[-2-k], a[-3-k], e_l))
                b_der_batch.append(e_l)

            w_der.append(w_der_batch)
            b_der.append(b_der_batch)

        return (w_der, b_der)

    # indexing of derivatives layers is back to front compared to init_vals
    #indexing follows this convention: layer, j(neuron in layer l+1), k(neuron in layer l)
    def __stochastic_grad_descent(self, init_vals, learning_rate, derivatives):
        final_vals = []
        for l in range(len(init_vals)): # l for layer
            new_vals_for_layer = []
            for j in range(init_vals[l].shape[0]): # rows
                new_row = []
                for k in range(init_vals[l].shape[1]): # columns
                    sum_der = 0
                    for batch in derivatives:
                        sum_der += batch[-1-l][j][k] 

                    new_row.append(init_vals[l][j][k] - (learning_rate/len(derivatives) * sum_der))
                
                new_vals_for_layer.append(new_row)

            new_vals_for_layer = np.array(new_vals_for_layer)
            final_vals.append(new_vals_for_layer)


        return final_vals

    def __flatten_input(self, input):
        flat_data = []
        matrix_element = input.shape[0]

        for matrix_element in input:
            flat_data.append(np.matrix.flatten(matrix_element))

        return flat_data

    # Finds the average number of neurons btw input and output - can tweak if produces sub optimal results
    def __calc_hidden_neurons(self, input_neurons, output_neurons):
        return ((input_neurons + output_neurons)/2)

    def __weight_shape(self, first_layer, second_layer):
        # first layer represents edge, second layer represents neurons in second layer
        return np.random.uniform(-1, 1, (int(second_layer), int(first_layer)))

    def __bias_shape(self, weight_shape):
        return np.random.uniform(-1, 1, (weight_shape.shape[0], 1))

    def __sigmoid_func(self, w, x, b):
        # got an overflow error when calc. e^-z. Look up 'RuntimeWarning: Overflow encountered in exp'
        # for more info on why I changed the dtype of z
        z = np.dot(w, x) + b
        z = z.astype('float128') 
        return 1/(1+np.exp(-z))


    # implementing back propogation & gradient descent
    # calc cost derivative for each input in batch then find avg
    def __calc_cost_derivative(self, expected_output, actual_output):
        # a^L is a vector of a foreach j
        # output should be vector of same shape
        return (np.subtract(actual_output, expected_output))

    # calc error for last layer
    def __calc_output_error(self, cost_der, w, b, a):
        sigmoid = self.__sigmoid_func(w, a, b)
        sigmoid_der = sigmoid * (1 - sigmoid)

        return np.multiply(cost_der, sigmoid_der)
    
    # calc error for previous layers
    def __calc_error_for_layer(self, error_l1, w_l1, w, b, a):
        """w_l1 & forward_error are the weight and error for layer l+1. The rest are for layer l"""
        first_term = np.dot(np.transpose(w_l1), error_l1)
        sigmoid = self.__sigmoid_func(w, a, b)
        sigmoid_der = sigmoid * (1 - sigmoid)

        return np.multiply(first_term, sigmoid_der)

    def __w_jk_derivative(self, w, a, error_l1):
        w_derivate = np.zeros(w.shape)
        
        # multiplies each j error by each k activation
        for j in range(w.shape[0]):
            for k in range(w.shape[1]):
                w_derivate[j][k] = a[k] * error_l1[j]

        return w_derivate

# a = np.random.rand(2, 3)
# b = np.random.rand(2, 3)
# c = []
# c.append(a)
# c.append(b)
# c = np.array(c)
# e = [1, 0]

# n = Network(c, 1, batch_size=1, expected_outputs=e)

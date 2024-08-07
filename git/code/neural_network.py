from matplotlib import pyplot as plt
import numpy as np
import pickle
import random

class Network:

    # input should be list of matrices
        # flat_input is a 2d array. An array of each input 
    def __init__(self, input_data, labels, train, num_output_neurons=None, num_hidden=None, flatten=False, batch_size=16, learning_rate=0.5,
    w_filename=None, b_filename=None):
        '''train=0 means the model will be trained\n
        train=1 means the model will be tested\n
        train=2 means the model will predict'''
        
        if train == 0:
            self.__weights = []
            self.__biases = []
            #self.__begin_training(input_data, num_output_neurons, labels, num_hidden, flatten, batch_size, learning_rate)
        elif train == 1:
            self.__test_model(input_data, labels, w_filename, b_filename, batch_size)


    def begin_training(self, input_data, num_output_neurons, training_labels, num_hidden, flatten, batch_size, learning_rate):
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

        costs_over_time = []
        is_correct = 0

        # Tweak to work w batched data 
        # For loop and pass each batch as input
        for i in range(len(batched_training_data)):
            # 1st item is matrix of dC/dw, 2nd item is list of dC/db
            # dC/dw matrix matches up w weights matrix i.e. dC/dW[-1][0] corresponds to weights[0][0]
            # the layers are in reverse however for dC/dw compared to weights matrix
            output_derivatives = self.__calc_w_b_derivatives(batched_training_data[i], self.__weights, self.__biases, batched_labels[i])
            self.__weights = self.__stochastic_grad_descent(self.__weights, learning_rate, output_derivatives[0])
            self.__biases = self.__stochastic_grad_descent(self.__biases, learning_rate, output_derivatives[1])
            costs_over_time += output_derivatives[2] # returns cost for each output neuron - find avg of each neuron foreach input to plot                     
            is_correct += output_derivatives[3]
            if (i%len(batched_training_data) == 100):
                print(f"Batch {i}/{len(batched_training_data)} is done ({i/len(batched_training_data) * 100}%)")
                
        print(f"Accuracy of {is_correct/len(input_data) * 100} %")

        plt.plot(costs_over_time)
        plt.xlabel("Number of inputs")
        plt.ylabel("Cost function")
        plt.show()

    def save_model(self, w_filename, b_filename):
        with open(w_filename, 'wb') as file: 
            pickle.dump(self.__weights, file)

        with open(b_filename, 'wb') as file: 
            pickle.dump(self.__biases, file)


    def __test_model(self, test_data, labels, w_filename, b_filename, batch_size):
        '''Returns accuracy of model via percentage of correct outputs'''

        with open(w_filename, 'rb') as file: 
            weights = pickle.load(file)

        with open(b_filename, 'rb') as file: 
            biases = pickle.load(file)

        batched_test_data = self.__create_batches(test_data, batch_size)
        batched_labels = self.__create_batches(labels, batch_size)

        num_of_correct_outputs = 0

        for i in range(len(batched_test_data)):
            for j in range(len(batched_test_data[i])):
                data = np.reshape(batched_test_data[i][j], (batched_test_data[i][j].shape[0], 1))
                a = self.__feed_forward(data, weights, biases) # activations of each input img
                output = self.__get_output_number(a[-1])

                if self.__is_output_corect(output, batched_labels[i][j]):
                    num_of_correct_outputs += 1

        print(f"Accuracy of {num_of_correct_outputs/len(test_data) * 100}%")

    def __get_output_number(self, output):
        return np.argmax(output)

    def __is_output_corect(self, output, label):
        return (output == label)


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
        cost = []

        correct_counter = 0

        #cost_der = 0
        stochastic_cost = 0
        a = 0

        for i in range(len(input)):
            w_der_layer = []
            b_der_layer = []
            input[i] = input[i].reshape(len(input[i]), 1) # Shape of input w/o this = (len, ). This line changes it to: (len, 1)
            # Initial shape caused problems with the shape of the outputs

            a = self.__feed_forward(input[i], w, b)

            # calculates derivatives for output layer
            cost_der = self.__calc_cost_derivative(input_labels[i], a[-1])
            stochastic_cost = self.__calc_cost(a[-1], input_labels[i])

            e_l = self.__calc_output_error(cost_der, w[-1], b[-1], a[-2])
            w_der_layer.append(self.__w_jk_derivative(w[-1], a[-2], e_l))
            b_der_layer.append(e_l)

            # backpropagation
            for k in range(len(w)):
                # -2-k is the index for the current layer
                index = len(w) - k - 1
                if index == 0:
                    break
                
                e_l = self.__calc_error_for_layer(e_l, w[-1-k], w[-2-k], b[-2-k], a[-3-k])
                w_der_layer.append(self.__w_jk_derivative(w[-2-k], a[-3-k], e_l))
                b_der_layer.append(e_l)

            predicted_output = self.__get_output_number(a[-1])

            if (predicted_output == input_labels[i]):
                correct_counter += 1

            if i != 0:
                for l in range(len(w_der)): # loops through layers
                    w_der[l] = np.add(w_der[l], w_der_layer[l])
                    b_der[l] = np.add(b_der[l], b_der_layer[l])
            else:
                w_der = w_der_layer
                b_der = b_der_layer

            cost.append(stochastic_cost)
            
        mean_w_der = w_der
        mean_b_der = b_der
        for l in range(len(w_der)):
            mean_w_der[l] = w_der[l] / len(input)
            mean_b_der[l] = b_der[l] / len(input)

        return (mean_w_der, mean_b_der, cost, correct_counter)

    def __feed_forward(self, inputs, w, b):
        a = [inputs]
        for i in range(len(w)):
            a.append(self.__sigmoid_func(w[i], a[i], b[i]))

        return a

    #indexing follows this convention: layer, j(neuron in layer l+1), k(neuron in layer l)
    def __stochastic_grad_descent(self, init_vals, learning_rate, derivatives):
        # Loops through each layer in the network and each input in a batch 
        # performs SGD on each input individually
        derivatives.reverse()

        for l in range(len(init_vals)): # loops through the layers and updates each weight/bias
                init_vals[l] -= learning_rate * derivatives[l]

        return init_vals

    def __flatten_input(self, input):
        flat_data = []
        matrix_element = input.shape[0]

        for matrix_element in input:
            flat_data.append(np.matrix.flatten(matrix_element))

        return flat_data

    # Finds the average number of neurons btw input and output
    def __calc_hidden_neurons(self, input_neurons, output_neurons):
        return ((input_neurons + output_neurons)/2)

    def __weight_shape(self, first_layer, second_layer):
        # first layer represents edge, second layer represents neurons in second layer
        return np.random.uniform(-1, 1, (int(second_layer), int(first_layer)))

    def __bias_shape(self, weight_shape):
        return np.zeros((weight_shape.shape[0], 1))

    def __sigmoid_func(self, w, x, b):
        # got an overflow error when calc. e^-z. Look up 'RuntimeWarning: Overflow encountered in exp'
        # for more info on why I changed the dtype of z
        z = np.dot(w, x) + b
        z = z.astype('float128') 
        return (1/(1+np.exp(-z)))


    # implementing back propogation & gradient descent
    # calc cost derivative for each input in batch then find avg
    # cross entropy cost function
    def __calc_cost_derivative(self, label, output):
        y_pred = output[label]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        cost_der = -(1/y_pred_clipped)

        # gets the activation of the expected output and converts from np array to float
        return cost_der

    def __calc_cost(self, output, label):
        y_pred = output[label]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        cost = -np.log(y_pred_clipped)

        return cost

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
        '''Returns gradient for each layer'''
        w_derivate = np.zeros(w.shape)
        
        # multiplies each j error by each k activation
        for j in range(w.shape[0]): # layers
            for k in range(w.shape[1]): # rows
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

import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle
import statistics

class NeuralModel:
    def __init__(self, number_input_layer, number_hidden_layer, number_output_layer, weight_input_hidden, weight_hidden_output, number_data_each_class):
        self.number_inputs_layer = number_input_layer
        self.number_hidden_layer = number_hidden_layer
        self.number_output_layer = number_output_layer
        self.network = []
        self.network.append(weight_input_hidden)
        self.network.append(weight_hidden_output)
        self.inputs = []
        self.data = []
        self.testing = []
        self.num_class = -1 #Store class data of each row is training now
        self.iteration = 0 #Store round of loop training data
        self.phase = 'Training' #Checking phase of datamining
        self.hidden_data = []
        self.result_each_iteration = []
        self.number_data_each_class = number_data_each_class
        self.state = 'train'
        self.gamma = 0.1
        self.abselon = [1 - self.gamma, 1 + self.gamma]
        self.beta = 10
        self.alpha = 0.01 # Constant
        self.sum_error = []
        self.number_weight_input = len(weight_input_hidden)
        self.learn_rate = [0.1 for _ in range(self.number_weight_input)]
        self.save_learnrate = [[] for _ in range(self.number_weight_input)]

        # file = open("../../../present/learning_rate_test/yeast/BPNN+MW+ALR/initial_weight/init_weight_2"+".txt", "a")
        # file.write(str(self.network)+"\n\n")
        # file.close()


    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        self.hidden_data = []
        for i in range(len(weight)):
            if layer == 0:
                if self.phase == 'Training':
                    net_input += weight[i][round(inputs[i], 1)]*inputs[i]
                else:
                    hidden_key = list(weight[i].keys())
                    hidden_key.sort()
                    x1, x2 = round(inputs[i], 1), float()
                    if inputs[i] > hidden_key[0] and inputs[i] < hidden_key[-1]:
                        x2 = round(round(inputs[i], 1) - 0.1, 1) if inputs[i] < round(inputs[i], 1) else round(round(inputs[i], 1) + 0.1, 1)
                    else:
                        x1 = hidden_key[0] if inputs[i] <= hidden_key[0] else hidden_key[1]
                        x2 = hidden_key[0] + 0.1 if inputs[i] <= hidden_key[0] else round(hidden_key[1] - 0.1, 1)
                    m = (weight[i][x2] - weight[i][x1])/(x2-x1)
                    c = (-x1*m)+weight[i][x1]
                    net_input += self.select_weight(m, inputs[i], c)
            else:
                net_input += weight[i]*inputs[i]
        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def forward_propagation(self, data):
        self.data = data
        self.inputs = data
        for layer in range(len(self.network)):
            next_inputs = []
            for neuron in range(len(self.network[layer])):
                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, layer)
                self.network[layer][neuron]['output'] = self.sigmoid(net_input)
                next_inputs.append(self.network[layer][neuron]['output'])
            self.inputs = next_inputs

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propagation(self, expected):
        # Begin at output layer (Backpropagation Steps)
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron['weights'][j] * neuron['errors']
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['errors'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, learning_rate):
        inputs = self.data[:-1]
        for i in range(len(self.network)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    # if i != 0:
                    if i != 0:
                        # neuron['weights'][j] += self.learning_rate[self.num_class] * neuron['errors'] * inputs[j]
                        neuron['weights'][j] += learning_rate * neuron['errors'] * inputs[j]
                    else:
                        # neuron['weights'][j][round(inputs[j], 1)] += self.learning_rate[self.num_class] * neuron['errors'] * inputs[j]
                        neuron['weights'][j][round(inputs[j], 1)] += self.learn_rate[round(inputs[j], 1)] * neuron['errors'] * inputs[j]

    def training(self, dataset, learn_rate, num_iteration, num_output, test):
        self.testing = test
        for iterate in range(int(num_iteration)+1):
            self.iteration = iterate
            sum_error = 0
            sum_each_error = [0 for _ in range(self.number_weight_input)]
            # file = open("../../newWay2/expected"+".txt", "a")
            # file.write("Iteration "+str(iterate)+"\n\n")
            # file.close()
            num = [0 for _ in range(num_output)]
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagation(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[int(row[-1])] = 1

                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                    for i in range(len(expected)):
                        sum_each_error[i] = (expected[i] - self.inputs[i])**2

                    self.back_propagation(expected)
                    self.update_weights(learn_rate)
                    # file = open("../../newWay2/expected"+".txt", "a")
                    # file.write(str(expected)+str(self.inputs)+"\n")
                    # file.close()
                    if self.inputs.index(max(self.inputs)) == expected.index(max(expected)):
                        num[self.num_class] += 1
                    # num[self.inputs.index(max(self.inputs))] += 1
            if iterate != num_iteration:
                self.calculated_abselon(iterate)
                for i in range(len(sum_each_error)):
                    sum_each_error[i] = sum_each_error[i]/len(dataset)
                self.sum_error.append(sum_each_error)
                # update learning rate
                for number_class in range(len(self.learn_rate)):
                    self.save_learnrate[number_class].append(self.learn_rate[number_class])
                    if self.sum_error[iterate-1][number_class] > self.sum_error[iterate][number_class]:
                        self.learn_rate[number_class] = self.learn_rate[number_class]*self.abselon[0]
                    if self.sum_error[iterate-1][number_class] < self.sum_error[iterate][number_class]:
                        self.learn_rate[number_class] = self.learn_rate[number_class]*self.abselon[1]
                    if self.learn_rate[number_class] >= 1:
                            self.learn_rate[number_class] = 1
                print('iteration=%d   learning_rate=%s   mse=%s' % (iterate, self.learn_rate, str(sum_each_error)))

            self.result_each_iteration.append(num)

            # file = open("../../../present/learning_rate_test/yeast/BPNN+MW+ALR/network/end_weight_2"+".txt", "a")
            # file.write(str(self.network)+"\n\n")
            # file.close()

    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagation(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c

    def calculated_abselon(self, iteration):
        # alpha = (self.gamma*(1-math.e**(-1/self.beta)))/(math.e**(-1/self.beta)*(1-math.e**(-500/self.beta)))
        self.abselon[0] = self.abselon[0] + self.alpha*math.e**(-iteration/self.beta)
        self.abselon[1] = self.abselon[1] - self.alpha*math.e**(-iteration/self.beta)

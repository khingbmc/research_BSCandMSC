import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle
import statistics

class TestNeural:
    def __init__(self, inputs, hiddens, outputs, w1, w2, num_data):
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.network = []
        hidden_layer = w1
        output_layer = w2
        # self.network.append(hidden_layer)
        # self.network.append(output_layer)
        self.network = [[{'weights': [0.06488825504703011, 0.8237079168346563, 0.7655014215202536, 0.07897991331387644, 0.154876370981179, 0.9550014088597999, 0.41031824288208674, 0.43716076983357854]}, {'weights': [0.31691253759245985, 0.8629488433653473, 0.5138388761792425, 0.8195076629594134, 0.7549550890409371, 0.8696858852889459, 0.6692592270926871, 0.5599559819530445]}, {'weights': [0.03419578025006664, 0.7699488993573711, 0.3940011090837181, 0.7253358645512011, 0.6390355637622985, 0.3532183309642297, 0.3044871429048326, 0.6904740170329008]}, {'weights': [0.8698701035314406, 0.03441384432739569, 0.6076451683390399, 0.020595887279422964, 0.6711361984832781, 0.0904908418553595, 0.8254572693891736, 0.126615109561235]}, {'weights': [0.37647910866738055, 0.5482326129577475, 0.8685878597255746, 0.9453778967478297, 0.16379780548465472, 0.48668540269528404, 0.4771797196940011, 0.4857314813867565]}, {'weights': [0.44766688173894054, 0.9733844572265832, 0.3727220998913855, 0.2880585587514505, 0.797659968943697, 0.750187272588638, 0.29702738230805326, 0.05605304470542638]}, {'weights': [0.5074203515118313, 0.23303047912246888, 0.3583666925153587, 0.4918709098118288, 0.7825648377522408, 0.6498464018128653, 0.2650826765532337, 0.07288380732768507]}, {'weights': [0.5245355911275075, 0.27835861312613475, 0.649829417134976, 0.3717050479604689, 0.23636950365071574, 0.9151222373071753, 0.10806894938791134, 0.7379103544959281]}, {'weights': [0.22074258736442298, 0.5774691679568487, 0.7880954464202256, 0.4082119655649671, 0.4803354517587294, 0.6314868958616051, 0.9470163507057942, 0.25701203466319755]}, {'weights': [0.4411951092366305, 0.855272710318133, 0.37133851270322127, 0.11730521097243596, 0.35214240359167526, 0.9995745709809964, 0.6746599987912837, 0.5427401607009286]}, {'weights': [0.41943622163858696, 0.09875942861016673, 0.7947761479121682, 0.14401595692461555, 0.8158406650603867, 0.624173415538537, 0.8579384079885282, 0.7976786004683765]}], [{'weights': [0.9089449466084514, 0.1393022381756871, 0.9998862607036548, 0.39778687056314044, 0.3470343933408958, 0.08778620082344002, 0.5281491030581541, 0.6747880531408661, 0.6321665766854444, 0.9859689927533632, 0.30566139805555503]}, {'weights': [0.8584285710358285, 0.12335434126579847, 0.019859439317739258, 0.9658447249796708, 0.9618111907829051, 0.2896709406848399, 0.9786881003886609, 0.6479288988812583, 0.5001290994386104, 0.3980378141102665, 0.48422345819543366]}]]        
        self.inputs = []
        self.data = []
        self.best_network = []
        self.testing = []
        self.num_class = -1
        self.iteration = 0
        self.check_all_weight = [[{'output':[]} for i in range(self.hiddens)] for i in range(self.outputs)]
        self.condition = [{} for _ in range(self.hiddens)]
        self.check_condition = False
        self.gaussian = []
        self.checking = False
        self.phase = 'Training'
        self.number_data_each_class = num_data
        self.gamma = 0.1
        self.abselon = [1 - self.gamma, 1 + self.gamma]
        self.beta = 10
        self.alpha = 0.01 # Constant
        self.sum_error = []
        self.learn_rate = [0.1 for _ in range(outputs)]
        self.save_learnrate = [[] for _ in range(outputs)]
        self.sum_error_1 = []
        self.sum_error_2 = []
        self.acc_iteration = [[0,0] for _ in range(500)]
        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/initial_weight/init_weight_2"+".txt", "a")
        # file.write(str(self.network)+"\n\n")
        # file.close()


    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0
        for i in range(len(weight)):
            net_input += weight[i]*inputs[i]
        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def forward_propagate(self, data):
        self.inputs = data
        self.data = data

        num = 0
        for layer in range(len(self.network)):
            next_inputs = []
            for neuron in range(len(self.network[layer])):

                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, num)
                self.network[layer][neuron]['output'] = self.sigmoid(net_input)

                next_inputs.append(self.network[layer][neuron]['output'])

            self.inputs = next_inputs


    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propagate(self, expected):
        #backprop is begin in outputLayer
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1: #Hidden Layer
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

    def update_weights(self, learn_rate):
        # print(self.num_class)
        # print(learn_rate)
        inputs = self.data[:-1]
        for i in range(len(self.network)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learn_rate[self.num_class]* neuron['errors'] * inputs[j]
                    # neuron['weights'][-1] += learn_rate * neuron['errors']



    def training(self, dataset, learn_rate, num_iteration, num_output, test):
        self.testing = test
        for iterate in range(int(num_iteration)+1):
            self.iteration = iterate
            sum_error = 0
            sum_each_error = [0 for _ in range(num_output)]
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagate(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[int(row[-1])] = 1
                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                    for i in range(len(expected)):
                        sum_each_error[i] = (expected[i] - self.inputs[i])**2

                    if expected.index(expected[int(row[-1])]) == self.inputs.index(max(self.inputs)):
                        self.acc_iteration[iterate][int(row[-1])] += 1
                    # backpropagation
                    self.back_propagate(expected)
                    self.update_weights(learn_rate)
                    # self.update_weights(sum(self.number_data_each_class)/self.number_data_each_class[self.num_class])

            if iterate != num_iteration:
                self.test_function(iterate)
                for i in range(len(sum_each_error)):
                    sum_each_error[i] = sum_each_error[i]/len(dataset)
                self.sum_error.append(sum_each_error)
                # update learning rate
                for number_class in range(len(self.learn_rate)):
                    if number_class == 0:
                        self.sum_error_1.append(self.sum_error[iterate][0])
                    else:
                        self.sum_error_2.append(self.sum_error[iterate][1])
                    self.save_learnrate[number_class].append(self.learn_rate[number_class])
                    if self.sum_error[iterate-1][number_class] > self.sum_error[iterate][number_class]:
                        self.learn_rate[number_class] = self.learn_rate[number_class]*self.abselon[0]
                    if self.sum_error[iterate-1][number_class] < self.sum_error[iterate][number_class]:
                        self.learn_rate[number_class] = self.learn_rate[number_class]*self.abselon[1]
                    if self.learn_rate[number_class] >= 1:
                        self.learn_rate[number_class] = 1
                print('iteration=%d   learning_rate=%s   rmse=%s' % (iterate, str(self.learn_rate),str(sum_each_error)))

            # if iterate == 99 or iterate == 199 or iterate == 299 or iterate == 399 or iterate == 499:
            #     accuracy = 0
            #     for row in self.testing:
            #         prediction = self.predict(row)
            #         print("Apply Model")
            #         print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
            #         file_object = open('../../../present/test_with_iterate/yeast/result/'+str(iterate+1)+'/yeast_prediction_5'+'.txt', 'a')
            #         file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
            #         file_object.write(str(row)+"\n\n")
            #         file_object.close()

            #         if row[-1] == prediction.index(max(prediction)):
            #             accuracy += 1
            #         sum_accuracy = accuracy/len(self.testing)*100

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/network/end_weight_2"+".txt", "a")
        # file.write(str(self.network)+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/error/BC_error_class1_fold2"+".txt", "a")
        # file.write(str(self.sum_error_1)+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/error/BC_error_class2_fold2"+".txt", "a")
        # file.write(str(self.sum_error_2)+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/learning_rate/BC_lr_class1_fold2"+".txt", "a")
        # file.write(str(self.save_learnrate[0])+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+ALR/learning_rate/BC_lr_class2_fold2"+".txt", "a")
        # file.write(str(self.save_learnrate[1])+"\n\n")
        # file.close()
        print("ACCEach")
        print(self.acc_iteration)
        return self.check_all_weight

    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagate(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c

    def test_function(self, iteration):
        # alpha = (self.gamma*(1-math.e**(-1/self.beta)))/(math.e**(-1/self.beta)*(1-math.e**(-500/self.beta)))
        self.abselon[0] = self.abselon[0] + self.alpha*math.e**(-iteration/self.beta)
        self.abselon[1] = self.abselon[1] - self.alpha*math.e**(-iteration/self.beta)

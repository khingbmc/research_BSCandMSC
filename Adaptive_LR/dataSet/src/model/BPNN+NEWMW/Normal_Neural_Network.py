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
        self.num_inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.network = []
        hidden_layer = w1
        output_layer = w2
        self.network.append(hidden_layer)
        self.network.append(output_layer)
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
        self.sum_error = []
        self.num_row = 0
        # self.array_a = [{} for _ in range(self.num_inputs)]
        # self.array_input = [self.array_a for _ in range(self.hiddens)]
        self.array_input = [{i:[] for i in range(self.num_inputs)} for _ in range(self.hiddens)]
        # self.array_b = [[] for _ in range(self.num_inputs)]
        self.array_change_weight = [{i:[] for i in range(self.num_inputs)} for _ in range(self.hiddens)]
        # file = open("../../../present/learning_rate_test/yeast/BPNN/initial_weight/init_weight_5"+".txt", "a")
        # file.write(str(self.network)+"\n\n")
        # file.close()


    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0
        for i in range(len(weight)):
            if layer == 0:
                list_point = weight[i].keys()
                if 'default' in list_point:
                    net_input += weight[i]['default']*inputs[i]
                else:
                    absolute_diff_func = lambda list_value : abs(list_value - inputs[i])
                    closets_value = min(list_point, key=absolute_diff_func)
                    if inputs[i] - closets_value  > 0:
                        less_greater = 'greater'
                    else:
                        less_greater = 'less'
                    net_input += weight[i][closets_value][less_greater]*inputs[i]
            else:
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
                # if layer == 0:
                    # for i in range(len(self.network[layer][neuron]['weights'])):
                        # self.array_input[neuron][i][str(self.inputs[i])] = 0
                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, layer)
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
                    for k in range(len(self.network[i + 1])):
                        neuron = self.network[i + 1][k]
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
        for layer in range(len(self.network)):
            if layer != 0:
                inputs = [neuron['output'] for neuron in self.network[layer - 1]]
            # print(inputs)
            for neuron in range(len(self.network[layer])):
                for j in range(len(inputs)):
                    # self.network[layer][neuron]['weights'][-1] += learn_rate * neuron['errors']*inputs[j]
                    if layer == 0:
                        # key_index = list(self.array_input[neuron][j].keys()).index(str(inputs[j]))
                        # print(key_index)
                        self.array_input[neuron][j].append(inputs[j])
                        self.array_change_weight[neuron][j].append(learn_rate * self.network[layer][neuron]['errors']*inputs[j])
                    else:
                        self.network[layer][neuron]['weights'][-1] += learn_rate * self.network[layer][neuron]['errors']*inputs[j]




    def training(self, dataset, learn_rate, num_iteration, num_output, test):
        self.testing = test
        for iterate in range(int(num_iteration)+1):
            self.array_input = [{i:[] for i in range(self.num_inputs)} for _ in range(self.hiddens)]
            self.array_change_weight = [{i:[] for i in range(self.num_inputs)} for _ in range(self.hiddens)]
            self.iteration = iterate
            sum_error = 0
            sum_each_error = [0 for _ in range(num_output)]
            self.num_row = 0
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagate(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[int(row[-1])] = 1
                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                    for i in range(len(expected)):
                        sum_each_error[i] = (expected[i] - self.inputs[i])**2

                    self.back_propagate(expected)
                    self.update_weights(learn_rate)
                self.num_row += 1
            self.split_positive_negative()
                    # self.update_weights(sum(self.number_data_each_class)/self.number_data_each_class[self.num_class])
            # print(iterate, end='\n')

            if iterate != num_iteration:
                for i in range(len(sum_each_error)):
                    sum_each_error[i] = sum_each_error[i]/len(dataset)
                self.sum_error.append(sum_each_error)
                # update learning rate
                print('iteration=%d   learning_rate=%s   rmse=%s' % (iterate, str(learn_rate),str(sum_each_error)))
        # file = open("../../../present/learning_rate_test/yeast/BPNN/network/end_weight_5"+".txt", "a")
        # file.write(str(self.network)+"\n\n")
        # file.close()
        return self.check_all_weight

    def split_positive_negative(self):
        positive_centroid_array = [{i:0 for i in range(self.num_inputs)} for _ in range(self.hiddens)]
        negative_centroid_array = [{i:0 for i in range(self.num_inputs)} for _ in range(self.hiddens)]

        positive_value = [{i:0 for i in range(self.num_inputs)} for _ in range(self.hiddens)]
        negative_value = [{i:0 for i in range(self.num_inputs)} for _ in range(self.hiddens)]

        for node_hidden in range(self.hiddens):
            for node_input in range(self.num_inputs):
                positive_centroid, positive_divider = 0, 0
                negative_centroid, negative_divider = 0, 0
                num_positive ,num_negative = 0, 0

                for data_row in range(len(self.array_input[node_hidden][node_input])):

                    if self.array_change_weight[node_hidden][node_input][data_row] < 0:
                        negative_centroid += self.array_input[node_hidden][node_input][data_row]*self.array_change_weight[node_hidden][node_input][data_row]
                        negative_divider += self.array_change_weight[node_hidden][node_input][data_row]
                        num_negative += 1

                    elif self.array_change_weight[node_hidden][node_input][data_row] >= 0:
                        positive_centroid += self.array_input[node_hidden][node_input][data_row]*self.array_change_weight[node_hidden][node_input][data_row]
                        positive_divider += self.array_change_weight[node_hidden][node_input][data_row]
                        num_positive += 1
                # print(positive_divider)
                # print(positive_centroid)
                # print(negative_divider)
                if (positive_divider != 0):
                    positive_centroid_array[node_hidden][node_input] = positive_centroid/positive_divider
                    positive_value[node_hidden][node_input] = positive_divider/num_positive
                else:
                    positive_centroid_array[node_hidden][node_input] = 0
                    positive_value[node_hidden][node_input] = 0
                if (negative_divider != 0):
                    negative_centroid_array[node_hidden][node_input] = negative_centroid/negative_divider
                    negative_value[node_hidden][node_input] = negative_divider/num_negative
                else:
                    negative_centroid_array[node_hidden][node_input] = 0
                    negative_value[node_hidden][node_input] = 0
        # print(positive_centroid_array, end='\n\n')
        # print(negative_centroid_array, end='\n\n')
        self.update_input_weight(positive_centroid_array, negative_centroid_array, positive_value, negative_value)

    def update_input_weight(self, positive_centroid, negative_centroid, positive_value, negative_value):
        for node_hidden in range(self.hiddens):
            for node_input in range(self.num_inputs):
                array_weight_value = {}
                point_positive, point_negative = positive_centroid[node_hidden][node_input], negative_centroid[node_hidden][node_input]
                positive, negative = positive_value[node_hidden][node_input], negative_value[node_hidden][node_input]
                mean_centroid = (positive_centroid[node_hidden][node_input]+negative_centroid[node_hidden][node_input])/2
                if 'default' in self.network[0][node_hidden]['weights'][node_input].keys():
                    old_weight = self.network[0][node_hidden]['weights'][node_input]['default']

                    if positive == 0 or negative == 0:
                        array_weight_value = {'default' : old_weight+negative} if positive == 0 else {'default' : old_weight+positive}
                        self.network[0][node_hidden]['weights'][node_input] = array_weight_value
                    else:
                        self.network[0][node_hidden]['weights'][node_input] = {}
                        array_weight_value = {'greater': positive+old_weight, 'less': negative+old_weight} if point_positive > point_negative else {'less': positive+old_weight, 'greater': negative+old_weight}
                        self.network[0][node_hidden]['weights'][node_input][mean_centroid] = array_weight_value

                elif 'default' not in self.network[0][node_hidden]['weights'][node_input].keys():
                    if len(self.network[0][node_hidden]['weights'][node_input].keys()) == 1:
                        old_weight_key = list(self.network[0][node_hidden]['weights'][node_input].keys())[0]
                        old_weight = self.network[0][node_hidden]['weights'][node_input][old_weight_key]
                        if positive == 0 or negative == 0:
                            if positive == 0:
                                if point_negative > old_weight_key:
                                    self.network[0][node_hidden]['weights'][node_input][old_weight_key]['greater'] += negative
                                else:
                                     self.network[0][node_hidden]['weights'][node_input][old_weight_key]['less'] += negative
                            elif negative == 0:
                                if point_positive > old_weight_key:
                                    self.network[0][node_hidden]['weights'][node_input][old_weight_key]['greater'] += positive
                                else:
                                     self.network[0][node_hidden]['weights'][node_input][old_weight_key]['less'] += positive
                        else:
                            if mean_centroid >= old_weight_key:
                                array_weight_value = {'greater': positive+old_weight['greater'], 'less': negative+old_weight['greater']} if point_positive > point_negative else {'less': positive+old_weight['greater'], 'greater': negative+old_weight['greater']}
                            elif mean_centroid < old_weight_key:
                                array_weight_value = {'greater': positive+old_weight['less'], 'less': negative+old_weight['less']} if point_positive > point_negative else {'less': positive+old_weight['less'], 'greater': negative+old_weight['less']}
                            self.network[0][node_hidden]['weights'][node_input][mean_centroid] = array_weight_value
                    else:
                        array_key = np.array(list(self.network[0][node_hidden]['weights'][node_input].keys()))
                        absolute_diff_func = lambda list_value : abs(list_value - mean_centroid)
                        closets_value = min(array_key, key=absolute_diff_func)
                        old_weight = self.network[0][node_hidden]['weights'][node_input][closets_value]
                        if closets_value == max(array_key) and mean_centroid > closets_value:
                            if positive == 0:
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['greater'] += negative
                            elif negative == 0:
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['greater'] += positive
                            else:
                                array_weight_value = {'greater': positive+old_weight['greater'], 'less': negative+old_weight['greater']} if point_positive > point_negative else {'less': positive+old_weight['greater'], 'greater': negative+old_weight['greater']}
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['greater'] = old_weight['greater']+positive if point_positive < point_negative else old_weight['greater']+negative
                                self.network[0][node_hidden]['weights'][node_input][mean_centroid] = array_weight_value
                        elif closets_value == min(array_key) and mean_centroid < closets_value:
                            if positive == 0:
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['less'] += negative
                            elif negative == 0:
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['less'] += positive
                            else:
                                array_weight_value = {'greater': positive+old_weight['less'], 'less': negative+old_weight['less']} if point_positive > point_negative else {'less': positive+old_weight['less'], 'greater': negative+old_weight['less']}
                                self.network[0][node_hidden]['weights'][node_input][closets_value]['less'] = old_weight['less']+positive if point_positive > point_negative else old_weight['greater']+negative
                                self.network[0][node_hidden]['weights'][node_input][mean_centroid] = array_weight_value
                        else:
                            old_weight_key = []
                            old_weight_key.append(array_key[array_key < mean_centroid].max())
                            old_weight_key.append(array_key[array_key > mean_centroid].min())
                            old_weight_key.sort()
                            old_weight_less = self.network[0][node_hidden]['weights'][node_input][old_weight_key[0]]
                            old_weight_greater = self.network[0][node_hidden]['weights'][node_input][old_weight_key[1]]
                            if positive == 0:
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[0]]['greater'] += negative
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[1]]['less'] += negative

                            elif negative == 0:
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[0]]['greater'] += positive
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[1]]['less'] += positive
                            else:
                                if closets_value > mean_centroid:
                                    array_weight_value = {'greater': positive+old_weight['less'], 'less': negative+old_weight['less']} if point_positive > point_negative else {'less': positive+old_weight['less'], 'greater': negative+old_weight['less']}
                                else:
                                    array_weight_value = {'greater': positive+old_weight['greater'], 'less': negative+old_weight['greater']} if point_positive > point_negative else {'less': positive+old_weight['greater'], 'greater': negative+old_weight['greater']}
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[0]]['greater'] = old_weight_less['greater']+positive if point_positive < point_negative else old_weight_less['greater'] + negative
                                self.network[0][node_hidden]['weights'][node_input][old_weight_key[1]]['less'] = old_weight_greater['less']+positive if point_positive > point_negative else old_weight_greater['less'] + negative
                                self.network[0][node_hidden]['weights'][node_input][mean_centroid] = array_weight_value


    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagate(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c

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
        self.network = [[{'weights': [0.1791617459294188, 0.20949535419419907, 0.32159256607715114, 0.2559110040058731, 0.5894734867633385, 0.7265703339619599, 0.053396734714828864, 0.6869924446146348, 0.7260764960484191]}, {'weights': [0.03631471862789082, 0.8386361317090046, 0.5378010903880449, 0.9146475013894813, 0.40369801706260633, 0.3840088536165571, 0.30409269032689257, 0.9743106383084608, 0.1458162190061285]}, {'weights': [0.252112766358789, 0.038507878238444704, 0.2003138072655012, 0.823688024684086, 0.5546729572356813, 0.7605398538065354, 0.3701165859034794, 0.8138821779270934, 0.5153920068185434]}, {'weights': [0.5129516465965127, 0.08289478343727208, 0.2775915574669692, 0.320472841314004, 0.49246111778764456, 0.2565040365304163, 0.01911706436684213, 0.468172845124437, 0.9107127651490304]}, {'weights': [0.7037612227156317, 0.4075121111603984, 0.809597986579944, 0.39877279122269615, 0.5603314574634248, 0.20720845252308795, 0.9279479463408908, 0.06314467853667738, 0.5938049541536504]}, {'weights': [0.47999337192258784, 0.27994457434481246, 0.33464823975379965, 0.6191641228617221, 0.20503367576778364, 0.3762784037229733, 0.7846772369445457, 0.545794786854605, 0.9092497340991079]}, {'weights': [0.7271978593906084, 0.546594742312516, 0.08207690630941478, 0.8169096284176823, 0.5200688969261043, 0.2203161816231788, 0.7483453786157107, 0.5231731913863417, 0.6952810121160047]}, {'weights': [0.50802973976921, 0.3004335193461627, 0.9268371432568888, 0.2679557352032178, 0.16851248134994645, 0.9379315610522461, 0.47561278063324763, 0.8903244922474931, 0.26504310972552947]}, {'weights': [0.03683262328461823, 0.048413039311544837, 0.8214255095871864, 0.7806951630266145, 0.4929803748675007, 0.8768924135942017, 0.16029102375625037, 0.22306923794346845, 0.033933282442406676]}, {'weights': [0.053440253804044935, 0.5472109480744625, 0.9578948822818665, 0.28172289640502224, 0.6833273738530621, 0.32905302895121147, 0.290631527325788, 0.6895059195786128, 0.6503548490701798]}, {'weights': [0.7115445183015805, 0.376659481884389, 0.38622321453773956, 0.7124922821969374, 0.5246152168979741, 0.9065573616242696, 0.8108687282428997, 0.5940074524832188, 0.3827238609246866]}], [{'weights': [0.6093538373490046, 0.590650192596591, 0.8713715161464182, 0.6315256250186093, 0.685813538172896, 0.1018853870449502, 0.24258263920472778, 0.2184604778664716, 0.8932888430497035, 0.5845676866900875, 0.4878564393334619]}, {'weights': [0.8529214953082211, 0.5011727470453174, 0.6089569019064568, 0.9394670551384207, 0.9099001782796808, 0.42354949025769184, 0.6339649208538661, 0.46247282270394385, 0.8145245491779319, 0.4473399539212264, 0.6983163279266904]}]]
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
        self.sum_error = []
        file = open("../../../present/learning_rate_test/BC/BPNN/initial_weight/init_weight_1"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()


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
        for i in range(len(self.network)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][-1] += learn_rate * neuron['errors']



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

                    self.back_propagate(expected)
                    self.update_weights(learn_rate)
                    # self.update_weights(sum(self.number_data_each_class)/self.number_data_each_class[self.num_class])

            if iterate != num_iteration:
                for i in range(len(sum_each_error)):
                    sum_each_error[i] = sum_each_error[i]/len(dataset)
                self.sum_error.append(sum_each_error)
                # update learning rate
                print('iteration=%d   learning_rate=%s   rmse=%s' % (iterate, str(learn_rate),str(sum_each_error)))
        file = open("../../../present/learning_rate_test/BC/BPNN/network/end_weight_1"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()
        return self.check_all_weight

    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagate(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c

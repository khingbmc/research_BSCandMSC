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
        self.network = [[{'weights': [0.9383954614870187, 0.35989509149553267, 0.9871346029353213, 0.6994482500591691, 0.2511251309406345, 0.09813158840619385, 0.3075192254087107, 0.7704627918832718]}, {'weights': [0.1540302380452092, 0.2861807269168948, 0.2582249089215267, 0.12107390459009515, 0.16736942829158763, 0.1139559037540393, 0.3700154181521377, 0.2931772223800435]}, {'weights': [0.7808610669996358, 0.07689525413854048, 0.5558809961486879, 0.005208166738836839, 0.03941824689902185, 0.32729996480119883, 0.6951420968000138, 0.4627891454562889]}, {'weights': [0.7201865271903592, 0.8279315552409462, 0.9768113417448204, 0.5740742906829539, 0.6203730983191329, 0.5154672087758293, 0.9073119917961937, 0.1682666760937762]}, {'weights': [0.15406570446592627, 0.61589422140918, 0.15950590095774653, 0.7799655329395793, 0.2840075333879759, 0.2861400950753604, 0.5013118996104687, 0.13857379330079833]}, {'weights': [0.4204660180078873, 0.4362986169118608, 0.0954829545978384, 0.647672850843528, 0.7694887745730279, 0.46412609229531787, 0.5733076735720666, 0.7000317090537123]}, {'weights': [0.6923410485862266, 0.7727298645349965, 0.04761513340225443, 0.6290636395295544, 0.1079348667091562, 0.7619581995853729, 0.6471289253667734, 0.6375317217257811]}, {'weights': [0.8511548320374118, 0.170740320096742, 0.27256780528943614, 0.9693245177929884, 0.42856248725303936, 0.8399307984081992, 0.6743570800148659, 0.7364666504257091]}, {'weights': [0.0489548619968847, 0.01868630010075445, 0.851640681604993, 0.28282559320517364, 0.6931081519926207, 0.5390036074961588, 0.13690252108762324, 0.0777747403718867]}, {'weights': [0.731278291056106, 0.8842314511935948, 0.6681360222895211, 0.03148615013945233, 0.3039462857567563, 0.8094543173971362, 0.37135264484589314, 0.7868802493698998]}, {'weights': [0.4120328708458517, 0.05685680583861874, 0.1218519895420429, 0.6843953364496521, 0.1220673331140556, 0.16717310866943835, 0.09823399063091898, 0.08632358584229682]}], [{'weights': [0.39737816726625586, 0.07715323646835714, 0.27369874902500846, 0.7256614857816004, 0.6468847560904761, 0.901083108393804, 0.9616986386271291, 0.07945413210454622, 0.7835282695194485, 0.2977405999702666, 0.071790827938645]}, {'weights': [0.44516184553565197, 0.049473834599792266, 0.6648388694322361, 0.1603982757674577, 0.6256442507196109, 0.9646614550707572, 0.5295452629558122, 0.19093626374618833, 0.7347418260768593, 0.884775377919754, 0.18376934762069608]}, {'weights': [0.879899688505106, 0.05908791148515502, 0.018170307966663413, 0.07933786838896595, 0.10953132947009037, 0.11945946164167942, 0.6720226511899662, 0.24500964338971631, 0.3673929983687292, 0.8941925484159021, 0.3966614031105661]}, {'weights': [0.5345508824028986, 0.7950705580506058, 0.7179220770575014, 0.003996370219260914, 0.02363209279945422, 0.49069689259117144, 0.2806627185764118, 0.5658101314262335, 0.1028660116474267, 0.7423767019955152, 0.5208087600625749]}, {'weights': [0.3104758330651508, 0.2634401451390892, 0.7204116902636232, 0.7818646854970001, 0.26173155151397065, 0.2768823613479471, 0.25778235033823804, 0.9941650391670369, 0.5789831999227991, 0.011359842300569634, 0.9485189261659187]}, {'weights': [0.16840471697469728, 0.4259198640550924, 0.35350954680189506, 0.9772258613895752, 0.83476169249289, 0.831482556404478, 0.7146498407470061, 0.8613118860992408, 0.4529292703312232, 0.9504755880182986, 0.2849836810893165]}, {'weights': [0.1748985815545976, 0.5920148000973323, 0.5580405241932567, 0.1663192466795922, 0.6717834617998092, 0.043110880637979965, 0.7515004711302978, 0.11953585936656752, 0.24200333135812835, 0.8757164641073736, 0.7207777864408236]}, {'weights': [0.5517514177369279, 0.8740610093927398, 0.4323608786164873, 0.876131654309052, 0.21364883660615996, 0.9266844927395825, 0.07516898323217291, 0.3475741413694565, 0.8730728809907724, 0.9432113969291485, 0.3750911562444761]}, {'weights': [0.31608792179324097, 0.8415599445916442, 0.9330621174290863, 0.7816798319974178, 0.826015983165775, 0.9458009241197447, 0.8009146833392151, 0.4289627625528577, 0.10255823788258556, 0.18028836091976197, 0.16995948935236938]}, {'weights': [0.832093024514633, 0.503919931733388, 0.14868936251050557, 0.12878505501948545, 0.30274811442119964, 0.778511449686131, 0.03594081521304615, 0.5107365529072375, 0.7297287564968122, 0.06844598561164073, 0.29292657777225195]}]]
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
        file = open("../../../present/learning_rate_test/yeast/BPNN+LR2/initial_weight/init_weight_5"+".txt", "a")
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
                    neuron['weights'][j] += learn_rate* neuron['errors'] * inputs[j]
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

                    self.back_propagate(expected)
                    # self.update_weights(learn_rate)
                    self.update_weights(sum(self.number_data_each_class)/self.number_data_each_class[self.num_class])

            if iterate != num_iteration:
                self.test_function(iterate)
                for i in range(len(sum_each_error)):
                    sum_each_error[i] = sum_each_error[i]/len(dataset)
                self.sum_error.append(sum_each_error)
                # update learning rate
                print('iteration=%d   learning_rate=%s   rmse=%s' % (iterate, str(self.learn_rate),str(sum_each_error)))
        file = open("../../../present/learning_rate_test/yeast/BPNN+LR2/network/end_weight_5"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+LR/error/BC_error_class1_fold2"+".txt", "a")
        # file.write(str(self.sum_error_1)+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+LR/error/BC_error_class2_fold2"+".txt", "a")
        # file.write(str(self.sum_error_2)+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+LR/learning_rate/BC_lr_class1_fold2"+".txt", "a")
        # file.write(str(self.save_learnrate[0])+"\n\n")
        # file.close()

        # file = open("../../../present/learning_rate_test/Diabetes/BPNN+LR/learning_rate/BC_lr_class2_fold2"+".txt", "a")
        # file.write(str(self.save_learnrate[1])+"\n\n")
        # file.close()
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

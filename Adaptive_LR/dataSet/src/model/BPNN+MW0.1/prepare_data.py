#Library
import pandas as pd
import math
import numpy as np

from sklearn.utils import shuffle
from random import random
from random import randint

# Neural Model Class
from Modify_Neural_Network import NeuralModel




def prepareData(num_inputs, number_node_hidden, learning_rate, number_of_iterate):
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    train_path = input("Type path of train file: ")
    test_path = input("Type path of test file: ")

    training_data = pd.read_csv("../../../dataFile/"+train_path+".csv")
    testing_data = pd.read_csv("../../../dataFile/"+test_path+".csv")


    max_val, min_val = [0 for i in range(num_inputs)], [0 for i in range(num_inputs)]

    for i in range(len(training_data.keys())):
        if(training_data.keys()[i] != 'class'):
            max_val[i] = max(training_data[training_data.keys()[i]])
            min_val[i] = min(training_data[training_data.keys()[i]])

    # count each data class
    list_data_class = [x for x in set(training_data['class'])]
    number_of_class = len(list_data_class)
    list_split_data = [[] for _ in range(number_of_class)]
    # split data in each class list
    for i in list_data_class:
        list_split_data[i] = training_data[training_data['class'] == i]

    number_data_each_class = []
    for j in range(len(list_split_data)):
        number_data_each_class.append(len(list_split_data[j]))
    print(len(list_split_data[0]),len(list_split_data[1]), len(training_data))
    print(number_data_each_class)

    # calculate learning rate in each class
    list_learning_rate = [0 for _ in range(number_of_class)]

    for class_index in range(len(list_split_data)):
        list_learning_rate[class_index] = (1-len(list_split_data[class_index])/len(training_data))*learning_rate

    # shuffle row in dataframe
    index = training_data.index
    training_data = shuffle(training_data)
    training_data.index = index

    index = testing_data.index
    testing_data = shuffle(testing_data)
    testing_data.index = index

    ID = training_data.index.values
    list_train = []

    for j in ID:
        format_data = []
        for i in training_data:
            format_data.append(training_data[i][j])
        list_train.append(format_data)

    ID2 = testing_data.index.values
    list_test = []
    for j in ID2:
        format_data = []
        for i in testing_data:
            format_data.append(testing_data[i][j])
        list_test.append(format_data)


    # print(data_key)

    num_outputs = len(set(training_data['class']))

    if(input("Do you want to nomalized this data?? (Yes or No) : ") == 'Yes'):
        print("Eiei")
        for i in range(len(list_train)):
            for j in range(len(max_val)):

                list_train[i][j] = normalized(list_train[i][j], max_val[j], min_val[j])

        for i in range(len(list_test)):
            for j in range(len(max_val)):

                list_test[i][j] = normalized(list_test[i][j], max_val[j], min_val[j])


# %% Init Weight in neural network model
    weight1 = []
    weight2 = []
    # for i in range(number_node_hidden):
    #     weight = {'weights':[]}
    #     # nm_weight = {'weights':[]}
    #     for j in range(num_inputs):
    #         weight_random = random()
    #         sub_weight = {}
    #         for k in np.arange(0.0, 1.1, 0.1):
    #             sub_weight[round(k, 1)] = weight_random
    #         weight['weights'].append(sub_weight)
    #     weight1.append(weight)
    # weight2 = [{'weights':[random() for i in range(number_node_hidden)]} for i in range(num_outputs)]
    save_weight = [[{'weights': [0.5204614589589418, 0.6439056024606266, 0.15680341893091698, 0.046688419593349906, 0.31745221527923884, 0.746656294991757, 0.37951753481146555, 0.9066891034077085, 0.4511048092600728]}, {'weights': [0.7593347490311895, 0.8465018085661613, 0.2546062271146884, 0.20836474726775744, 0.26814183556538584, 0.5562877506490556, 0.156793445425206, 0.11473148894129026, 0.6450136146382218]}, {'weights': [0.07388584663706821, 0.7866442437087809, 0.8156201598343499, 0.14234503496385764, 0.49825157137161746, 0.4245942690750676, 0.6946726606709989, 0.9833593685362694, 0.1384258581420127]}, {'weights': [0.8029840590110895, 0.820356486056968, 0.2648750214746928, 0.18600666873665228, 0.3053749722660327, 0.5984216448613471, 0.5168874459028528, 0.07797575708153415, 0.8249305137504682]}, {'weights': [0.9601815430445421, 0.669644982239067, 0.8854788857214927, 0.7330965696826252, 0.8477553800588152, 0.22903383782388176, 0.8383828097298408, 0.5331418037132665, 0.6422176401066196]}, {'weights': [0.42780454900934306, 0.998922459189718, 0.06361074596215677, 0.4658858275021548, 0.3071120112819592, 0.09933673936556786, 0.08798363492439376, 0.6361422307551388, 0.7106816217484593]}, {'weights': [0.6631488116896272, 0.6330513948551739, 0.37824215341218637, 0.7942029530457477, 0.1738631234022776, 0.8132147395331668, 0.7986275450854959, 0.02591480326969653, 0.3570253793711313]}, {'weights': [0.17681814472059965, 0.20608508059672448, 0.6618950805599757, 0.7049478368278413, 0.9404878455835903, 0.7754831430999024, 0.3933887281917432, 0.08583037830495688, 0.2740043938958121]}, {'weights': [0.9409489427861958, 0.6232816104107506, 0.220025898669996, 0.6296358219891548, 0.46475537374055753, 0.11586935040730739, 0.15054237193340458, 0.8014369478045309, 0.8557825041523752]}, {'weights': [0.006168200863198581, 0.348644142487163, 0.3810730529281071, 0.9409189472208169, 0.49581694273257193, 0.7118170426243497, 0.15396761185181518, 0.8200376667130767, 0.5518678493968272]}, {'weights': [0.0068957907762698945, 0.5210063824398071, 0.5658499691181283, 0.8881573816310561, 0.4174122712850503, 0.5823495070637601, 0.6611013949436467, 0.13114378402902727, 0.2990125555249844]}], [{'weights': [0.0023370684358317018, 0.36603085893675946, 0.5939769069322279, 0.4090742785970798, 0.5185248366653551, 0.6736733805407792, 0.04809502805639987, 0.2881980137307437, 0.556759522776439, 0.6877624769028793, 0.1266076172455851]}, {'weights': [0.12310958040067888, 0.8913001413189596, 0.7067587337101404, 0.48481565294863793, 0.13647275118610114, 0.3441556584068364, 0.635893015056279, 0.9301248187844117, 0.16025183995418057, 0.8188356563453595, 0.9506898957039416]}, {'weights': [0.7019623573794328, 0.25941421313069846, 0.6147809368071301, 0.6582558068832743, 0.12159613130299873, 0.9573576559749596, 0.5667595035796433, 0.7549213245676717, 0.779167093896511, 0.9179204143124002, 0.39316544114625307]}, {'weights': [0.9813946454775111, 0.19148439565344177, 0.3432700133432226, 0.5842645184073424, 0.11552042806848495, 0.5866620377554125, 0.7512713260125219, 0.469925960889639, 0.5097182656960836, 0.4367766503497724, 0.7564559515782592]}]]
    for i in range((len(save_weight))):
        if i == 0:
            for j in range(len(save_weight[i])):
                weight = {'weights':[]}
                for k in range(len(save_weight[i][j]['weights'])):
                    sub_weight1 = {}
                    for c in range(num_outputs):
                        sub_weight2 = {}
                        for a in np.arange(0.0, 1.1, 0.1):
                            sub_weight2[round(a, 1)] = save_weight[i][j]['weights'][k]
                        sub_weight1[c] = sub_weight2
                    weight['weights'].append(sub_weight1)
                weight1.append(weight)
        else:
            weight2 = save_weight[i]
            break
    print(weight1)

# %% Model part
    network = NeuralModel(num_inputs, number_node_hidden, num_outputs, weight1, weight2, number_data_each_class)

    network.training(list_train,learning_rate , number_of_iterate, num_outputs, list_test)


    accuracy = 0
    for row in list_test:
        prediction = network.predict(row)
        print("Apply Model")
        print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
        # file_object = open('../../../present/learning_rate_test/yeast/BPNN+MW/result/yeast_prediction_5'+'.txt', 'a')
        # file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
        # file_object.write(str(row)+"\n\n")
        # file_object.close()

        if row[-1] == prediction.index(max(prediction)):
            accuracy += 1
        sum_accuracy = accuracy/len(list_test)*100
    print("Mean Accuracy = ", sum_accuracy)
    # file_object = open('../../../present/learning_rate_test/yeast/BPNN+MW/result/yeast_prediction_5'+'.txt', 'a')
    # file_object.write("Accuracy : "+ str(sum_accuracy))
    # file_object.close()

prepareData(int(input('Please fill number of node input: ')), int(input('Please fill number of node hidden: ')),float(input('Please fill number of learning rate: ')), int(input('Please fill round of iteration in training phase: ')))
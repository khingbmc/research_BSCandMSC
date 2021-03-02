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
    a = [[{'weights': [0.1791617459294188, 0.20949535419419907, 0.32159256607715114, 0.2559110040058731, 0.5894734867633385, 0.7265703339619599, 0.053396734714828864, 0.6869924446146348, 0.7260764960484191]}, {'weights': [0.03631471862789082, 0.8386361317090046, 0.5378010903880449, 0.9146475013894813, 0.40369801706260633, 0.3840088536165571, 0.30409269032689257, 0.9743106383084608, 0.1458162190061285]}, {'weights': [0.252112766358789, 0.038507878238444704, 0.2003138072655012, 0.823688024684086, 0.5546729572356813, 0.7605398538065354, 0.3701165859034794, 0.8138821779270934, 0.5153920068185434]}, {'weights': [0.5129516465965127, 0.08289478343727208, 0.2775915574669692, 0.320472841314004, 0.49246111778764456, 0.2565040365304163, 0.01911706436684213, 0.468172845124437, 0.9107127651490304]}, {'weights': [0.7037612227156317, 0.4075121111603984, 0.809597986579944, 0.39877279122269615, 0.5603314574634248, 0.20720845252308795, 0.9279479463408908, 0.06314467853667738, 0.5938049541536504]}, {'weights': [0.47999337192258784, 0.27994457434481246, 0.33464823975379965, 0.6191641228617221, 0.20503367576778364, 0.3762784037229733, 0.7846772369445457, 0.545794786854605, 0.9092497340991079]}, {'weights': [0.7271978593906084, 0.546594742312516, 0.08207690630941478, 0.8169096284176823, 0.5200688969261043, 0.2203161816231788, 0.7483453786157107, 0.5231731913863417, 0.6952810121160047]}, {'weights': [0.50802973976921, 0.3004335193461627, 0.9268371432568888, 0.2679557352032178, 0.16851248134994645, 0.9379315610522461, 0.47561278063324763, 0.8903244922474931, 0.26504310972552947]}, {'weights': [0.03683262328461823, 0.048413039311544837, 0.8214255095871864, 0.7806951630266145, 0.4929803748675007, 0.8768924135942017, 0.16029102375625037, 0.22306923794346845, 0.033933282442406676]}, {'weights': [0.053440253804044935, 0.5472109480744625, 0.9578948822818665, 0.28172289640502224, 0.6833273738530621, 0.32905302895121147, 0.290631527325788, 0.6895059195786128, 0.6503548490701798]}, {'weights': [0.7115445183015805, 0.376659481884389, 0.38622321453773956, 0.7124922821969374, 0.5246152168979741, 0.9065573616242696, 0.8108687282428997, 0.5940074524832188, 0.3827238609246866]}], [{'weights': [0.6093538373490046, 0.590650192596591, 0.8713715161464182, 0.6315256250186093, 0.685813538172896, 0.1018853870449502, 0.24258263920472778, 0.2184604778664716, 0.8932888430497035, 0.5845676866900875, 0.4878564393334619]}, {'weights': [0.8529214953082211, 0.5011727470453174, 0.6089569019064568, 0.9394670551384207, 0.9099001782796808, 0.42354949025769184, 0.6339649208538661, 0.46247282270394385, 0.8145245491779319, 0.4473399539212264, 0.6983163279266904]}]]
    for i in range(len(a)):
        if i == 0:
            for j in range(len(a[i])):
                weight = {'weights':[]}
                for k in range(len(a[i][j]['weights'])):
                    sub_weight = {}
                    for c in np.arange(0.0, 1.1, 0.1):
                        sub_weight[round(c, 1)] = a[i][j]['weights'][k]
                    weight['weights'].append(sub_weight)
                weight1.append(weight)
        else:
            weight2 = a[i]
            break

# %% Model part
    network = NeuralModel(num_inputs, number_node_hidden, num_outputs, weight1, weight2, number_data_each_class)

    network.training(list_train,learning_rate , number_of_iterate, num_outputs, list_test)


    accuracy = 0
    for row in list_test:
        prediction = network.predict(row)
        print("Apply Model")
        print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
        # file_object = open('../../../present/learning_rate_test/yeast/BPNN+MW+ALR/result/yeast_prediction_2'+'.txt', 'a')
        # file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
        # file_object.write(str(row)+"\n\n")
        # file_object.close()

        if row[-1] == prediction.index(max(prediction)):
            accuracy += 1
        sum_accuracy = accuracy/len(list_test)*100
    print("Mean Accuracy = ", sum_accuracy)
    # file_object = open('../../../present/learning_rate_test/yeast/BPNN+MW+ALR/result/yeast_prediction_2'+'.txt', 'a')
    # file_object.write("Accuracy : "+ str(sum_accuracy))
    # file_object.close()

prepareData(int(input('Please fill number of node input: ')), int(input('Please fill number of node hidden: ')),float(input('Please fill number of learning rate: ')), int(input('Please fill round of iteration in training phase: ')))
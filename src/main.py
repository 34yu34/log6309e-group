from splitdatabgl import split_bgl   
from models.traditional import SVM
from models.traditional import DecisionTree
from models.traditional import LR
from models.MLP import MLP
from extensions.stat_ranking import ModelData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


csv_input_file = '../data/BGL/BGL_2k.log_structured.csv'
parsed_data_file = '../data/BGL/BGL-log.splitted.npz'
csv_extension_1_path = '../data/extension1.csv'

def lr_model_eval(x_train, y_train, x_test, y_test):
    lr = LR(max_iter=100000)
    lr.fit(x_train, y_train)
    return lr.evaluate(x_test, y_test)

def decision_tree_model_eval(x_train, y_train, x_test, y_test):
    decision_tree = DecisionTree()
    decision_tree.fit(x_train, y_train)
    return decision_tree.evaluate(x_test, y_test)

def SVM_model_eval(x_train, y_train, x_test, y_test):
    svm = SVM(x_train, y_train, x_test, y_test)
    return svm.evaluate()


def MLP_model_eval(x_train, y_train, x_test, y_test, data_path = parsed_data_file):
    model = MLP(data_path)
    loss_list, precision_list, recall_list, f1_list, accuracy_list = model.train_eval(1000)
    print(f1_list)
    return (reduce(lambda x, y: x + y, precision_list) / len(precision_list), reduce(lambda x, y: x + y, recall_list) / len(recall_list), reduce(lambda x, y: x + y, f1_list) / len(f1_list))

n_iter = 50

labels = ['LR', 'Tree', 'SVM', 'MLP']
models = [lr_model_eval,decision_tree_model_eval, SVM_model_eval, MLP_model_eval]

def train_models(n_iter, models, shuffle=True):
    data = [[] for i in range(len(models))]

    for i in range(n_iter):
        train_x, train_y, test_x, test_y = split_bgl(csv_input_file, parsed_data_file, shuffle)

        for i in range(len(models)):
            metric = models[i](train_x, train_y, test_x, test_y)
            data[i].append(metric[2])
    return data


def E1():
    plt.figure(figsize=(10, 6))

    data = train_models(n_iter, models)
    numpy_data = np.array(data)
    df_data = pd.DataFrame(np.transpose(numpy_data))
    df_data.to_csv(csv_extension_1_path)

    means = np.mean(numpy_data, axis=1)
    var = np.var(numpy_data, axis=1)

    print(means)

    for i in range(len(labels)):
        print(f"the {labels[i]} means is : {means[i]}")
        print(f"the {labels[i]} variance is : {var[i]}")

    plt.boxplot(data, patch_artist=True, labels=labels)
    plt.title(f"F1 score of {n_iter} fitting with the BGL dataset")
    plt.show()



def E1_load():
    df = pd.read_csv(csv_extension_1_path)

    print(df.rank(axis=0, method="average"))

def E4():
    d = train_models(1, models, False)

    arr = np.array(d)
    print(np.average(arr, axis=2))

    for i in range(len(labels)):
        print(labels[i], d[i])
        for d_line in d[i]:
            print(reduce(lambda x, y : x+y, d_line) / len(d_line))

E1()
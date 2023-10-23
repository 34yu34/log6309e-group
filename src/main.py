import json
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
from metric_visualizer import MetricVisualizer

csv_input_file = '../data/BGL/BGL.log_structured.csv'
parsed_data_file = '../data/BGL/BGL-log.splitted.npz'
csv_extension_1_path = '../data/extension2.csv'
json_res_path = '../data/results.json'


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

n_iter = 10

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

    for i in range(len(labels)):
        print(f"the {labels[i]} means is : {means[i]}")
        print(f"the {labels[i]} variance is : {var[i]}")

    plt.boxplot(data, patch_artist=True, labels=labels)
    plt.title(f"F1 score of {n_iter} fitting with the BGL dataset")
    plt.show()


def E1_load():

    f = open(json_res_path)
    data_raw = json.load(f)
    df = pd.json_normalize(data_raw['random'])

    scores_f = [df['LR.f1'][0], df['Tree.f1'][0], df['SVM.f1'][0], df['MLP.f1'][0]]
    scores_p = [df['LR.precision'][0], df['Tree.precision'][0], df['SVM.precision'][0], df['MLP.precision'][0]]
    scores_r = [df['LR.recall'][0], df['Tree.recall'][0], df['SVM.recall'][0], df['MLP.recall'][0]]

    # f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # f.set_figheight(12)
    # f.set_figwidth(18)
    # f.suptitle('Different model scores over 20 fitting')

    # ax1.boxplot(scores_f, patch_artist=True, labels=labels)
    # ax1.set_title("F1 scores of each models")
    # ax2.boxplot(scores_p, patch_artist=True, labels=labels)
    # ax2.set_title("precision scores of each models")
    # ax3.boxplot(scores_r, patch_artist=True, labels=labels)
    # ax3.set_title("recall scores of each models")


    MV = MetricVisualizer(name='example', trial_tag='Model')

    MV.rank_test_by_metric()

    plt.boxplot(scores_f, patch_artist=True, labels=labels)
    plt.title("F1 scores of each models over 20 fitting")
    plt.show()

    df_data = pd.DataFrame(df[['LR.f1', 'Tree.f1', 'SVM.f1', 'MLP.f1']])
    df_data.to_csv(csv_extension_1_path)



def E4_load():
    f = open(json_res_path)
    data_raw = json.load(f)
    df = pd.json_normalize(data_raw['random'])
    df2 = pd.json_normalize(data_raw['sequential'])

    scores_f = np.array([df['LR.f1'][0], df['Tree.f1'][0], df['SVM.f1'][0], df['MLP.f1'][0]])
    scores_p = np.array([df['LR.precision'][0], df['Tree.precision'][0], df['SVM.precision'][0], df['MLP.precision'][0]])
    scores_r = np.array([df['LR.recall'][0], df['Tree.recall'][0], df['SVM.recall'][0], df['MLP.recall'][0]])
    scores_f_2 = np.array([df2['LR.f1'][0], df2['Tree.f1'][0], df2['SVM.f1'][0], df2['MLP.f1'][0]])
    scores_p_2 = np.array([df2['LR.precision'][0], df2['Tree.precision'][0], df2['SVM.precision'][0], df2['MLP.precision'][0]])
    scores_r_2 = np.array([df2['LR.recall'][0], df2['Tree.recall'][0], df2['SVM.recall'][0], df2['MLP.recall'][0]])

    print(f"average F1 random_based ({np.mean(scores_f, axis=1)}) vs time-based ({np.mean(scores_f_2, axis=1)}) with diff {np.mean(scores_f, axis=1)-np.mean(scores_f_2, axis=1)}")
    print(f"average precision random_based ({np.mean(scores_p, axis=1)}) vs time-based ({np.mean(scores_p_2, axis=1)}) with diff {np.mean(scores_p, axis=1)-np.mean(scores_p_2, axis=1)}")
    print(f"average recall random_based ({np.mean(scores_r, axis=1)}) vs time-based ({np.mean(scores_r_2, axis=1)}) with diff {np.mean(scores_r, axis=1)-np.mean(scores_r_2, axis=1)}")


def E4():
    d = train_models(1, models, False)

    arr = np.array(d)
    print(np.average(arr, axis=2))

    for i in range(len(labels)):
        print(labels[i], d[i])
        for d_line in d[i]:
            print(reduce(lambda x, y : x+y, d_line) / len(d_line))

E4_load()
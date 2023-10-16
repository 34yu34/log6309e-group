from splitdatabgl import split_bgl   
from models.traditional import SVM
from models.traditional import DecisionTree
from models.traditional import LR
from extensions.stat_ranking import ModelData
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce



def lr_model_eval(x_train, y_train, x_test, y_test):
    lr = LR(max_iter=100000)
    lr.fit(train_x, train_y)
    return lr.evaluate(test_x, test_y)

def decision_tree_model_eval(x_train, y_train, x_test, y_test):
    decision_tree = DecisionTree()
    decision_tree.fit(train_x, train_y)
    return decision_tree.evaluate(test_x, test_y)

def SVM_model_eval(x_train, y_train, x_test, y_test):
    svm = SVM(train_x, train_y, test_x, test_y)
    return svm.evaluate()

n_iter = 100

labels = ['LR', 'Tree', 'SVM']
models = [lr_model_eval,decision_tree_model_eval, SVM_model_eval]

data = [[] for i in range(len(models))]

for i in range(n_iter):
    train_x, train_y, test_x, test_y = split_bgl(
        '../data/BGL/BGL_2k.log_structured.csv')
    # model_data = ModelData(train_x, train_y, test_x, test_y)

    for i in range(len(models)):
        metric = models[i](train_x, train_y, test_x, test_y)
        data[i].append(metric[2])

plt.boxplot(data, patch_artist=True, labels=labels)

plt.title(f"F1 score of {n_iter} fitting with the BGL dataset")
plt.show()
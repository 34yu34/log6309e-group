from splitdatabgl import split_bgl   
from models.traditional import SVM
from models.traditional import DecisionTree
from models.traditional import LR
from extensions.stat_ranking import ModelData
import pandas as pd


train_x, train_y, test_x, test_y = split_bgl()

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
  
decision_tree = DecisionTree()
decision_tree.fit(train_x, train_y)

print(decision_tree.feature_importances_)


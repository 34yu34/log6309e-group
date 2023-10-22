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
from logrep.converter import convert_bgl_to_pyz, process_bgl
from logrep.ttidfgenerator import generate_train_test
import json


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

def MLP_model_eval(x_train, y_train, x_test, y_test):
    model = MLP(x_train, y_train, x_test, y_test)
    loss_list, precision_list, recall_list, f1_list, accuracy_list = model.train_eval(1000)
    return precision_list[-1], recall_list[-1], f1_list[-1]

def split_data(data: dict, labels: dict, train_keys: list, test_keys: list):
    train_x = {}
    test_x = {}
    train_y = []
    test_y = []
    for key in train_keys:
        train_x[key] = data[key]
        train_y.append(labels[key])

    for key in test_keys:
        test_x[key] = data[key]
        test_y.append(labels[key])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def train_models(n_it, models, model_names, shuffle=True):
    print("Loading structured logs...")
    data, labels = process_bgl(csv_input_file)
    keys = sorted(list(data.keys()))
    if shuffle:
        np.random.shuffle(keys)

    data_per_group = len(keys) // (n_it + 1)
    test_size = int(0.3*len(keys))
    results = {}
    for i in range(n_it):
        print("=====================================")
        print("Iteration {}/{}...".format(i+1, n_it))
        print("=====================================")
        # Split data into train and test
        upper_bound = min(len(keys), i*data_per_group+test_size)
        test_keys = keys[i*data_per_group:upper_bound]
        train_keys = [k for k in keys if k not in test_keys]
        train_x, train_y, test_x, test_y = split_data(data, labels, train_keys, test_keys)
        
        # Generate log representation
        train_x, test_x = generate_train_test(train_x, train_y, test_x, test_y)
        
        # Test models
        for model_idx, model in enumerate(models):
            precision, recall, f1 = model(train_x, train_y, test_x, test_y)
            if model_names[model_idx] not in results:
                results[model_names[model_idx]] = {"precision": [], "recall": [], "f1": []}
            results[model_names[model_idx]]["precision"].append(precision)
            results[model_names[model_idx]]["recall"].append(recall)
            results[model_names[model_idx]]["f1"].append(f1)
    return results


if __name__ == "__main__":

    csv_input_file = '../data/BGL/BGL.log_structured.csv'
    labels = ['LR', 'Tree', 'SVM', 'MLP']
    models = [lr_model_eval,decision_tree_model_eval, SVM_model_eval, MLP_model_eval]
    
    # Random split
    random_results = train_models(n_it=20, models=models, model_names=labels, shuffle=True)
    
    # Sequential split
    seq_results = train_models(n_it=20, models=models, model_names=labels, shuffle=False)
    results = {"random": random_results, "sequential": seq_results}
    
    with open('results.json', 'w') as fp:
        json.dump(results, fp)
        print('dictionary saved successfully to file')



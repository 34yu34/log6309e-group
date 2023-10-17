# Multi-layer perception for log anomaly detection

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support


class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.output = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.output(out)
        return out

class MLP_L1(object):
    def __init__(self, data_path='') -> None:
        seq_level_data = np.load(data_path, allow_pickle=True)

        x_train = seq_level_data["x_train_l1"]
        y_train = seq_level_data["y_train_l1"]
        x_test = seq_level_data["x_test_l1"]
        self.y_test = seq_level_data["y_test_l1"]

        # balance pos/neg samples
        x_train = np.array(x_train, dtype=np.float64)
        x_test = np.array(x_test, dtype=np.float64)
        pos = y_train==1
        print(x_train.shape)
        print(y_train.shape)
        print('positive:',sum(pos))
        print('ratio:',x_train.shape[0]/sum(pos))
        ratio = x_train.shape[0]/sum(pos)

        index = np.argwhere(pos == True)
        fea_dim = x_train.shape[1]
        pos_items = x_train[index,:].reshape(-1,fea_dim)
        pos_rep = np.repeat(pos_items, ratio-1, axis =0)

        new_x_train = np.concatenate((x_train, pos_rep),axis =0)
        new_y_train = np.append(y_train, np.ones(pos_rep.shape[0]))

        # shuffle new dataset
        index=np.arange(new_x_train.shape[0])
        np.random.shuffle(index)

        new_x_train_s = new_x_train[index]
        new_y_train_s = new_y_train[index]
        self.batch_size = y_train.shape[0]

        self.y_train_tensor = torch.from_numpy(new_y_train_s).type(torch.LongTensor)
        self.y_train_onehot = torch.from_numpy(new_y_train_s).type(torch.LongTensor)
        self.x_train_tensor = torch.from_numpy(new_x_train_s).float()

        self.x_test_tensor = torch.from_numpy(x_test).float()
        self.y_test_tensor = torch.from_numpy(self.y_test.reshape(-1,1))
        self.y_test_onehot = torch.zeros(self.y_test_tensor.shape[0], 2).scatter_(1, self.y_test_tensor, 1)
        
        self.net = Net(fea_dim,200,2)
        print(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.01)
        self.loss_func = torch.nn.CrossEntropyLoss()
            
    def train_eval(self, num_it=2000):
        loss_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []
        for t in range(num_it):
            out = self.net(self.x_train_tensor)
            loss = self.loss_func(out, self.y_train_onehot)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if t%10 == 0:
                print('itr: %d, loss: %f'% (t,loss))
                out_ = self.net(self.x_test_tensor)
                y_ = torch.argmax(out_, -1)
                acc = (y_==torch.from_numpy(self.y_test)).sum()/y_.shape[0]
                precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_, average='binary')
                loss_list.append(loss.item())
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                accuracy_list.append(acc)
                print('Testset: Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, acc:{:.3f}\n'.format(precision, recall, f1, acc))
        return loss_list, precision_list, recall_list, f1_list, accuracy_list
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from logrep.converter import convert_bgl_to_pyz


#convert_bgl_to_pyz('data/BGL/BGL.log_structured.csv', 'data/BGL/BGL.log.splitted')


# In[8]:


#from logrep.ttidfgenerator import generate

#generate('data/BGL', 'BGL.log.splitted')


# In[9]:


import numpy as np
loaded = np.load('data/BGL/text-tfidf-template-BGL.log.splitted.npz', allow_pickle=True)
train_x = loaded['x_train'][()]
train_y = loaded['y_train']
test_x = loaded['x_test'][()]
test_y = loaded['y_test']


# In[10]:
from models.MLP import MLP
model = MLP("data/BGL/text-tfidf-template-BGL.log.splitted.npz")
model.train_eval()


print("DecisionTree")
from models.traditional import DecisionTree

decision_tree = DecisionTree()
decision_tree.fit(train_x, train_y)
decision_tree.evaluate(test_x, test_y)


# In[6]:

print("LR")
from models.traditional import LR

lr = LR(max_iter=100000)
lr.fit(train_x, train_y)
lr.evaluate(test_x, test_y)


# In[7]:


print("SVM")
from models.traditional import SVM

svm = SVM(train_x, train_y, test_x, test_y)
svm.evaluate()


# In[ ]:





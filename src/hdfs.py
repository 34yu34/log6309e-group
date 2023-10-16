
#!/usr/bin/env python
# coding: utf-8

# In[21]:


#from src.hdfsparser import parse_to_csv

#parse_to_csv('data/HDFS', 'data/HDFS', 'HDFS.log')


# In[1]:


#from logrep.converter import convert_hdfs_to_splitted_pyz

#convert_hdfs_to_splitted_pyz('data/HDFS/preprocessed/HDFS.npz', 'data/HDFS/HDFS.log.splitted')


# In[ ]:


#from logrep.ttidfgenerator import generate

#generate('data/HDFS', 'HDFS.log.splitted')


# In[3]:


import numpy as np
loaded = np.load('data/HDFS/text-tfidf-template-HDFS.log.splitted.npz', allow_pickle=True)
train_x = loaded['x_train'][()]
train_y = loaded['y_train']
test_x = loaded['x_test'][()]
test_y = loaded['y_test']


# In[5]:



# In[8]:


#from tqdm import tqdm

#size = max(train_x.keys()) + 1

# Initialize an array of zeros
#result_array = np.zeros(size)

# Assign values from the dictionary to the corresponding indices
#for key, value in tqdm(train_x.items()):
#    result_array[key] = np.array(value)


# In[3]:


#print(result_array)
from models.MLP import MLP

model = MLP('data/HDFS/text-tfidf-template-HDFS.log.splitted.npz')
loss, precision, recall, f1, accuracy = model.train_eval()


# In[3]:

print("DecisionTree")

from models.traditional import DecisionTree

decision_tree = DecisionTree()
decision_tree.fit(train_x, train_y)
decision_tree.evaluate(test_x, test_y)


print("LR")

from models.traditional import LR

lr = LR(max_iter=100000)
lr.fit(train_x, train_y)
lr.evaluate(test_x, test_y)

print("SVM")

from models.traditional import LR

lr = LR(max_iter=100000)
lr.fit(train_x, train_y)
lr.evaluate(test_x, test_y)





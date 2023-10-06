import numpy as np
import pandas as pd
from tqdm import tqdm

# aggregate log event-level representation to seqence-level

structured_logs = pd.read_csv('../../data/BGL_2k/BGL_2k.log_structured.csv')
evt_level_rep = pd.DataFrame.to_numpy(structured_logs)

x_train = evt_level_rep["x_train"][()]
y_train = evt_level_rep["y_train"]
x_test = evt_level_rep["x_test"][()]
y_test = evt_level_rep["y_test"]

# Aggregation
x_train_agg = []
for i in tqdm(range(x_train.shape[0])):
    fea = np.mean(x_train[i], axis = 0)
    x_train_agg.append(fea)
x_train_agg = np.array(x_train_agg)
x_train_agg.shape


x_test_agg = []
for i in tqdm(range(x_test.shape[0])):
    fea = np.mean(x_test[i], axis = 0)
    x_test_agg.append(fea)
x_test_agg = np.array(x_test_agg)
x_test_agg.shape

np.savez('data/BGL_2k/agg/BGL_word2vec_template_TFIDF_300d_agg.npz',x_train=x_train_agg, y_train = y_train, x_test = x_test_agg, y_test= y_test)
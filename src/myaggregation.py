import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def aggregate(input_file_name, output_file_name):
    # aggregate log event-level representation to sequence-level

    df = pd.read_csv(input_file_name + '.csv')
    evt_level_rep = pd.DataFrame.to_numpy(df)


    df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d-%H.%M.%S.%f")
    g = df.groupby(pd.Grouper(key='Time', freq='6H'))    # x_train = train.to_dict('index')

    train, test = train_test_split(g.count(), test_size=0.2)

    x_train = train
    y_train = np.zeros(len(train))
    # x_test = test.to_dict('index')
    x_test = test
    y_test = np.ones(len(test))

    # Aggregation
    x_train_agg = []
    # for i in tqdm(range(len(x_train))):
    for i in tqdm(range(x_train.shape[0])):
        fea = np.mean(x_train.to_numpy()[i], axis=0)
        x_train_agg.append(fea)
    x_train_agg = np.array(x_train_agg)
    print(x_train_agg.shape)

    x_test_agg = []
    # for i in tqdm(range(len(x_test))):
    for i in tqdm(range(x_test.shape[0])):
        fea = np.mean(x_test[i], axis=0)
        x_test_agg.append(fea)

    x_test_agg = np.array(x_test_agg)
    print(x_test_agg.shape)

    np.savez(output_file_name + '.npz',
             x_train=x_train_agg,
             y_train=y_train,
             x_test=x_test_agg,
             y_test=y_test)
aggregate('../../data/BGL_2k/BGL_2k.log_structured', '../../data/BGL_2k/BGL_2k.agg')
# aggregate('../../data/BGL/BGL.log_structured', '../../data/BGL/BGL.agg')

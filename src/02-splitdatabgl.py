import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: Make it work to run `articlepackage/logrep/aggregation.py` on the output
def split_bgl(input_file: str = '../data/BGL/BGL.log_structured.csv',
              output_file_path: str = '../data/BGL/BGL-log.splitted.npz'):

    df = pd.read_csv(input_file)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d-%H.%M.%S.%f")
    g = df.groupby(pd.Grouper(key='Time', freq='6H'))

    train, test = train_test_split(g.size(), test_size=0.2)

    x_train = train
    y_train = np.zeros(len(train))
    x_test = test
    y_test = np.ones(len(test))

    np.savez(output_file_path,
             x_train=x_train,
             y_train=y_train,
             x_test=x_test,
             y_test=y_test)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_bgl(input_file: str = '../data/BGL/BGL.log_structured.csv',
              output_file_path: str = '../data/BGL/BGL-log.splitted.npz'):

    df = pd.read_csv(input_file)

    df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d-%H.%M.%S.%f")

    df['IsAnomaly'] = df['Label'] != '-'
    g = df.groupby([pd.Grouper(key='Time', freq='6H'), 'EventId'])
    h = df.groupby([pd.Grouper(key='Time', freq='6H')])
    anomaly_df = h.apply(lambda x: x.loc[:, 'IsAnomaly'].any()).reset_index()

    count_df = g.apply(lambda x: x.shape[0]).reset_index()
    count_df.rename({0:'count'}, axis=1, inplace=True)

    anomaly_df.rename({0:'IsAnomaly'}, axis=1, inplace=True)
    merged = pd.merge(anomaly_df, count_df, how='outer', on='Time')

    event_count_vector = merged.pivot(index=["Time", 'IsAnomaly'],
                                      columns="EventId",
                                      values='count').fillna(0).drop(columns=[np.NaN])


    train, test = train_test_split(event_count_vector, test_size=0.2)

    test_y = test.reset_index(level='IsAnomaly')['IsAnomaly'].to_numpy() * 1
    train_y = train.reset_index(level='IsAnomaly')['IsAnomaly'].to_numpy() * 1
    test_x = test.reset_index(level='IsAnomaly').drop(columns=['IsAnomaly']).to_numpy()
    train_x = train.reset_index(level='IsAnomaly').drop(columns=['IsAnomaly']).to_numpy()

    np.savez(output_file_path,
             x_train=train_x,
             y_train=train_y,
             x_test=test_x,
             y_test=test_y)

    return train_x, train_y, test_x, test_y
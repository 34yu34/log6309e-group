# aggregate log event-level representation to seqence-level
import numpy as np
from tqdm import tqdm


def aggregate(parsed_data_file_path, output_file_path):
    evt_level_rep = np.load(parsed_data_file_path, allow_pickle=True)

    x_train = evt_level_rep["x_train"][()]
    y_train = evt_level_rep["y_train"]
    x_test = evt_level_rep["x_test"][()]
    y_test = evt_level_rep["y_test"]

    # Aggregation
    x_train_agg = []
    for i in tqdm(range(x_train.shape[0])):
        fea = np.mean(x_train[i], axis=0)
        x_train_agg.append(fea)
    x_train_agg = np.array(x_train_agg)
    print(x_train_agg.shape)

    x_test_agg = []
    for i in tqdm(range(x_test.shape[0])):
        fea = np.mean(x_test[i], axis=0)
        x_test_agg.append(fea)
    x_test_agg = np.array(x_test_agg)
    print(x_test_agg.shape)

    np.savez(output_file_path,
             x_train=x_train_agg,
             y_train=y_train,
             x_test=x_test_agg,
             y_test=y_test)

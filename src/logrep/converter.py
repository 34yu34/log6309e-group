import csv
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def convert_to_pyz(input_file, output_file):
    data, labels = process(input_file)
    train_x, train_y, test_x, test_y = format_sets(data, labels)
    save_sets(output_file, train_x, train_y, test_x, test_y)
    print("Done 🎉")


def process(input_file):
    data = {}
    group_count = 0
    labels = {}
    SIX_H = 66060
    columns = None

    with open(input_file, 'r') as f:
        reader = csv.reader(x.replace('\0', '') for x in f)

        group_start = None
        for idx, row in tqdm(enumerate(reader)):
            if not row:
                continue

            if idx == 0:
                columns = row
                continue

            # Initialize first group
            if not group_start:
                group_start = row[2]  # Timestamp
                group_end = int(row[2]) + SIX_H
                group_data = []
                labels[str(group_start)] = int(False)

            # Check if group is finished and reset
            if int(row[2]) > group_end:
                data[str(group_start)] = group_data
                group_start = int(row[2])
                group_end = int(row[2]) + SIX_H
                group_count += 1
                group_data = []
                labels[str(group_start)] = int(False)

            # Save label data
            labels[str(group_start)] = int((row[1] != '-') or labels[str(group_start)])

            # Save event data
            log_data = {}
            for field in columns:
                log_data[field] = row[columns.index(field)]
            group_data.append(log_data)


    return data, labels


def format_sets(data, labels):
    train, test = train_test_split(list(data.keys()), test_size=0.2)

    train_x = {}
    test_x = {}
    train_y = []
    test_y = []
    for key in train:
        train_x[key] = data[key]
        train_y.append(labels[key])

    for key in test:
        test_x[key] = data[key]
        test_y.append(labels[key])

    return train_x, train_y, test_x, test_y


def save_sets(output_file: str,
              train_x: dict,
              train_y: List,
              test_x: dict,
              test_y: List):
    np.savez(output_file,
             x_train=np.array(train_x),
             y_train=np.array(train_y),
             x_test=np.array(test_x),
             y_test=np.array(test_y)
             )

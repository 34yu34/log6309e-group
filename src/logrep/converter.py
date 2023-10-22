import csv
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def convert_bgl_to_pyz(input_file: str, output_file: str):
    data, labels = process_bgl(input_file)
    train_x, train_y, test_x, test_y = format_sets(data, labels)
    save_sets(output_file, train_x, train_y, test_x, test_y)
    print("Done 🎉")


def process_bgl(input_file: str):
    data = {}
    group_count = 0
    labels = {}
    SIX_H = 6 * 60 * 60
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


def convert_hdfs_to_splitted_pyz(preprocessed_pickle_file_path: str,
                                 output_file: str,
                                 preprocessed_log_templates_file_path='../data/HDFS/preprocessed/HDFS.log_templates.csv'
                                 ):
    data, labels = extract_groups_and_labels_from_hdfs(preprocessed_pickle_file_path,
                                                       preprocessed_log_templates_file_path)
    train_x, train_y, test_x, test_y = format_sets(data, labels, test_size=0.3)
    save_sets(output_file, train_x, train_y, test_x, test_y)
    print("Done 🎉")


def extract_groups_and_labels_from_hdfs(preprocessed_pickle_file_path: str,
                                        preprocessed_log_templates_file_path: str) -> (dict, dict):
    npz = np.load(preprocessed_pickle_file_path, allow_pickle=True)
    labels = dict(enumerate(npz['y_data'].flatten(), 1))
    labels = {str(k):v for k,v in labels.items()}
    groups = dict(enumerate(npz['x_data'].flatten(), 1))

    logsdf = pd.read_csv(preprocessed_log_templates_file_path)

    groups_with_event_template = {}

    print("Adding 'EventTemplate' on events in groups")
    for group_id in tqdm(groups):
        gs = []
        for event_id in groups[group_id]:
            event = {
                'EventId': str(event_id),
                'EventTemplate': logsdf[logsdf['EventId'] == event_id]['EventTemplate'].to_list()[0]
            }
            gs.append(event)
        groups_with_event_template[str(group_id)] = gs

    print('====== Grouping summary ======')
    print("Total number of sessions: {}".format(npz['x_data'].size))
    return groups_with_event_template, labels


def format_sets(data: dict, labels: dict, test_size=0.2):
    train, test = train_test_split(list(data.keys()), test_size=test_size)

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

    print('====== Splitting summary ======')
    print("Total number of test sessions: {}".format(len(test_x)))
    print("Total number of train sessions: {}".format(len(train_x)))

    return train_x, train_y, test_x, test_y


def save_sets(output_file: str,
              train_x: dict,
              train_y: List,
              test_x: dict,
              test_y: List):
    print("Saving to file: {}".format(output_file))
    np.savez(output_file,
             x_train=np.array(train_x),
             y_train=np.array(train_y),
             x_test=np.array(test_x),
             y_test=np.array(test_y)
             )

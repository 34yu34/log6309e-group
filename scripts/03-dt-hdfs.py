#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
sys.path.append('../loglizer/')
from loglizer import preprocessing
from loglizer.dataloader import HDFS
from loglizer.models import DecisionTree
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


struct_log = 'data/HDFS_v1/HDFS.log_structured.csv' # The structured log file
label_file = 'data/HDFS_v1/preprocessed/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = HDFS.loadDataset(struct_log,
                                                                label_file=label_file,
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = DecisionTree()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
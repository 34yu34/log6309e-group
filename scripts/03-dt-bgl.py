#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
sys.path.append('../loglizer/')
from loglizer import preprocessing
from loglizer.dataloader import BGL
from loglizer.models import DecisionTree
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


struct_log = 'data/BGL/BGL.log_structured.csv' # The structured log file

if __name__ == '__main__':
    # TODO: Double check data loading parameters from paper
    # TODO: Implement data loading options for BGL dataset (uniform split vs sequential split)
    (x_train, y_train), (x_test, y_test) = BGL.loadDataset(struct_log,
                                                          train_ratio=0.5)

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = DecisionTree()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
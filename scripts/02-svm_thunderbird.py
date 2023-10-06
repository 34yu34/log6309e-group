#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
sys.path.append('../loglizer/')
from loglizer.models import SVM
from loglizer import preprocessing
from loglizer.dataloader import Thunderbird

struct_log = 'data/Thunderbird_2k/Thunderbird_2k.log_structured.csv' # The structured log file

if __name__ == '__main__':
    # TODO: Double check data loading parameters from paper
    # TODO: Implement data loading options for Thunderbird dataset (uniform split vs sequential split)
    (x_train, y_train), (x_test, y_test) = Thunderbird.loadDataset(struct_log,
                                                                   train_ratio=0.5)
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = SVM()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
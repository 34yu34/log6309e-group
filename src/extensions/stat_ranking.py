


# Import for Cross-Validation
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class ModelData:
        
        def __init__(self, train_x, train_y, test_x, test_y) -> None:
                self.train_x = train_x
                self.train_y = train_y
                self.test_x = test_x
                self.test_y = test_y
        
        def cross_validate(self, folds, model_eval_lambda, is_random_based = True):
                
                kf = KFold(n_splits=folds, shuffle=is_random_based, random_state=42)
                
                metrics = []

                for train_index, test_index in kf.split(self.train_x):
                        x_train, x_test = self.train_x[train_index],  self.train_x[test_index]
                        y_train, y_test = self.train_y[train_index],  self.train_y[test_index]
                        
                        print(x_train, y_train)
                        
                        precision, recall, f1 = model_eval_lambda(x_train, y_train, x_test, y_test)
                        
                        metrics.append((precision, recall, f1))
                
                return metrics


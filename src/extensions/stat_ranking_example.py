# thios code was copied from : https://xai4se.github.io/defect-prediction/model-ranking.html 


## Load Data and preparing datasets

# Import for Load Data
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

# Import for Split Data into Training and Testing Samples
from sklearn.model_selection import train_test_split

# Import for Construct Defect Models (Classification)
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forests
from sklearn.tree import DecisionTreeClassifier # C5.0 (Decision Tree)
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Machine (GBM)
import xgboost as xgb # eXtreme Gradient Boosting Tree (xGBTree)

# Import for Cross-Validation
from sklearn.model_selection import cross_val_score

train_dataset = pd.read_csv(("../../datasets/lucene-2.9.0.csv"), index_col = 'File')
test_dataset = pd.read_csv(("../../datasets/lucene-3.0.0.csv"), index_col = 'File')

outcome = 'RealBug'
features = ['OWN_COMMIT', 'Added_lines', 'CountClassCoupled', 'AvgLine', 'RatioCommentToCode']

# process outcome to 0 and 1
train_dataset[outcome] = pd.Categorical(train_dataset[outcome])
train_dataset[outcome] = train_dataset[outcome].cat.codes

test_dataset[outcome] = pd.Categorical(test_dataset[outcome])
test_dataset[outcome] = test_dataset[outcome].cat.codes

X_train = train_dataset.loc[:, features]
X_test = test_dataset.loc[:, features]

y_train = train_dataset.loc[:, outcome]
y_test = test_dataset.loc[:, outcome]


# commits - # of commits that modify the file of interest
# Added lines - # of added lines of code
# Count class coupled - # of classes that interact or couple with the class of interest
# LOC - # of lines of code
# RatioCommentToCode - The ratio of lines of comments to lines of code
features = ['nCommit', 'AddedLOC', 'nCoupledClass', 'LOC', 'CommentToCodeRatio']

X_train.columns = features
X_test.columns = features
training_data = pd.concat([X_train, y_train], axis=1)
testing_data = pd.concat([X_test, y_test], axis=1)


cv_kfold = 10
model_performance_df = pd.DataFrame()
## Construct defect models and generate the 10-fold Cross Validation AUC

# Logistic Regression
lr_model = LogisticRegression(random_state=1234)
model_performance_df['LR'] = cross_val_score(lr_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# Random Forests
rf_model = RandomForestClassifier(random_state=1234, n_jobs = 10)
model_performance_df['RF'] = cross_val_score(rf_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# C5.0 (Decision Tree)
dt_model = DecisionTreeClassifier(random_state=1234)
model_performance_df['DT'] = cross_val_score(dt_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# Neural Network
nn_model = MLPClassifier(random_state=1234)
model_performance_df['NN'] = cross_val_score(nn_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# Gradient Boosting Machine (GBM)
gbm_model = GradientBoostingClassifier(random_state=1234)
gbm_model.fit(X_train, y_train)  
model_performance_df['GBM'] = cross_val_score(gbm_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# eXtreme Gradient Boosting Tree (xGBTree)
xgb_model = xgb.XGBClassifier(random_state=1234)
model_performance_df['XGB'] = cross_val_score(xgb_model, X_train, y_train, cv = cv_kfold, scoring = 'roc_auc')

# export to csv, display, and visualise the data frame
model_performance_df.to_csv('model_performance.csv', index = False)
display(model_performance_df)
model_performance_df.plot(kind = 'box', ylim = (0, 1), ylabel = 'AUC')
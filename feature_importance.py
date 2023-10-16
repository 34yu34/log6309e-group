from splitdatabgl import split_bgl   
from models.traditional import SVM
from models.traditional import DecisionTree
from models.traditional import LR
from extensions.stat_ranking import ModelData
import pandas as pd
from sklearn.inspection import permutation_importance

train_x, train_y, test_x, test_y = split_bgl()

def lr_model_eval(x_train, y_train, x_test, y_test):
    lr = LR(max_iter=100000)
    lr.fit(train_x, train_y)
    return lr.evaluate(test_x, test_y)

def decision_tree_model_eval(x_train, y_train, x_test, y_test):
    decision_tree = DecisionTree()
    decision_tree.fit(train_x, train_y)
    return decision_tree.evaluate(test_x, test_y)

def SVM_model_eval(x_train, y_train, x_test, y_test):
    svm = SVM(train_x, train_y, test_x, test_y)
    return svm.evaluate()
  
decision_tree = DecisionTree()
decision_tree.fit(train_x, train_y)

print ("decision_tree.classifier.feature_importances_ : ")
print(decision_tree.classifier.feature_importances_)

result = permutation_importance(
    decision_tree.classifier, test_x, test_y, n_repeats=10, random_state=42, n_jobs=2
)

print ("result.importances_mean : ")
print (result.importances_mean)


forest_importances = pd.Series(result.importances_mean)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
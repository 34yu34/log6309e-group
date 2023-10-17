from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1


class DecisionTree(object):

    def __init__(self, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See DecisionTreeClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                      max_features=max_features, class_weight=class_weight)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1


class SVM(object):

    def __init__(self, kernel="poly", degree=3, class_weight='balanced', coef0=1, C=10, iterations=5):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection
        """
        self.classifier = svm.SVC(kernel=kernel, degree=degree, class_weight=class_weight, coef0=coef0, C=C)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
            y: ndarray, the label vector of shape (num_instances,)
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
            y_true: ndarray, the label vector of shape (num_instances,)
        """
        prec_l = []
        recall_l = []
        f1_l = []

        for i in range(self.iterations):
            y_pred = self.predict(X)
            precision, recall, f1 = metrics(y_true, y_pred)
            print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
            prec_l.append(precision)
            recall_l.append(recall)
            f1_l.append(f1)
        precision, recall, f1 = sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l)
        print('average: ', sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l))
        return precision, recall, f1


class LR(object):

    def __init__(self, penalty='l2', C=100, tol=0.01, class_weight=None, max_iter=100):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection
        """
        self.classifier = LogisticRegression(penalty=penalty, C=C, tol=tol, class_weight=class_weight,
                                             max_iter=max_iter)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

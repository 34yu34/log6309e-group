


# Import for Cross-Validation
import pandas as pd
from sklearn.model_selection import cross_val_score






def stat_rank(performance : pd.DataFrame, name: str, model, train_x, train_y, k_fold : int):
    
        # scoring methods at link : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # higher return values are better than lower return values.

        performance[name] = cross_val_score(model, train_x, train_y, cv = k_fold, scoring = 'roc_auc')
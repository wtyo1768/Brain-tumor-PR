from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import cross_validate
from loader import numerical_loader
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from cfg import *


class CLF(RandomForestClassifier):
    def predict(self,X, **kwargs):
        super().__init__(**kwargs)
        threshold = .5
        result = self.predict_proba(X, **kwargs)
        return [1 if p>threshold else 0 for p in result[:,1]]



def run():
        clf = lgb.LGBMClassifier(
                # class_weight='balanced',
                max_depth=-1,
                num_leaves=31,
                objective='binary',
                n_estimators=100,
        )
        # clf = LogisticRegression(
        #         solver='liblinear',
        #         class_weight='balanced',
        #         max_iter=100,
        #         # random_state=seed
        # )
        # clf = CLF(
        #         solver='liblinear',
        #         class_weight='balanced',
        #         max_iter=-1,
        #         # threshold=.5
        # )
        # clf = CLF(
        #         class_weight='balanced',
        #         max_depth=3,
        #         n_estimators=25,
        #         # threshold=.5
        # )
        # clf = NuSVC(class_weight='balanced', max_iter=-1, nu=.25)

        scoring = ['accuracy', 'precision', 'recall', 'roc_auc']
        return cross_validate(clf, df, y_df, cv=K_FOLD, scoring=scoring, return_train_score=True)


if __name__=='__main__':
        df, y_df  = numerical_loader()
        
        metric = [ 'accuracy', 'precision', 'recall', 'roc_auc']
        metric = ['train_' + val for val in metric] + ['test_' + val for val in metric]
        
        total_result = {}
        for m in metric: total_result[m] = []
        
        for i in range(5):
                result = run()
                for m in metric: total_result[m].append(result[m])
        
        for m in metric[4:]:  print(m, ':', np.mean(total_result[m])) 
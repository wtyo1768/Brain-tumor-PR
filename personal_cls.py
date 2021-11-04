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
#     def __init__(self, threshold=0.5, **kwargs):
#         super().__init__(**kwargs)
#         self.threshold = threshold

    def predict(self,X, **kwargs):
        super().__init__(**kwargs)
        threshold = .5
        result = self.predict_proba(X, **kwargs)
        return [1 if p>threshold else 0 for p in result[:,1]]


df = numerical_loader()
y_df = df.pop('Progression/Recurrence (Yes:1 No:0)')
y_df = y_df.astype('int32')
df = df.fillna(df.mean())


print('Number of Patients and feature :', df.shape[0], df.shape[1], )
print('Number of Case 1 and Case 2 :', y_df[y_df==0].shape[0],  y_df[y_df==1].shape[0])
print('-------------------------')
print("Using feature:")
for i, c in enumerate(df.columns):
        print(i,'|', c)
print('-------------------------')

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
df = sel.fit_transform(df)

def run():
        clf = lgb.LGBMClassifier(
                class_weight='balanced',
                max_depth=-1,
                num_leaves=30,
                objective='binary',
                n_estimators=20,
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


if False:
        param = {'num_leaves': 10, 'objective': 'binary'}
        param['metric'] = ['auc', 'average_precision']
        num_round = 10
        train_data = lgb.Dataset(df, label=y_df)
        lgb.cv(param, train_data, num_round, nfold=5)
        
else:

        metric = [ 'accuracy', 'precision', 'recall', 'roc_auc']
        metric = ['train_' + val for val in metric] + ['test_' + val for val in metric]

        total_result = {}
        for m in metric: total_result[m] = []

        for i in range(5):
                result = run()
                for m in metric: total_result[m].append(result[m])

        for m in metric:  print(m, ':', np.mean(total_result[m])) 

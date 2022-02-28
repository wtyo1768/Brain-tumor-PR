from matplotlib.pyplot import cla
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import cross_validate
from loader import numerical_loader
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE 
from cfg import *
import random
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from sklearn import feature_selection
from sklearn.pipeline import Pipeline


def run(X, y, clf, ros=False, thr=.5, prob=False, seed=None, ):

    met =  ['accuracy', 'precision', 'recall', 'roc_auc', 'fpr', 'tpr']
    metric = ['train_' + val for val in met] + ['test_' + val for val in met]
    result = {}
    for m in metric:result[m] = []
    
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        if ros: X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)
        
        if prob:
                y_pred = clf.predict_proba(X_test)
                fpr, tpr, _ = metrics.roc_curve(y_test, y_pred[:, 1])
                y_pred = [1 if logit>thr else 0 for logit in y_pred[:, 1]]
                # print(y_pred[:, 1])
        else:   
                y_pred = clf.predict(X_test)
                y_score = clf.decision_function(X_test)
                fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
                
        result['test_accuracy'].append(metrics.accuracy_score(y_test, y_pred, ),)
        result['test_precision'].append(metrics.precision_score(y_test, y_pred, zero_division=1),)
        result['test_recall'].append(metrics.recall_score(y_test, y_pred, zero_division=1),)
        result['test_roc_auc'].append(metrics.roc_auc_score(y_test, y_pred),)
        result['test_fpr'].append(fpr)
        result['test_tpr'].append(tpr)
    return result


clf = lgb.LGBMClassifier(
   # boosting_type='dart',
   max_depth=-1,
   num_leaves=6,
   objective='binary',
   n_estimators=25,
   # class_weight='balanced'
)

clf = Pipeline([
    ('selector', SelectPercentile(feature_selection.f_classif, percentile=50)), 
    ('lgb', clf)
])

if __name__=='__main__':
        df, y_df  = numerical_loader()
        
        metric = [ 'accuracy', 'precision', 'recall', 'roc_auc']
        metric = ['train_' + val for val in metric] + ['test_' + val for val in metric]
        
        
        total_result = {}
        for m in metric: total_result[m] = []
        
        # SEEDS = random.sample(range(1, 100), NUM_RANDOM_STATE)
        for i in range(NUM_RANDOM_STATE):
                print(df.shape)
                result = run(df, y_df, clf, seed=SEEDS[i], prob=True, thr=.5
                        #      , ros=True
                )
                for m in metric: total_result[m].append(result[m])
        
        for m in metric[4:]:  print(m, ':', np.mean(total_result[m])) 
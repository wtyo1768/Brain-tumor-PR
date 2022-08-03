
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE 
from cfg import *


'''
description: helper training function for cross validation  
param {array} X 
param {array} y
param {*} clf: classifier
param {bool} ros: resampling
param {float} thr: threshold of predicting  
param {bool} prob: using predict_proba function when prediciton 
param {int} seed: random seed 
param {bool} slow_imp: toggle importance calculator
param {array} feature_imp: feature importance of tree model
return {dict}: matric
'''
def run(X, y, clf, ros=False, thr=.5, prob=False, seed=None, slow_imp=False, feature_imp=None):
    
    met =  ['accuracy', 'precision', 'recall', 'roc_auc', 'fpr', 'tpr']
    metric = ['train_' + val for val in met] + ['test_' + val for val in met]
    result = {}
    for m in metric:result[m] = []
    
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(X, y):
        
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        if ros: 
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        
        clf.fit(X_train, y_train)
        
        if slow_imp:
            feature_imp += clf['lgb'].feature_importances_
        if prob:
                y_pred = clf.predict_proba(X_test)
                fpr, tpr, _ = metrics.roc_curve(y_test, y_pred[:, 1])
                y_pred = [1 if logit>thr else 0 for logit in y_pred[:, 1]]
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
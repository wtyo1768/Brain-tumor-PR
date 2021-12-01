# from comet_ml import Experiment
from comet_ml import OfflineExperiment
import pandas as pd 
from cfg import *
import cv2
import numpy as np
from loader import data_pipe
from skimage.feature import texture
import glob
from personal_cls import run
import lightgbm as lgb
import random

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def glcm_features(f):
    img = cv2.imread(f, 0)
    g = texture.greycomatrix(img, [1, 2, 3, 4, 5, 6, 7], [0, np.pi/2], levels=256, normed=True, symmetric=True)
    features = []
    for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        f = texture.greycoprops(g, p)
        features.append(f)
    return np.array(features).flatten()


labels = [0, 1]
dtype = ['T1', 'T1c', 'T2', 'Flair']
# dtype = ['Flair']

rawX, y = [], []
for label in labels:
    pr_class = 'PR' if label else 'non_PR'
    for f in glob.glob(f'{segmented_img_dir}/{pr_class}/T1c/*.jpg'):
        mri_features = []
        # for d in ['T1c']:
        for d in dtype:
            textual_feature = glcm_features(f.replace('T1c', d))
            mri_features.append(textual_feature)
            
        rawX.append(np.array(mri_features).flatten())
        y.append(label)


X, y = np.array(rawX), np.array(y)
print(X.shape, y.shape)

thr = .8
X = VarianceThreshold(threshold=(thr * (1 - thr))).fit_transform(X)

clf = NuSVC( max_iter=-1, nu=.1, kernel='rbf', degree=4, 
            probability=True,
            class_weight='balanced', 
            )
clf_name = clf.__class__.__name__
# clf = lgb.LGBMClassifier(
#     # boosting_type='dart',
#     max_depth=-1,
#     num_leaves=6,
#     class_weight='balanced', 
#     objective='binary',
#     n_estimators=10,
# )

print('After VT transform | dim', X.shape[1])
clf = Pipeline(
    [('scaler', SelectPercentile(chi2, percentile=50)), 
     ('svm', clf)]
)

#TODO plot roc curve to comet ml

if __name__=='__main__':
    
    experiment = OfflineExperiment(
        api_key=COMET_APT_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORK_SPACE,
        display_summary_level=0,
        # disabled=True,
    )
    experiment.add_tags(['Img', clf_name])
    if len(dtype)>1 : experiment.add_tag('Mixed')
    else: experiment.add_tag(dtype[0])

    met =  ['accuracy', 'precision', 'recall', 'roc_auc']
    metric = ['test_' + val for val in met]

    total_result = {}
    for m in metric: total_result[m] = []

    seeds = random.sample(range(1, 100), NUM_RANDOM_STATE)
    for i in range(NUM_RANDOM_STATE):
        result = run(
            X, y, clf, thr=.55, seed=seeds[i],
            prob=True,
            ros=True
        )
        for m in metric: total_result[m].append(result[m])
        print(result['test_fpr'])
        experiment.log_curve(f"roc-curve{i}", result['test_fpr'], result['test_tpr'], step=i)

    for m in met: 
        print(m ,'|\t',
            #   np.mean(result[f'train_{m}']),
              np.mean(result[f'test_{m}']))
        experiment.log_metric(m, np.mean(result[f'test_{m}']))
        
    experiment.log_parameters({'seeds':seeds})
    experiment.end()

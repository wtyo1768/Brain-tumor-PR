from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import cross_validate
from loader import numerical_loader
import numpy as np


df = numerical_loader()
y_df = df.pop('Progression/Recurrence (Yes:1 No:0)')
df = df.fillna(df.mean())

print('Number of Patients and feature :', df.shape[0], df.shape[1], )
print('Number of Case 1 and Case 2 :', y_df[y_df==0].shape[0],  y_df[y_df==1].shape[0])

clf = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=100,
        # random_state=seed
)
# clf = NuSVC(class_weight='balanced', max_iter=100, nu=.3)


scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(clf, df, y_df, cv=10, scoring=scoring, return_train_score=True)

for metric in [ 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(metric, ':', np.mean(scores[metric])) 


import numpy as np
import itertools
import csv
import random

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##################
# parameters
##################

num_try = 100

##################
# training data
##################

csv_file = open("data.csv", "r")

h = next(csv.reader(csv_file))

x_compositions = []
t1 = []

for row in csv.reader(csv_file):
    x_compositions.append(row[1])

    if float(row[2]) < 10.0:
        t1.append(0)
    else:
        t1.append(1)


x_elements = []

for i in range(len(x_compositions)):

    x_elements.append([x_compositions[i][3:5], x_compositions[i][5:7], x_compositions[i][7:9]])


x_elements_all = itertools.chain.from_iterable(x_elements)

elements = list(set(itertools.chain.from_iterable(x_elements)))
elements.sort()

X = []

for i in range(len(x_elements)):

    each_bits = []

    for j in range(len(elements)):

        if elements[j] in x_elements[i]:
            each_bits.append(1)
        else:
            each_bits.append(0)

    X.append(each_bits)


X_train = X
y_train = t1


total_accuracy = []

for k in range(num_try):

    kf = KFold(n_splits=5, shuffle=True, random_state=111 + k)
    kf.get_n_splits(X_train)

    true_y = []
    pred_y = []

    for train_index, test_index in kf.split(X_train):

        diparameter={"n_estimators":[i for i in range(10,50,10)],"max_depth":[i for i in range(1,10,1)]}

        clf=GridSearchCV(RandomForestClassifier(),param_grid=diparameter,cv=5,n_jobs=1)

        clf.fit(np.array(X_train)[train_index], np.array(y_train)[train_index])

        true_y.extend(list(np.array(y_train)[test_index]))
        pred_y.extend(list(clf.predict(np.array(X_train)[test_index])))

    total_accuracy.append(accuracy_score(true_y,pred_y))


print("Accuracy mean = ", np.mean(total_accuracy))
print("Accuracy std = ", np.std(total_accuracy))


rf = RandomForestClassifier(n_estimators = clf.best_params_['n_estimators'], max_depth = clf.best_params_['max_depth']) 

rf.fit(X_train, y_train)

fea_rf_imp = pd.DataFrame({'imp': rf.feature_importances_, 'col': elements})
fea_rf_imp = fea_rf_imp.sort_values(by='imp', ascending=False)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Set1").colors)
plt.figure(figsize = (6.4, 4.8))
sns.barplot('col', 'imp', data = fea_rf_imp)
plt.ylabel('Features', fontsize=18)
plt.xlabel('Importance', fontsize=18)

plt.savefig("Importance_descriptors.pdf")

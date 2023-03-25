import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

cancer = load_breast_cancer()

dataset = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
# print(dataset.head())

x = dataset
y = cancer['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

model = SVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

grid_params = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), grid_params, verbose=3)
grid.fit(x_train, y_train)
print(grid.best_params_)
g_pred = grid.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, g_pred))

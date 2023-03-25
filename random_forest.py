import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kyphosis.csv')
sns.pairplot(data, hue='Kyphosis')
x = data.drop(['Kyphosis'], axis=1)
y = data['Kyphosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# decision tree classifier
decision = DecisionTreeClassifier()
decision.fit(x_train, y_train)
dec_pred = decision.predict(x_test)
print(confusion_matrix(y_test, dec_pred))

# random forest classifier
forest = RandomForestClassifier(n_estimators=200)
forest.fit(x_train, y_train)
for_pred = forest.predict(x_test)
print(confusion_matrix(y_test, for_pred))

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('USA_Housing.csv')
# print(df.describe())
# sns.displot(df["Price"], kde=True)

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]  # features that has to be taken into account
y = df["Price"]  # target that has to be predicted

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)  # splitting the data
lm = LinearRegression()  # creating the estimator, essentially an object for the model
lm.fit(x_train, y_train)  # fitting the data in the model
coeff = pd.DataFrame(lm.coef_, index=x_train.columns, columns=["Coeff"])  # getting coefficient for each feature against price

predictions = lm.predict(x_test)
print(predictions)

plt.scatter(y_test, predictions)
sns.displot((y_test-predictions))

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.show()

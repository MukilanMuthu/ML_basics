import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('titanic_train.csv')
# sns.heatmap(train.isnull(), cbar=False)

# sns.countplot(data=train, x='Survived', hue='Pclass')
# sns.histplot(data=train, x='Fare', bins=40)


def impute_age(df):
    age = df[0]
    p_class = df[1]
    data = pd.DataFrame(df)
    if pd.isnull(age):
        if p_class == 1:
            return 37
        elif p_class == 2:
            return 29
        else:
            return 24

    else:
        return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop(['Cabin'], axis=1, inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(), cbar=False)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
train.drop(['Sex', 'Name', 'Embarked', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print(train)

plt.show()

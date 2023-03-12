import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dn = pd.read_csv('histogram2.csv', header=None)
print(dn)
sns.displot(dn.iloc[1], kde=True)
sns.jointplot(x=dn[0], y=dn.iloc[0], kind='reg')
# sns.lmplot(x=dn[0], y=dn[1], data=dn)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
print(data.keys())
df = pd.DataFrame(data['data'], columns=data['feature_names'])
print(df.head())

std = StandardScaler()
std.fit(df)
scaled_data = std.transform(df)

# PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
X_pca = pca.transform(scaled_data)  # reduces 30 features to 2 features with maximum variance
print(X_pca.shape)

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=data['target'])
plt.show()

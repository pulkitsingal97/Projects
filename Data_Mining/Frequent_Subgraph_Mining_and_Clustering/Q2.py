import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import sys

input_file = sys.argv[1]
dim = int(sys.argv[2])
plot_name = sys.argv[3]

columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
df = pd.read_csv(input_file, sep=' ', header=None, names=columns[0:dim], engine='python')
df['c_id'] = 0

X_train, X_test, y_train, y_test = train_test_split(df[columns[0:dim]], df[['c_id']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

K = range(1, 16)
dist = []

for k in K:
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42, n_init=10)
    kmeans.fit(X_train_norm) 
    dist.append(kmeans.inertia_)

plt.plot(K, dist)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.savefig(plot_name)
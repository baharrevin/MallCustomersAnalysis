

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Yüksek boyutlu verileri 2 boyuta indirerek görselleştirmek için

data = pd.read_csv("mall_customers.csv")

X = data[['Annual Income (k$)','Spending Score (1-100)']]


wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
# grafikte oluşan “dirsek” (elbow) noktası, ideal küme sayısını gösterir

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)
clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

# K-Means algoritmasıyla oluşturulan müşteri kümelerini ve her kümenin merkezini görselleştirir

plt.figure(figsize=(8,6))
colors = ['red','green','blue','cyan','purple']

for i in range(5):
    plt.scatter(X[clusters == i]['Annual Income (k$)'],
                X[clusters == i]['Spending Score (1-100)'],
                c=colors[i], label=f"Cluster {i}")

plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=300, c='yellow', label='Centers',marker='X')

plt.xlabel("Annual Revenue (k$)")
plt.ylabel("Spend Score")
plt.title("Customer Segmentation - KMeans")
plt.legend()
plt.grid(True)
plt.show()


# ******** interpretation ********
"""
This chart defines 5 different customer groups according to annual income and spending score.

The behavioral profile of each group is different and decisions such as marketing, campaigns, pricing can be made based on data thanks to this analysis.

For cluster 0 (red): Income is medium - Spending score is also medium: It can be the core of the main target audience.
For cluster 1 (green): Income is high - Spending score is high: The company's most profitable and loyal customers
For cluster 2 (Blue): Income is low - Spending is very high: Spending tendency but income is low, discounted products, campaigns
For cluster 3 (Light Blue): Income is very high - Spending score is low: Marketing, Directing spending with campaigns
For cluster 4 (Purple): Income is low - Spending is low: Low priority customer group for the company

"""



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

data['PCA1'] = X_pca[:,0]
data['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))

for i in range(5):
    plt.scatter(data[data['Cluster'] == i]['PCA1'],
                data[data['Cluster'] == i]['PCA2'],
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Cluster Visualization with PCA")
plt.legend()
plt.grid(True)
plt.show()

# not: Diyelim ki elimizde 10 farklı özellik var (örneğin: yaş, gelir, boy, kilo, vb).
# PCA bunları, en fazla bilgi kaybı olmadan 2 tane yeni özellik haline getirir:
# PCA Component 1 ve PCA Component 2
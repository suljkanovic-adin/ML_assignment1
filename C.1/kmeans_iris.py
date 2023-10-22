from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

#2.
data = pd.read_csv("./iris.csv")

#selecting rows, excluding the last column, because kMeans doesnt use labels
X = data.iloc[:, :-1].values

#kMeans clustering applied, using 3 clusters because of 3 known distinct species in iris
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

#excluding the last two columns, label and cluster
properties = data.columns[:-2]

plt.figure(figsize=(15, 10))
plot_number = 1

for i in range(len(properties)):
    for j in range(i+1, len(properties)):
        plt.subplot(2, 3, plot_number)
        plt.scatter(data[properties[i]], data[properties[j]], c=data['Cluster'], cmap='rainbow', edgecolor='k', s=60)
        plt.xlabel(properties[i])
        plt.ylabel(properties[j])
        plt.title(f"{properties[i]} vs {properties[j]}")
        plot_number += 1

plt.tight_layout()
plt.show()

#3.
labels = []

for cluster in range(3):  
    true_labels = data[data['Cluster'] == cluster]['species']
    most_common_label = true_labels.mode().iloc[0]
    labels.append(most_common_label)

data['MappedCluster'] = data['Cluster'].apply(lambda x: labels[x])

accuracy = accuracy_score(data['species'], data['MappedCluster'])
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(data['species'], data['MappedCluster'])
print("Confusion Matrix:")
print(conf_matrix)



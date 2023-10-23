import numpy as np
import pandas as pd

def initialize_centres(X, k):
    #randomly initialize the centres
    random_idx = np.random.choice(X.shape[0], k, replace=False)
    centres = X[random_idx, :]
    return centres

def assign_clusters(X, centres):
    #assign each data point to the nearest centroid
    distances = np.sqrt(((X - centres[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centres(X, labels, k):
    #update the centres based on the mean of the points assigned to them
    new_centres = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centres

def kmeans(X, k, max_iters=100):
    #K-means clustering
    centres = initialize_centres(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centres)
        new_centres = update_centres(X, labels, k)
        if np.all(centres == new_centres):
            break
        centres = new_centres
    return labels, centres

data = pd.read_csv("./iris.csv")
X = data.iloc[:, :-1].values

#executing the "home-made" K-Means clustering
labels, centres = kmeans(X,3)




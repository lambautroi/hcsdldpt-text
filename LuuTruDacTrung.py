from sklearn.cluster import KMeans
from Class import Cluster
import jsonpickle as json

def kmeans(data):
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(data)
    return model.labels_, model.cluster_centers_

def ClusterUseKmeans(features):
    data = [f.feature for f in features]
    labels, centers = kmeans(data)
    clusters = []
    for i, center in enumerate(centers):
        cluster_features = [features[j] for j in range(len(labels)) if labels[j] == i]
        clusters.append(Cluster(center=center, features=cluster_features))
    return clusters

def save(data):
    with open('metadata/data.json', 'w') as f:
        f.write(json.dumps(data))

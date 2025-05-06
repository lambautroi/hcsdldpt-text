import numpy as np
from sklearn.cluster import KMeans
import json

class VanBan:
    def __init__(self, path, feature):
        self.path = path
        self.feature = feature
        self.cluster = None

def ClusterUseKmeans(features, k=3):
    if not features:
        return []
    features = np.array(features)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(features)
    clusters = []
    for i, f in enumerate(features):
        clusters.append({"center": int(labels[i]), "features": f.tolist()})
    return clusters

def LuuJSON(data, file_path='metadata/data.json'):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

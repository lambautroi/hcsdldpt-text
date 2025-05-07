class Feature:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

class Cluster:
    def __init__(self, id, features, label):
        self.id = id
        self.features = features
        self.label = label

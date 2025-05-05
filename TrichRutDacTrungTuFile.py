import TrichRutDacTrung as ft
import os
from Class import Feature
import LuuTruDacTrung as luu

folder_path = 'data'
files = os.listdir(folder_path)
listFeatures = []

for file in files:
    full_path = os.path.join(folder_path, file)
    feature_vector = ft.features(full_path)[0]
    feature = Feature(link=full_path, feature=feature_vector)
    listFeatures.append(feature)

Clusters = luu.ClusterUseKmeans(listFeatures)
luu.save(Clusters)

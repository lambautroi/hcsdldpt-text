from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def SimilarityCalculation(clusters, features):
    labelCount = []

    for feature in features:
        max_sim = -1
        best_cluster = None
        for cluster in clusters:
            sim = cosine_similarity([feature], [cluster.center])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_cluster = cluster

        for f in best_cluster.features:
            sim = cosine_similarity([feature], [f.feature])[0][0]
            if sim > 0.2:  # ngưỡng lọc
                labelCount.append((f.link, sim))

    df = pd.DataFrame(labelCount, columns=['f_link', 'score'])
    top_3_links = df.groupby('f_link')['score'].mean().nlargest(3).index.tolist()
    return top_3_links

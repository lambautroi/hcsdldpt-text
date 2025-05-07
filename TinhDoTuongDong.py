from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def SimilarityCalculation(clusters, feature):
    # Đảm bảo đặc trưng đầu vào là numpy 2D
    feature = np.array(feature).reshape(1, -1)

    similarities = []
    for cluster in clusters:
        vector = np.array(cluster["feature"]).reshape(1, -1)
        score = cosine_similarity(feature, vector)[0][0]
        similarities.append((cluster["link"], score))

    # Sắp xếp giảm dần theo độ tương đồng
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Trả về top 3 file giống nhất + độ tương đồng
    top_links = [f"{link} (score: {round(score, 3)})" for link, score in similarities[:3]]
    return top_links

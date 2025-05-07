import json
from sklearn.cluster import KMeans
import numpy as np

def ClusterUseKmeans(features_array, n_clusters=5):
    print("🔄 Đang tiến hành phân cụm KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(features_array)
    labels = kmeans.labels_
    print("✅ Phân cụm xong.")
    return labels.tolist()

def LuuDanhSachDacTrungVaNhom(features_array, labels, filename, file_paths=None):
    print(f"💾 Đang lưu kết quả vào {filename}...")

    if file_paths is None:
        file_paths = [f"file_{i}.txt" for i in range(len(labels))]

    data = [
        {
            "id": idx,
            "feature": features_array[idx].tolist(),
            "label": labels[idx],
            "link": file_paths[idx]
        }
        for idx in range(len(labels))
    ]
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Lưu thành công.")

import os
from TrichRutDacTrung import TrichRutDacTrung
from TienXuLy import preprocess_text
import LuuTruDacTrung as luu

folder = "data"
extractor = TrichRutDacTrung()

list_features = []
file_names = []

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        path = os.path.join(folder, filename)
        print(f"Đang xử lý {path}...")
        raw = extractor.read_pdf(path)
        if not raw.strip():
            print(f"⚠ Bỏ qua {filename} vì không đọc được nội dung.")
            continue
        clean = preprocess_text(raw)
        vector = extractor.extract_all_features([clean])
        list_features.append(vector[0])
        file_names.append(filename)

# Clustering
clusters = luu.ClusterUseKmeans(list_features)
luu.LuuJSON(clusters)
print("✅ Hoàn tất. Kết quả lưu vào data.json")

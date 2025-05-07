import os
import TrichRutDacTrung as td
import LuuTruDacTrung as luu

DATA_FOLDER = "data"
OUTPUT_FILE = "metadata/data.json"
N_CLUSTERS = 5  # Tuỳ chỉnh nếu muốn thay đổi số cụm

# Bước 1: Đọc toàn bộ văn bản trong thư mục data/
print("📁 Đang load các file từ thư mục data/...")
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".txt", ".pdf"))]
file_paths = [os.path.join(DATA_FOLDER, f) for f in files]

texts = []
for file_path in file_paths:
    try:
        text = td.extract_text_from_file(file_path)
        clean = td.clean_text(text)
        texts.append(clean)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")

# Bước 2: Trích rút đặc trưng từ văn bản
features_array = td.extract_all_features(texts)

# Bước 3: Phân cụm
labels = luu.ClusterUseKmeans(features_array, n_clusters=N_CLUSTERS)

# Bước 4: Lưu đặc trưng + nhãn cụm + link
luu.LuuDanhSachDacTrungVaNhom(
    features_array,
    labels,
    OUTPUT_FILE
)

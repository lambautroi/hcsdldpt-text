import os
from pathlib import Path
from TrichRutDacTrung import extract_features_from_file
from LuuTruDacTrung import ClusterUseKmeans, LuuDanhSachDacTrungVaNhom

DATA_DIR = "data"
OUTPUT_FILE = "metadata/data.json"

def main():
    print("🚀 Bắt đầu trích rút đặc trưng từ thư mục 'data/'...\n")
    
    all_files = []
    list_features = []

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt") or filename.endswith(".pdf"):
            print(f"📄 Đang xử lý: {filename}")
            try:
                features = extract_features_from_file(filepath)
                list_features.append(features)
                all_files.append(filename)  # chỉ lấy tên file
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {filename}: {e}")

    print(f"\n✅ Đã trích rút đặc trưng cho {len(list_features)} văn bản.")

    if not list_features:
        print("⚠️ Không có đặc trưng nào được trích rút. Thoát.")
        return

    # Chuẩn hóa kích thước nếu cần (chắc ăn)
    import numpy as np
    features_array = np.array(list_features)
    if len(set(len(f) for f in list_features)) > 1:
        print("⚠️ Các đặc trưng có độ dài khác nhau. Đang chuẩn hóa lại...")
        from sklearn.preprocessing import StandardScaler
        features_array = StandardScaler().fit_transform(features_array)
    print(f"✅ Đặc trưng sau chuẩn hóa, shape: {features_array.shape}")

    # Clustering : Phân cụm
    labels = ClusterUseKmeans(features_array)

    # Lưu đặc trưng + nhãn cụm + link
    LuuDanhSachDacTrungVaNhom(features_array, labels, OUTPUT_FILE, file_paths=all_files)

if __name__ == "__main__":
    main()

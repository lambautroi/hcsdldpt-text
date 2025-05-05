# 📘 HỆ THỐNG NHẬN DẠNG VĂN BẢN TƯƠNG ĐỒNG (PDF)

## 🧠 Mục tiêu

Hệ thống này giúp **tìm ra các văn bản PDF có nội dung tương đồng cao** với một file PDF đầu vào. Dùng nhiều đặc trưng NLP như:

-   Bag of Words
-   TF-IDF
-   Word2Vec
-   Topic Modeling (LDA)
-   POS Tags
-   Passive Voice Count

---

## 🗂️ Cấu trúc thư mục

📦project-root/
├── 📁data/ # Chứa các file .pdf mẫu
├── 📁metadata/ # Chứa file data.json sau khi lưu đặc trưng
├── 📄Class.py # Định nghĩa lớp Feature và Cluster
├── 📄TrichRutDacTrung.py # Xử lý đặc trưng văn bản
├── 📄TrichRutDacTrungTuFile.py# Trích xuất đặc trưng từ thư mục data
├── 📄TinhDoTuongDongTest.py # Tính độ tương đồng giữa văn bản
├── 📄LuuTruDacTrung.py # KMeans và lưu cluster
├── 📄Home.py # Giao diện GUI nhận dạng
├── 📄requirements.txt # Danh sách thư viện cần cài
└── 📄README.md # Tài liệu hướng dẫn

Cách Sử Dụng

1. Chuẩn Bị Dữ Liệu
   Thêm các file .pdf văn bản cần lưu đặc trưng vào thư mục data/

Chạy: python TrichRutDacTrungTuFile.py
=> Hệ thống sẽ trích xuất đặc trưng, gom cụm và lưu vào metadata/data.json.

2. Giao Diện Người Dùng
   Chạy: python Home.py
   Giao diện hiện lên ➤ chọn file .pdf ➤ hệ thống trả về Top 3 văn bản gần giống nhất, kèm nội dung hoặc chức năng mở file tương ứng.

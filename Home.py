from tkinter import Tk, Frame, Button, Label, filedialog
import TrichRutDacTrung as ft
import jsonpickle as json
import TinhDoTuongDong

# Đọc dữ liệu đã lưu (feature + link) từ metadata
with open('metadata/data.json', 'r', encoding='utf-8') as f:
    clusters = json.loads(f.read())

# Hàm xoá các widget cũ
def clear_widgets(parent):
    for widget in parent.winfo_children():
        widget.destroy()

# Hàm xử lý khi người dùng chọn văn bản
def nhanDang():
    file = filedialog.askopenfilename(filetypes=[("Text and PDF files", "*.txt *.pdf")])
    if file:
        clear_widgets(frame)
        Label(frame, text="🔍 Đang phân tích văn bản...", font=("Arial", 12)).pack()

        # Trích xuất đặc trưng
        features = ft.extract_features_from_file(file)

        # Tính độ tương đồng
        links = TinhDoTuongDong.SimilarityCalculation(clusters, features)

        # Hiển thị kết quả
        Label(frame, text="✅ Top 3 văn bản tương đồng nhất:", font=("Arial", 14)).pack(pady=10)
        for link in links:
            Label(frame, text=link, font=("Arial", 11)).pack()

# Thiết lập giao diện
window = Tk()
window.title("🧠 Nhận dạng văn bản")
window.geometry('600x400')
frame = Frame(window)
frame.pack(pady=10)
Button(window, text="📂 Chọn văn bản", command=nhanDang, font=("Arial", 12)).pack(pady=10)
window.mainloop()

from tkinter import Tk, Frame, Button, Label, filedialog
import TrichRutDacTrung as ft
import jsonpickle as json
import TinhDoTuongDong

with open('metadata/data.json', 'r') as f:
    clusters = json.loads(f.read())

def clear_widgets(parent):
    for widget in parent.winfo_children():
        widget.destroy()

def nhanDang():
    file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file:
        clear_widgets(frame)
        features = ft.features(file=file)
        links = TinhDoTuongDong.SimilarityCalculation(clusters, features)
        Label(frame, text="Top 3 văn bản tương đồng nhất:", font=("Arial", 14)).pack()
        for link in links:
            Label(frame, text=link).pack()

window = Tk()
window.title("Nhận dạng văn bản")
window.geometry('600x400')
frame = Frame(window)
frame.pack(pady=10)
Button(window, text="Chọn văn bản", command=nhanDang).pack(pady=10)
window.mainloop()

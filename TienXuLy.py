import re
import string
from underthesea import word_tokenize

def preprocess_text(text):
    # Chuyển lowercase
    text = text.lower()

    # Xóa số
    text = re.sub(r'\d+', '', text)

    # Xóa dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # Tách từ tiếng Việt
    text = word_tokenize(text, format="text")

    return text

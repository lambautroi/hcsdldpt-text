import os
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

POS_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
    "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB"
]

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

# Hàm tiền xử lý , làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_bow(texts):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(texts).toarray()

def extract_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

def extract_word2vec(texts):
    tokenized = [nltk.word_tokenize(text) for text in texts]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    vectors = []
    for sentence in tokenized:
        vectors.append(np.mean([model.wv[word] for word in sentence if word in model.wv] or [np.zeros(100)], axis=0))
    return np.array(vectors)

def extract_lda(texts, num_topics=5):
    tokenized = [nltk.word_tokenize(text) for text in texts]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    lda_vectors = []
    for doc in corpus:
        topic_dist = [0] * num_topics
        for topic_id, prob in lda.get_document_topics(doc):
            topic_dist[topic_id] = prob
        lda_vectors.append(topic_dist)
    return np.array(lda_vectors)

def extract_pos_tags(texts):
    pos_features = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        tag_counts = Counter(tag for _, tag in tags)
        total = sum(tag_counts.values())
        tag_freqs = [tag_counts.get(tag, 0) / total if total > 0 else 0 for tag in POS_TAGS]
        pos_features.append(tag_freqs)
    return np.array(pos_features)

def extract_passive_voice(texts):
    features = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        count = sum(1 for s in sentences if re.search(r'\b(be|is|are|was|were|been|being)\b\s+\w+ed\b', s))
        features.append([count / len(sentences) if sentences else 0])
    return np.array(features)

def extract_all_features(texts):
    print(f"🔍 Đang trích rút đặc trưng cho {len(texts)} văn bản...")
    bow = extract_bow(texts)
    tfidf = extract_tfidf(texts)
    w2v = extract_word2vec(texts)
    lda = extract_lda(texts)
    pos = extract_pos_tags(texts)
    passive = extract_passive_voice(texts)

    print("📏 Đang chuẩn hóa chiều đặc trưng...")
    min_len = min(map(len, [bow[0], tfidf[0], w2v[0], lda[0], pos[0], passive[0]]))
    all_features = np.concatenate([
        bow[:, :min_len],
        tfidf[:, :min_len],
        w2v[:, :min_len],
        lda[:, :min_len],
        pos[:, :min_len],
        passive[:, :min_len]
    ], axis=1)

    print(f"✅ Hoàn tất. Kích thước đặc trưng: {all_features.shape}")
    return all_features

def extract_features_from_file(filepath):
    """
    Trích rút đặc trưng từ 1 file duy nhất (.pdf hoặc .txt)
    Trả về vector đặc trưng 1D
    """
    raw_text = extract_text_from_file(filepath)
    if not raw_text.strip():
        print("⚠️ File rỗng hoặc không đọc được.")
        return None

    cleaned_text = clean_text(raw_text)

    print("🔍 Đang trích đặc trưng từ file đơn lẻ...")
    bow = extract_bow([cleaned_text])
    tfidf = extract_tfidf([cleaned_text])
    w2v = extract_word2vec([cleaned_text])
    lda = extract_lda([cleaned_text])
    pos = extract_pos_tags([cleaned_text])
    passive = extract_passive_voice([cleaned_text])

    min_len = min(len(bow[0]), len(tfidf[0]), len(w2v[0]), len(lda[0]), len(pos[0]), len(passive[0]))
    final_vector = np.concatenate([
        bow[:, :min_len],
        tfidf[:, :min_len],
        w2v[:, :min_len],
        lda[:, :min_len],
        pos[:, :min_len],
        passive[:, :min_len]
    ], axis=1)

    print(f"✅ Vector đặc trưng có shape: {final_vector.shape}")
    return final_vector[0]  # Trả về vector 1 chiều (1D)

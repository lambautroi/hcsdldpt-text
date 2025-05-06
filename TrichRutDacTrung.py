import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
import spacy

class TrichRutDacTrung:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def read_pdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            text += page_text
        if len(text.strip()) < 100:
            print("⚠️ Văn bản trích xuất quá ngắn!")
        return text

    def extract_bow_features(self, texts):
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(texts)
        return bow_matrix.toarray(), vectorizer.get_feature_names_out()

    def extract_tfidf_features(self, texts):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

    def extract_word2vec_features(self, texts, vector_size=100):
        tokenized_texts = [text.split() for text in texts]
        if not any(tokenized_texts):
            return np.zeros((len(texts), vector_size))
        model = Word2Vec(tokenized_texts, vector_size=vector_size, window=5, min_count=1, workers=4)
        doc_vectors = []
        for tokens in tokenized_texts:
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))
        return np.array(doc_vectors)

    def extract_lda_features(self, texts, num_topics=5):
        vectorizer = CountVectorizer(max_features=1000)
        tf_matrix = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_matrix = lda.fit_transform(tf_matrix)
        return lda_matrix

    def extract_pos_features(self, texts):
        pos_features = []
        for text in texts:
            doc = self.nlp(text)
            noun_count = verb_count = adj_count = 0
            for token in doc:
                if token.pos_ == "NOUN":
                    noun_count += 1
                elif token.pos_ == "VERB":
                    verb_count += 1
                elif token.pos_ == "ADJ":
                    adj_count += 1
            pos_features.append([noun_count, verb_count, adj_count])
        return np.array(pos_features)

    def extract_passive_voice_features(self, texts):
        passive_counts = []
        for text in texts:
            doc = self.nlp(text)
            passive_count = 0
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == "nsubjpass":
                        passive_count += 1
                        break
            passive_counts.append(passive_count)
        return np.array(passive_counts).reshape(-1, 1)

    def extract_all_features(self, texts):
        bow_features, _ = self.extract_bow_features(texts)
        tfidf_features, _ = self.extract_tfidf_features(texts)
        word2vec_features = self.extract_word2vec_features(texts)
        lda_features = self.extract_lda_features(texts)
        pos_features = self.extract_pos_features(texts)
        passive_features = self.extract_passive_voice_features(texts)

        # Xác định chiều tối đa cần thiết cho mỗi loại đặc trưng
        target_dim = 10  # Hoặc bất kỳ số nào đại ca muốn cố định

        def resize_features(features, target_dim):
            if features.shape[1] >= target_dim:
                return features[:, :target_dim]
            else:
                pad_width = target_dim - features.shape[1]
                return np.pad(features, ((0, 0), (0, pad_width)), 'constant')

        bow_features = resize_features(bow_features, target_dim)
        tfidf_features = resize_features(tfidf_features, target_dim)
        word2vec_features = resize_features(word2vec_features, target_dim)
        lda_features = resize_features(lda_features, target_dim)


        combined_features = np.hstack((
            bow_features,
            tfidf_features,
            word2vec_features,
            lda_features,
            pos_features,
            passive_features
        ))
        return combined_features

feature_extractor = TrichRutDacTrung()

def features(path):
    raw_text = feature_extractor.read_pdf(path)
    cleaned_text = raw_text.replace("\n", " ").strip()
    texts = [cleaned_text]
    all_features = feature_extractor.extract_all_features(texts)
    return all_features[0]
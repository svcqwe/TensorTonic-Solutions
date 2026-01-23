import numpy as np
import math
from collections import Counter

def tfidf_vectorizer(documents):
    # 1. Пустой список документов 
    if not documents:
        return np.array([]), []

    # 2. Токенизация 
    tokenized = []
    for doc in documents:
        if doc is None or doc.strip() == "":
            tokenized.append([])
        else:
            tokenized.append(doc.lower().split())

    # 3. Проверка: все документы пустые 
    if all(len(doc) == 0 for doc in tokenized):
        return np.array([]), []

    n_docs = len(tokenized)

    # 4. Vocabulary 
    vocab = sorted(set(word for doc in tokenized for word in doc))
    n_vocab = len(vocab)

    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # 5. Document Frequency 
    df = Counter()
    for doc in tokenized:
        for word in set(doc):
            df[word] += 1

    # 6. IDF (СТРОГО по формуле задания) 
    idf = {word: math.log(n_docs / df[word]) for word in vocab}

    # 7. TF-IDF Matrix 
    tfidf = np.zeros((n_docs, n_vocab), dtype=float)

    for i, doc in enumerate(tokenized):
        if not doc:
            continue

        counts = Counter(doc)
        total = len(doc)

        for word, count in counts.items():
            j = word_to_idx[word]
            tf = count / total
            tfidf[i, j] = tf * idf[word]

    return tfidf, vocab

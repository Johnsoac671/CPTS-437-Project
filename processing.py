from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        

    def vectorize(self, data):
        return self.model.encode(data).tolist()


def dot_product(a, b):
    return sum(
        [x[0] * x[1] for x in zip(a, b)]
        )
from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        

    def vectorize(self, data):
        return self.model.encode(data).tolist()



class Tweet:
    def __init__(self, label, vector):
        self.label = label
        self.vector = vector
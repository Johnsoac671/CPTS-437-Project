import pandas as pd
from processing import dot_product, Vectorizer

class SVM:
    
    def __init__(self, iterations, learning_rate, C):
        self.models = {}
        self.vectorizer = Vectorizer()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.C = C
        
    
    def predict(self, model, tweet):
        w, b = self.models[model]
        tweet = self.vectorizer.vectorize(tweet)
        
        approx = dot_product(w, tweet) - b
        
        if approx == 0:
            return 0
        else:
            return 1 if approx > 0 else -1
        
    
    def train(self, model, data):
        _, vector_length = data.shape
        
        w = [0 for weight in range(vector_length)]
        b = 0
        
        for _ in range(self.iterations):
            for vector in data.itertuples():
                dot = dot_product(vector.embedding, w)
                
                if (vector.sentiment * (dot - b)) >= 1:
                    w = [weight - (self.learning_rate * (2 * self.C * weight)) for weight in w]
                    
                else:
                    correction_dot = dot_product(vector.embedding, [vector.sentiment for _ in range(vector_length)])
                    w = [weight - (self.learning_rate * (2 * self.C * weight - correction_dot)) for weight in w]
                    b -= self.learning_rate * vector.sentiment
        
        self.models[model] = (w, b)

import pandas as pd
import ast
import numpy as np

from processing import dot_product, Vectorizer

class SVM:
    
    def __init__(self, iterations, learning_rate, C):
        self.models = {}
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.C = C
        
    
    def predict(self, tweet):
        
        labels = ["Positive", "Negative", "Neutral", "Irrelevant"]
        estimates = []
        
        for label in labels:
            w, b = self.models[label]
            
            approx = dot_product(w, tweet) - b
            
            estimates.append((label, approx))
        
        return max(estimates, key=lambda x: x[1])
        
    
    def train(self, target, data):
        vector_length = len(data.iloc[0]['embedding'] )
        
        w = np.zeros(vector_length)
        b = 0
        
        for _ in range(self.iterations):
            for vector in data.itertuples():
                vector_arr = np.array(vector.embedding)
                
                dot = np.dot(vector_arr, w)
                
                if (vector.sentiment * (dot - b)) >= 1:
                    w -= self.learning_rate * (2 * self.C * w)
                    
                else:
                    w -= self.learning_rate * (2 * self.C * w - np.dot(vector_arr, vector.sentiment))
                    
                    b -= self.learning_rate * vector.sentiment
        
        self.models[target] = (w, b)
    
    
    def test(self, test_set):
        correct = 0
        
        for tweet in test_set.itertuples():
            prediction, _ = self.predict(tweet.embedding)
            
            if prediction == tweet.sentiment:
                correct += 1
        
        
        return correct / test_set.shape[0]



def build_dataset(file_path):
    df = pd.read_csv(file_path)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    
    return df.loc[:, ["tweet_content", "sentiment", "embedding"]]


def replace_labels(df: pd.DataFrame, target):
    df_temp = df.copy()
    df_temp["sentiment"] = df_temp["sentiment"].apply(lambda x: 1 if x == target else -1)
    return df_temp


def training_testing_split(df, percentage, seed=None):
    """splits the dataset into training and testing data"""
    
    if seed:
        df = df.sample(frac=1, random_state=seed)
    else:
        df = df.sample(frac=1)
    
    train_df = df[:round(df.shape[0] - (df.shape[0] * percentage))]
    test_df = df[round(df.shape[0] - (df.shape[0] * percentage)):]
    
    return train_df, test_df


classifier = SVM(1000, 0.001, 1)
dataset = build_dataset("twitterData1000.csv")
training, testing = training_testing_split(dataset, 0.2)

labels = ["Positive", "Negative", "Neutral", "Irrelevant"]

for label in labels:
    training_temp = replace_labels(training, label)
    classifier.train(label, training_temp)


print(classifier.test(testing))

    





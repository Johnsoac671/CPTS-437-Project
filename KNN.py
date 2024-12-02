from processing import Vectorizer
import math
import pandas as pd
import numpy as np
import ast

class KNN:
    def __init__(self, dataset=None, k=5):
        self.dataset = dataset    
        self.k = k
    
    
    def update_dataset(self, new_data):
        self.dataset = new_data
        self.embeddings = np.vstack(self.dataset["embedding"].to_numpy())
        
    
    def predict(self, tweet):
        """returns the predicted sentiment of the given tweet"""
        
        distances = np.sqrt(np.sum((self.embeddings - tweet) ** 2, axis=1))
            
        sentiments = sorted(zip(distances, self.dataset["sentiment"]), key=lambda x: x[0])
        
        nearest = sentiments[:self.k]
        nearest_sentiments = [x[1] for x in nearest]
        
        return self.most_common(nearest_sentiments)

    
    def test(self, test_set: pd.DataFrame):
        """predicts each test tweet, and returns the accuracy"""
        
        correct = 0
        
        for tweet in test_set.itertuples():
            prediction = self.predict(tweet.embedding)
            
            if prediction == tweet.sentiment:
                correct += 1
        
        
        return correct / test_set.shape[0]
    
    
    # def get_distance(self, vector, data):
    #   """returns the cosine similarity between the two vectors"""
        
    #     dot_product = sum(
    #         [x[0] * x[1] for x in zip(vector, data)]
    #         )
        
    #     vec_mag = self.get_magnitude(vector)
    #     data_mag = self.get_magnitude(data)
        
    #     return dot_product / (vec_mag * data_mag)
    
    
    # def get_distance(self, vector, data):
    #     """returns the (Euclidian) distance between the given vectors"""
        
    #     return math.sqrt(
    #         sum(
    #             [(x[0] - x[1]) ** 2 for x in zip(vector, data)]
    #             ))

    def get_magnitude(self, vector):
        """returns the magnitude of the given vector"""
        
        return math.sqrt(
            sum(
                [x*x for x in vector]
                ))
        
    
    def most_common(self, values):
        """returns the most common item in the collection"""
        
        return max(set(values), key=values.count)


def build_dataset(csv="twitterData1000.csv"):
    """grabs the data from the csv file, and converts the embeddings into Python Lists"""
    
    df = pd.read_csv(csv)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df.loc[:, ["tweet_content", "sentiment", "embedding"]]


def training_testing_split(df, percentage, seed=None):
    """splits the dataset into training and testing data"""
    
    if seed:
        df = df.sample(frac=1, random_state=seed)
    else:
        df = df.sample(frac=1)
    
    train_df = df[:round(df.shape[0] - (df.shape[0] * percentage))]
    test_df = df[round(df.shape[0] - (df.shape[0] * percentage)):]
    
    return train_df, test_df


if __name__ == "__main__":

    
    classifier = KNN()
    dataset = build_dataset()
    best = (0, 0.0)
    
    for k in range(1, 100):
        accuracy = 0
        
        epochs = 5
        for epoch in range(epochs):
            train, test = training_testing_split(dataset, 0.2)
            
            classifier.update_dataset(train)
            classifier.k = k

            accuracy += classifier.test(test)
        
        average_accuracy = accuracy / epochs
        
        if average_accuracy > best[1]:
            best = (k, average_accuracy)
        
        print(f"{k}: {round(average_accuracy / epochs, 5)}")

    print(f"Best: {best}")
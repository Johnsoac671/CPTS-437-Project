from processing import Vectorizer
import math
import pandas as pd
import ast

class KNN:
    # test change
    def __init__(self, dataset, k=5):
        self.dataset = dataset
        self.vectorizer = Vectorizer()
        self.k = k
        
    
    def predict(self, data):
        """returns the predicted sentiment of the given tweet"""
        
        vectorized_data = self.vectorizer.vectorize(data)
        
        distances = []
        
        for tweet in self.dataset.itertuples():
            distance = self.get_distance(tweet.embedding, vectorized_data)
            distances.append((distance, tweet.sentiment))
            
        distances.sort(key=lambda x: x[0])
        
        nearest = distances[:self.k]
        nearest_labels = [x[1] for x in nearest]
        
        prediction = self.most_common(nearest_labels)
        return prediction

    
    def test(self, test_set):
        """predicts each test tweet, and returns the accuracy"""
        
        correct = 0
        
        for tweet in test_set.itertuples():
            prediction = self.predict(tweet.tweet_content)
            
            if prediction == tweet.sentiment:
                correct += 1
        
        
        return correct / test_set.shape[0]
    
    
    # def get_distance(self, vector, data):
        """returns the cosine similarity between the two vectors"""
        
    #     dot_product = sum(
    #         [x[0] * x[1] for x in zip(vector, data)]
    #         )
        
    #     vec_mag = self.get_magnitude(vector)
    #     data_mag = self.get_magnitude(data)
        
    #     return dot_product / (vec_mag * data_mag)
    
    
    def get_distance(self, vector, data):
        """returns the (Euclidian) distance between the given vectors"""
        
        return math.sqrt(
            sum(
                [(x[0] - x[1]) ** 2 for x in zip(vector, data)]
                ))

        

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


def training_testing_split(df, percentage, seed=671):
    """splits the dataset into training and testing data"""
    
    df = df.sample(frac = 1, random_state=seed)
    
    train_df = df[:round(df.shape[0] - (df.shape[0] * percentage))]
    test_df = df[round(df.shape[0] - (df.shape[0] * percentage)):]
    
    return train_df, test_df


if __name__ == "__main__":
    train, test = training_testing_split(build_dataset(), 0.2)
    
    bob = KNN(train, 5)
    for x in range(1, 100):
        bob.k = x

        print(f"K = {x}: {bob.test(test)}")
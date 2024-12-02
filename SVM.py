import pandas as pd
import ast
import numpy as np

from processing import dot_product, Vectorizer

class SVM:
    
    def __init__(self, labels, iterations, learning_rate, C):
        self.models = {}
        self.labels = labels
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.C = C
        
    
    def predict(self, tweet: list):
        
        estimates = []
        
        for label in self.labels:
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
    
    
    def test(self, test_set: pd.DataFrame):
        """predicts each test tweet, and returns the accuracy"""
        
        labels = ["Positive", "Negative", "Neutral"]
        precisions = {label:[0, 0] for label in labels} # True Positives : True and False Positives
        recalls = {label:[0, 0] for label in labels} # True Positives : True Positives and False Negatives
        
        correct = 0
        
        for tweet in test_set.itertuples():
            prediction, _ = self.predict(tweet.embedding)
            
            precisions[prediction][1] += 1
            recalls[tweet.sentiment][1] += 1
            
            if prediction == tweet.sentiment:
                correct += 1
                precisions[prediction][0] += 1
                recalls[tweet.sentiment][0] += 1
        
        accuracy = correct / test_set.shape[0]
        precision = {key : value[0] / value[1] for key, value in precisions.items()}
        recall = {key : value[0] / value[1] for key, value in recalls.items()}
        
        return accuracy, precision, recall



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


if __name__ == "__main__":
    labels = ["Positive", "Negative", "Neutral"]

    classifier = SVM(labels, 200, 0.001, 0.1)
    dataset = build_dataset("twitterData1000.csv")

    total_accuracy = 0
    total_precision = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_recall = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    for _ in range(5):
        training, testing = training_testing_split(dataset, 0.2)



        for label in labels:
            training_temp = replace_labels(training, label)
            classifier.train(label, training_temp)


        accuracy, precision, recall = classifier.test(testing)
        
        total_accuracy += accuracy
        
        for label in precision:
            total_precision[label] += precision[label]
        for label in recall:
            total_recall[label] += recall[label]
    
    avg_accuracy = total_accuracy / 5
    avg_precision = {label: round(total_precision[label] / 5, 3) for label in total_precision}
    avg_recall = {label: round(total_recall[label] / 5, 3) for label in total_recall}
    
    print(f"Accuracy: {round(avg_accuracy, 5)}, "
          f"Precision: {avg_precision}, Recall: {avg_recall}")
        
    


    





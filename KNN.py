from processing import Vectorizer
import math
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

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
        
        labels = ["Positive", "Negative", "Neutral"]
        precisions = {label:[0, 0] for label in labels} # True Positives : True and False Positives
        recalls = {label:[0, 0] for label in labels} # True Positives : True Positives and False Negatives
        
        correct = 0
        
        for tweet in test_set.itertuples():
            prediction = self.predict(tweet.embedding)
            
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
    best_k = {"k": 0, "accuracy": 0.0, "precision": {}, "recall": {}}
    KNNPerformanceOverTime = {}

    for k in range(1, 21):
        total_accuracy = 0
        total_precision = {"Positive": 0, "Negative": 0, "Neutral": 0}
        total_recall = {"Positive": 0, "Negative": 0, "Neutral": 0}
        epochs = 5

        for epoch in range(epochs):

            train, test = training_testing_split(dataset, percentage=0.2)

            # test KNN on current k value
            classifier.update_dataset(train)
            classifier.k = k
            accuracy, precision, recall = classifier.test(test)

            # update performance metrics
            total_accuracy += accuracy
            for label in precision:
                total_precision[label] += precision[label]
            for label in recall:
                total_recall[label] += recall[label]

        # calculate average metric for the current k value
        avg_accuracy = total_accuracy / epochs
        avg_precision = {label: round(total_precision[label] / epochs, 3) for label in total_precision}
        avg_recall = {label: round(total_recall[label] / epochs, 3) for label in total_recall}

        KNNPerformanceOverTime[k] = (avg_accuracy * 100)

        # check if current k outperforms previous best k
        if avg_accuracy > best_k["accuracy"]:
            best_k = {"k": k, "accuracy": avg_accuracy, "precision": avg_precision, "recall": avg_recall}


    print(f"Best: k = {best_k['k']}, Accuracy: {round(best_k['accuracy'], 5)}, "
          f"Precision: {best_k['precision']}, Recall: {best_k['recall']}")
    
    plt.plot(KNNPerformanceOverTime.keys(), KNNPerformanceOverTime.values(), label='Accuracy')

    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy %')
    plt.title('KNN Accuracy')
    plt.xticks([int(x) for x in KNNPerformanceOverTime.keys()])
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.5)
    plt.show()
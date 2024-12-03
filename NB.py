import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from math import log
from wordcloud import WordCloud


class NaiveBayes:
    def __init__(self):
        self.class_priors = defaultdict(float)
        self.word_likelihoods = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()

    def train(self, data, labels):
        label_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_samples = len(labels)

        for tweet, label in zip(data, labels):
            label_counts[label] += 1
            for word in tweet.split():
                self.vocabulary.add(word)
                word_counts[label][word] += 1
        self.class_priors = {label: count / total_samples for label, count in label_counts.items()}

        for label in label_counts:
            total_words = sum(word_counts[label].values())
            self.word_likelihoods[label] = {
                word: (word_counts[label][word] + 1) / (total_words + len(self.vocabulary))
                for word in self.vocabulary
            }

    def predict(self, tweet):
        scores = {}
        for label, prior in self.class_priors.items():
            score = log(prior)
            for word in tweet.split():
                if word in self.vocabulary:
                    score += log(self.word_likelihoods[label].get(word, 1 / len(self.vocabulary)))
            scores[label] = score
        return max(scores, key=scores.get)

    def test(self, test_data, test_labels):
        predictions = [self.predict(tweet) for tweet in test_data]
        return accuracy_score(test_labels, predictions)


def build_dataset(file_path):
    df = pd.read_csv(file_path)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    df['sentiment'] = df['sentiment'].replace("Irrelevant", "Neutral")
    return df["tweet_content"].values, df["sentiment"].values


def generate_wordcloud(label, word_likelihoods):
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud = wordcloud.generate_from_frequencies(word_likelihoods[label])
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Sentiment: {label}')
    plt.show()


if __name__ == "__main__":
    tweets, sentiments = build_dataset("twitterData1000.csv")
    X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments, test_size=0.2, random_state=42)
    nb = NaiveBayes()
    nb.train(X_train, y_train)
    accuracy = nb.test(X_test, y_test)
    print(f"Naive Bayes Accuracy: {accuracy}")

    predictions = [nb.predict(tweet) for tweet in X_test]
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    for sentiment in nb.class_priors.keys():
        generate_wordcloud(sentiment, nb.word_likelihoods)

    sentiment_counts = pd.Series(y_train).value_counts()
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution in Training Data')
    plt.show()

    tweet_lengths = [len(tweet.split()) for tweet in tweets]
    plt.figure(figsize=(8, 5))
    plt.hist(tweet_lengths, bins=20, color='orange', edgecolor='black')
    plt.xlabel('Tweet Length (# of Words')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tweet Lengths')
    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import ast


df = pd.read_csv("twitterData1000.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)


X = np.vstack(df['embedding']) 
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


NN = MLPClassifier(max_iter=2000, activation="tanh")


NN.fit(X_train, y_train)


y_pred = NN.predict(X_test)


print(classification_report(y_test, y_pred))

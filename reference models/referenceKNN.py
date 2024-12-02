import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import ast



data = pd.read_csv('twitterData1000.csv')

data['embedding'] = data['embedding'].apply(ast.literal_eval)

X = np.vstack(data['embedding'].values)
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

epochs = 5
best = (0, 0.0)
for k in range(1, 100):
    total = 0
    for _ in range(epochs):
        knn = KNeighborsClassifier(n_neighbors=k)


        knn.fit(X_train, y_train)


        y_pred = knn.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)

        total += accuracy
    
    average = total / epochs
    
    if average > best[1]:
        best = (k, round(average, 5))
        

print(f"Best: {best[0]}, Accuracy: {best[1]}")


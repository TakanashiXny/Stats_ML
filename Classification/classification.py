import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


X = np.load("tfidf_text.npy")
y = np.load("label.npy")
random_state = 438
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)

# model = LogisticRegression(C=2)
# model = LinearSVC(dual=True, C=4)
# model = DecisionTreeClassifier(max_depth=20)
model = GaussianNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

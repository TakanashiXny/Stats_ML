import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


X = np.load("tfidf_text.npy")
y = np.load("label.npy")
random_state = 438
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)


kFold = KFold(n_splits=5, shuffle=True, random_state=438)
param_grid = {'max_depth': [5, 10, 15, 20, 25]}
model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, param_grid, cv=kFold)

# 拟合模型
grid_search.fit(X_train, y_train)

# 得到最佳参数和准确率
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
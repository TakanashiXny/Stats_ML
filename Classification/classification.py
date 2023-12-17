import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='parameters')  # 创建解析器
    parser.add_argument('--model', type=str, default='LogisticRegression', help='input the model')  # 添加参数

    args = parser.parse_args()  # 解析参数
    return args

if __name__ == '__main__':
    args = parse_arguments()
    model_str = args.model
    X = np.load("tfidf_text.npy")
    y = np.load("label.npy")
    random_state = 438
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)


    kFold = KFold(n_splits=5, shuffle=True, random_state=438)
    model = None
    param_grid = None
    if model_str == 'LogisticRegression':
        param_grid = {
            'C': [1, 2, 3],
            'solver': ['lbfgs', 'newton-cholesky']
        }
        model = LogisticRegression()
    elif model_str == 'LinearSVC':
        param_grid = {
            'C': [1, 2, 3],
            'multi_class': ['ovr', 'crammer_singer']
        }
        model = LinearSVC(dual=True, max_iter=5000)
    elif model_str == 'DecisionTree':
        param_grid = {'max_depth': [5, 10, 15, 20, 25]}
        model = DecisionTreeClassifier()

    grid_search = GridSearchCV(model, param_grid, cv=kFold)

    # 拟合模型
    grid_search.fit(X_train, y_train)

    # 得到最佳参数和准确率
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

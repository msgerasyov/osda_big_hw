import classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def print_results(scores):
    for k, v in scores.items():
        print(k + ":", v)

if __name__ == "__main__":

    #load dataset
    print("Loading dataset")
    df = pd.read_csv('heart.csv')
    X, y = classifier.load_dataset(df)

    #Testing FCA algorithm
    fca_classifier = classifier.FCAClassifier()
    fca_classifier.fit(X, y)
    print("Baseline FCA accuracy:", fca_classifier.score(X, y))

    #tune fca classifier parameter
    print("Tuning parameter")
    parameters = {'threshold': np.linspace(0.1, 1, 10)}
    grid_search = GridSearchCV(fca_classifier, parameters)
    grid_search.fit(X, y)
    fca_classifier_tuned = classifier.FCAClassifier(
                            threshold=grid_search.best_params_['threshold'])
    print("Tuned FCA accuracy:", grid_search.best_score_)

    #calculate metrics
    print("Evaluating")
    metrics = ["accuracy", "precision", "recall", "f1"]

    fca_scores = classifier.calculate_metrics(fca_classifier_tuned, metrics, X, y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn_scores = classifier.calculate_metrics(knn, metrics, X, y)

    lr = LogisticRegression(random_state=42)
    lr_scores = classifier.calculate_metrics(lr, metrics, X, y)

    print("Scores")
    print("========================")
    print("FCA:")
    print_results(fca_scores)
    print("========================")
    print("KNN:")
    print_results(knn_scores)
    print("========================")
    print("Logistic regression:")
    print_results(lr_scores)

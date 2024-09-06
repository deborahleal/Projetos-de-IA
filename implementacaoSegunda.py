from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


classifiers = {
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree (C4.5)": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}


results = {name: train_and_evaluate(clf, X_train, X_test, y_train, y_test) 
           for name, clf in classifiers.items()}


results_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Accuracy"])

print(results_df)




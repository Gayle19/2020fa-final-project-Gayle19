import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report


def eval_classifier(clf, X, y):
    """Function to evaluate a classifier for any given scikit-learn model
    :param clf: Classifier to be used Ex: SVM, Random Forest, Naive Bayes
    :param X: Features values
    :param y: label values
    :returns pandas dataframe of classification report"""
    kf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
    for train_index, test_index in kf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        class_report = classification_report(y[test_index], y_pred, output_dict=True)
        df = pd.DataFrame(class_report)
    return df


def encode_onehot(df, col):
    """Function to one hot encode categorical features
    :param df: Pandas Dataframe
    :param col: Column containing categorical values
    :returns pandas dataframe with encoding values"""
    df2 = (
        pd.get_dummies(df[col], prefix="", prefix_sep="")
        .max(level=0, axis=1)
        .add_prefix(col + " - ")
    )
    df3 = pd.concat([df, df2], axis=1)
    df3 = df3.drop([col], axis=1)
    return df3


def impute_data(df, columns):
    """Function to apply imputation to missing values depending on the data type
    :param df: Pandas Dataframe
    :param columns: Columns containing missing values
    :returns pandas dataframe with missing values imputed"""
    for col in columns:
        if df[col].dtype == object:
            df[col].fillna(df[col].value_counts().idxmax(), inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import Lasso
import pickle


def prep_dfs(results_csv, label_csv):
    '''
    This function takes the input features and labels csvs,
    imports them into a pandas dataframe, and cleans them.
    I deleted the code as it contains confidential information
    regarding the data used.
    INPUT:
        - csv of results
        - csv of target labels
    OUTPUT:
        - dataframe containing features
        - series containing labels
        - array of feature names
    '''
    X = None
    labels = None
    features = None
    
    return X, labels, features


def important_lasso_features(X, labels, features):
    '''
    Train Lasso model on entire dataset
    Return list of only features with non-zero betas
    INPUT:
        - array of feature values
        - array of labels array
        - array of feature names
    OUTPUT:
        - list of important features identified by Lasso
    '''
    lasso = Lasso(max_iter=10000).fit(X, labels)
    lasso_features = [feature for feature,
                      coef in zip(features, lasso.coef_) if coef != 0]

    return lasso_features


def rf_important_features(X, labels, features):
    '''
    Train Random Forest model on entire dataset
    Return list of top features
    INPUT:
        - array of feature values
        - array of labels
        - array of feature names
    OUTPUT:
        - array of important features identified by random forest
    '''
    rf = RandomForestClassifier(max_features=X.shape[1])
    rf.fit(X, labels)
    importances = rf.feature_importances_
    rf_features = features[importances > 0]

    return rf_features


def important_rf_features_total(X, labels, features, total_features=500):
    '''
    Run multiple iterations of training Random Forest model on entire dataset
    Return list of top features of predetermined length
    INPUT:
        - array of feature values
        - array of labels
        - array of feature names
        - integer of features to return
    OUTPUT:
        - list of important features identified by multiple iterations of Random Forest
    '''
    rf_features_total = []
    while len(rf_features_total) < total_features:
        rf_features = rf_important_features(X_train, y_train, features)
        for feature in rf_features:
            if feature not in rf_features_total:
                rf_features_total.append(feature)

    return rf_features_total


def get_model():
    '''
    INPUT: NONE
    OUTPUT: gradient boosted model
    '''
    cv = LeaveOneOut()
    params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1],
              'max_depth': [3, 5, 10], 'subsample': [0.5]}
    model = GridSearchCV(
        GradientBoostingClassifier(), params, cv=cv)
    
    return model


def model_metrics(X, labels, features, per_split=1, num_splits=1):
    '''
    Run Lasso and Random Forest for feature selection
    Then run Gradient Boosted model with GridSearchCV
    Get model metrics back
    INPUT:
        - array of feature values
        - array of labels
        - array of feature names
        - integer of # of times a model is fit on a given train_test split
        - integer of # of train_test splits to make
    OUTPUT:
        - List of Lasso/GBT accuracies
        - List of Lasso/GBT precisions
        - List of Lasso/GBT recalls
        - List of RF/GBT accuracies
        - List of RF/GBT precisions
        - List of RF/GBT recalls
    '''
    lasso_accuracies = []
    lasso_precisions = []
    lasso_recalls = []
    rf_accuracies = []
    rf_precisions = []
    rf_recalls = []

    for split in xrange(num_splits):
        print 'split {}'.format(split)

        X_train, X_test, y_train, y_test = train_test_split(X, labels)

        # Lasso Regression for Feature Identification
        lasso_features = important_lasso_features(
            X_train, y_train, features)
        X_train_lasso = X_train[lasso_features]
        X_test_lasso = X_test[lasso_features]

        rf_features_total = important_rf_features_total(
            X_train, y_train, features, total_features=500)
        X_train_rf = X_train[rf_features_total]
        X_test_rf = X_test[rf_features_total]

        for iteration in xrange(per_split):
            print 'iteration {}'.format(iteration)

            # GBT Model

            # Lasso
            lasso_model = get_model()
            lasso_model.fit(X_train_lasso, y_train)
            pred_lasso = lasso_model.predict(X_test_lasso)
            accuracy_lasso = lasso_model.score(X_test_lasso, y_test)
            precision_lasso = precision_score(y_test, pred_lasso)
            recall_lasso = recall_score(y_test, pred_lasso)

            lasso_accuracies.append(accuracy_lasso)
            lasso_precisions.append(precision_lasso)
            lasso_recalls.append(recall_lasso)

            # RF
            rf_model = get_model()
            rf_model.fit(X_train_rf, y_train)
            pred_rf = rf_model.predict(X_test_rf)
            accuracy_rf = rf_model.score(X_test_rf, y_test)
            precision_rf = precision_score(y_test, pred_rf)
            recall_rf = recall_score(y_test, pred_rf)

            rf_accuracies.append(accuracy_rf)
            rf_precisions.append(precision_rf)
            rf_recalls.append(recall_rf)

    return lasso_accuracies, lasso_precisions, lasso_recalls, rf_accuracies, rf_precisions, rf_recalls


def lasso_gbt(X, labels):
    '''
    Run Lasso for feature selection
    Then run Gradient Boosted model with GridSearchCV
    Get pickled model
    INPUT:
        - array of feature values
        - array of labels
    OUTPUT: pickled lasso_gbt model
    '''
    lasso_features = important_lasso_features(
        X_train, y_train, features)
    X_lasso = X[lasso_features]

    lasso_model = get_model()
    lasso_model.fit(X_lasso, labels)
    
    with open('lasso_gbt.pkl', 'wb') as pkl:
        pickle.dump(lasso_model, pkl)


def rf_gbt(X, labels):
    '''
    Run Random Forest for feature selection
    Then run Gradient Boosted model with GridSearchCV
    Get pickled model
    INPUT:
        - array of feature values
        - array of labels
    OUTPUT: pickled rf_gbt model
    '''
    rf_features_total = important_rf_features_total(
        X, labels, features, total_features=500)
    X_rf = X[rf_features_total]

    rf_model = get_model()
    rf_model.fit(X_rf, labels)
    
    with open('rf_gbt.pkl', 'wb') as pkl:
        pickle.dump(rf_model, pkl)


if __name__ == '__main__':

    X, labels, features = prep_dfs(
        'results_csv', 'labels_csv')

    lasso_accuracies, lasso_precisions, lasso_recalls, rf_accuracies, rf_precisions, rf_recalls = comparisons(
        X, labels, features, per_split=3, num_splits=30)

    rf_gbt(X, labels)
    lasso_gbt(X, labels)

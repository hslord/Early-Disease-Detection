from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from collections import Counter


def prep_dfs(results_csv, labels_csv):
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
        rf_features = rf_important_features(X, labels, features)
        for feature in rf_features:
            if feature not in rf_features_total:
                rf_features_total.append(feature)

    return rf_features_total


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


def get_model():
    '''
    INPUT: NONE
    OUTPUT: gradient boosted model
    '''
    model = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=5, subsample=0.5)

    return model


def get_features(X, labels, features, iterations=10):
    '''
    1) Take data and perform train-test-split
    2) Run train data through Lasso and RandomForest feature identifications
    3) Reduce train data to 2 versions of important features (Lasso and RF)
    4) Run datasets through Gradient Boosting, get top 25 features from each
    INPUT:
        - array of feature values
        - array of labels
        - array of feature names
        - integer number of iterations
    OUTPUT:
        - list of lists of top 25 RandomForest features selected using Gradient Boosting
        - list of lists of top 25 Lasso features selected using Gradient Boosting
    '''
    final_rf_features = []
    final_lasso_features = []

    for iteration in xrange(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, labels)

        # Lasso Regression for Feature Identification
        lasso_features = important_lasso_features(X_train, y_train)
        X_train_lasso = X_train[lasso_features]
        X_test_lasso = X_test[lasso_features]

        # Random Forest for Feature Identification
        rf_features = important_rf_features_total(
            X_train, y_train, features, total_features=500)
        X_train_rf = X_train[rf_features]
        X_test_rf = X_test[rf_features]

        # GBT Feature Identification
        # Lasso
        lasso_model = get_model()
        lasso_model.fit(X_train_lasso, y_train)
        lasso_importances = lasso_model.feature_importances_
        zip_lasso_importances = sorted(
            zip(lasso_importances, lasso_features), key=lambda tup: tup[0], reverse=True)
        top25_feats_lasso = [feat for imp, feat in zip_lasso_importances][:25]
        final_lasso_features.append(top25_feats_lasso)
        # RF
        rf_model = get_model()
        rf_model.fit(X_train_rf, y_train)
        rf_importances = rf_model.feature_importances_
        zip_rf_importances = sorted(
            zip(rf_importances, rf_features), key=lambda tup: tup[0], reverse=True)
        top25_feats_rf = [feat for imp, feat in zip_rf_importances][:25]
        final_rf_features.append(top25_feats_rf)

    return final_lasso_features, final_rf_features


def get_feat_dict(feat_lol):
    '''
    INPUT:
        - list of list of important features
    OUTPUT:
        - counter dictionary of number of times
        each feature came up in the iterations
    '''
    feat_dict = Counter(feat_lol[0])
    for lst in feat_lol[1:]:
        for feat in lst:
            feat_dict[feat] += 1
    
    return feat_dict


def lasso_feats_gbt(X, labels, features):
    '''
    Train model using Lasso feature selection
    Reduce X array to features from Lasso
    Run them through Gradient Boosted model
    This model's main use is for feature selection, not prediction
    INPUT:
    - array of feature values
    - array of labels
    - array of feature names
    OUTPUT:
    - pickled model
    '''
    lasso_features = important_lasso_features(X, labels, features)
    X_lasso = X[lasso_features]

    lasso_feats_model = get_model()
    lasso_feats_model.fit(X_lasso, labels)
    
    with open('lasso_feats_gbt.pkl', 'wb') as pkl:
        pickle.dump(lasso_feats_model, pkl)


def rf_feats_gbt(X, labels, features):
    '''
    Train model using Random Forest feature selection
    Reduce X array to features from Random Forest
    Run them through Gradient Boosted model
    This model's main use is for feature selection, not prediction
    INPUT:
    - array of feature values
    - array of labels
    - array of feature names
    OUTPUT:
    - pickled model
    '''
    important_features = important_rf_features_total(
        X, labels, features, total_features=500)
    X_rf = X[important_features]

    rf_feats_model = get_model()
    rf_feats_model.fit(X_rf, labels)
    
    with open('rf_feats_gbt.pkl', 'wb') as pkl:
        pickle.dump(rf_feats_model, pkl)


if __name__ == '__main__':
    X, labels, features = prep_dfs(
        'results.csv', 'labels.csv')

    lasso, rf = get_features(X, labels, features, iterations=50)

    lasso_feat_dict = get_feat_dict(lasso)
    rf_feat_dict = get_feat_dict(rf)

    lasso_feats_gbt(X, labels, features)
    rf_feats_gbt(X, labels, features)

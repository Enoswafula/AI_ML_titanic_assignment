from sklearn.ensemble import RandomForestClassifier

def feature_importance(X, y):

    model = RandomForestClassifier()
    model.fit(X, y)

    importance = model.feature_importances_

    return importance

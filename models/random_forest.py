from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import shap

def train(X, y_reg, y_clf):
    # Regression
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_reg.fit(X, y_reg)
    
    # Classification
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X, y_clf)
    
    # Save
    joblib.dump(rf_reg, 'models/rf_reg.pkl')
    joblib.dump(rf_clf, 'models/rf_clf.pkl')
    joblib.dump(shap.TreeExplainer(rf_reg), 'models/rf_reg_explainer.pkl')
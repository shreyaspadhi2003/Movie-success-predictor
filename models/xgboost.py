from xgboost import XGBRegressor, XGBClassifier
import joblib
import shap

def train(X, y_reg, y_clf):
    # Regression
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_reg.fit(X, y_reg)
    
    # Classification
    xgb_clf = XGBClassifier(n_estimators=100, eval_metric='logloss')
    xgb_clf.fit(X, y_clf)
    
    # Save
    joblib.dump(xgb_reg, 'models/xgb_reg.pkl')
    joblib.dump(xgb_clf, 'models/xgb_clf.pkl')
    joblib.dump(shap.TreeExplainer(xgb_reg), 'models/xgb_reg_explainer.pkl')
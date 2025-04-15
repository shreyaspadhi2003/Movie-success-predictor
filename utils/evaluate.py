from sklearn.metrics import r2_score, accuracy_score
import joblib
import shap
import matplotlib.pyplot as plt

def evaluate_model(model_path, X_test, y_test, model_type="regression"):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # Metrics
    if model_type == "regression":
        score = r2_score(y_test, y_pred)
        print(f"R² Score: {score:.3f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {score:.3f}")
    
    # SHAP with safety checks
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500], check_additivity=False)  # Smaller sample
        
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_test[:500], show=False)
        plt.savefig(f"{model_path}_summary.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP visualization skipped: {str(e)}")
# --- train.py ---

import joblib
from utils.preprocess import load_data, engineer_features, get_targets, create_name_lookups, get_dataset_stats
from models.random_forest import train as train_rf
from models.xgboost import train as train_xgb
from utils.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print(" Loading and preprocessing data...")
    df = load_data()

    stats = get_dataset_stats(df)
    joblib.dump(stats, 'models/dataset_stats.pkl')

    print(" Creating name lookup dictionaries...")
    name_lookups = create_name_lookups(df)
    joblib.dump(name_lookups, 'models/name_lookups.pkl')

    print(" Engineering features...")
    X = engineer_features(df)
    joblib.dump(df.attrs['top_genres'], 'models/top_genres.pkl')
    y_reg, y_clf, valid_indices = get_targets(df)
    X = X.loc[valid_indices]

    print(f"\n Data Summary:")
    print(f"- Samples: {X.shape[0]}")
    print(f"- Features: {X.shape[1]}")
    print(f"- NaN in features: {X.isna().sum().sum()}")
    print(f"- NaN in targets: {y_reg.isna().sum()}")

    print("\n Splitting data...")
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    print("\n Training Random Forest...")
    train_rf(X_train, y_reg_train, y_clf_train)

    print("\n Training XGBoost...")
    train_xgb(X_train, y_reg_train, y_clf_train)

    print("\n Evaluating Models...")
    print("\n Random Forest:")
    evaluate_model('models/rf_reg.pkl', X_test, y_reg_test, "regression")
    evaluate_model('models/rf_clf.pkl', X_test, y_clf_test, "classification")

    print("\n XGBoost:")
    evaluate_model('models/xgb_reg.pkl', X_test, y_reg_test, "regression")
    evaluate_model('models/xgb_clf.pkl', X_test, y_clf_test, "classification")

    print("\n Saving Confusion Matrix and Metrics...")
    clf_model = joblib.load('models/rf_clf.pkl')
    y_pred = clf_model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_clf_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # Classification Metrics
    metrics_text = []
    metrics_text.append("Classification Metrics:\n")
    metrics_text.append(f"Accuracy: {accuracy_score(y_clf_test, y_pred):.3f}")
    metrics_text.append(f"Precision: {precision_score(y_clf_test, y_pred):.3f}")
    metrics_text.append(f"Recall: {recall_score(y_clf_test, y_pred):.3f}")
    metrics_text.append(f"F1 Score: {f1_score(y_clf_test, y_pred):.3f}")
    try:
        auc = roc_auc_score(y_clf_test, y_pred)
        metrics_text.append(f"AUC-ROC: {auc:.3f}")
    except:
        metrics_text.append("AUC-ROC: Not Applicable")
    metrics_text.append("\nConfusion Matrix:")
    metrics_text.append(str(cm))
    metrics_text.append("\n\n")

    # Regression Metrics
    reg_model = joblib.load('models/rf_reg.pkl')
    y_reg_pred = reg_model.predict(X_test)
    metrics_text.append("Regression Metrics:\n")
    metrics_text.append(f"MAE: {mean_absolute_error(y_reg_test, y_reg_pred):.3f}")
    metrics_text.append(f"MSE: {mean_squared_error(y_reg_test, y_reg_pred):.3f}")
    metrics_text.append(f"RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.3f}")
    metrics_text.append(f"R²: {r2_score(y_reg_test, y_reg_pred):.3f}")
    n = X_test.shape[0]
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2_score(y_reg_test, y_reg_pred)) * (n - 1) / (n - p - 1)
    metrics_text.append(f"Adjusted R²: {adj_r2:.3f}")

    with open("models/performance_metrics.txt", "w") as f:
        f.write("\n".join(metrics_text))

    print("All done! Models and metrics saved")

if __name__ == "__main__":
    main()

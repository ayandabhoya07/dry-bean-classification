import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel("Dry_Bean_Dataset.xlsx")
print("Dataset Shape:", df.shape)


# ============================================================
# 2. REMOVE SELECTED FEATURES
# ============================================================

features_to_drop = [
    "Perimeter",
    "EquivDiameter",
    "ConvexArea",
    "Eccentricity",
    "ShapeFactor3"
]

df_clean = df.drop(columns=features_to_drop)

print("\nShape after feature removal:")
print(df_clean.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df_clean.drop(columns=["Class"])

y = df_clean["Class"]


# ============================================================
# 4. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass Encoding:")

for original, encoded in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(f"{original} -> {encoded}")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 6. INITIAL MODEL COMPARISON
# ============================================================

# These models were used during the initial comparison.
# They are kept here for documentation/reproducibility.
#
# XGBoost was selected for further optimization based on
# the observed classification performance.


# models = {

#     "Logistic Regression":
#         LogisticRegression(
#             max_iter=1000
#         ),

#     "Decision Tree":
#         DecisionTreeClassifier(
#             random_state=42
#         ),

#     "Random Forest":
#         RandomForestClassifier(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         ),

#     "SVM":
#         SVC(
#             kernel="linear"
#         ),

#     "Naive Bayes":
#         GaussianNB(),

#     "KNN":
#         KNeighborsClassifier(
#             n_neighbors=5
#         ),

#     "XGBoost":
#         XGBClassifier(
#             random_state=42,
#             eval_metric="mlogloss",
#             n_jobs=-1
#         )
# }


# ============================================================
# 7. MODEL SELECTION
# ============================================================

print("\n")
print("=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print("""
Multiple classification models were initially evaluated
using classification reports and confusion matrices.

Based on the observed classification performance,
XGBoost was selected for further hyperparameter optimization.
""")


# ============================================================
# 8. INITIAL XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)


# ============================================================
# 9. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300,
        400
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        8
    ],

    "min_child_weight": [
        1,
        3,
        5
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ]
}


# ============================================================
# 10. STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 11. RANDOMIZED SEARCH
# ============================================================

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
    return_train_score=False
)


# ============================================================
# 12. START HYPERPARAMETER TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

print("\nParameter combinations tested:", 20)
print("Cross-validation folds:", 5)
print("Total model fits:", 20 * 5)


random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 13. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

for parameter, value in random_search.best_params_.items():

    print(
        f"{parameter}: {value}"
    )


# ============================================================
# 14. BEST CROSS-VALIDATION SCORE
# ============================================================

print("\nBest CV Macro F1 Score:")

print(
    f"{random_search.best_score_:.4f}"
)


# ============================================================
# 15. GET BEST MODEL
# ============================================================

best_xgb = random_search.best_estimator_


# ============================================================
# 16. FINAL TEST SET PREDICTION
# ============================================================

y_pred_encoded = best_xgb.predict(
    X_test
)


# Convert encoded predictions back
# to original bean names

y_pred = label_encoder.inverse_transform(
    y_pred_encoded.astype(int)
)

y_true = label_encoder.inverse_transform(
    y_test
)


# ============================================================
# 17. FINAL CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred
    )
)


# ============================================================
# 18. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)


print("\n")
print("=" * 70)
print("FINAL TUNED XGBOOST PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix - Tuned XGBoost"
)

plt.tight_layout()

plt.show()


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": best_xgb.feature_importances_

})


# Sort from highest to lowest

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# Print feature importance

print("\n")
print("=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 21. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 22. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_excel(
    "XGBoost_Feature_Importance.xlsx",
    index=False
)


# ============================================================
# 23. SAVE TUNING RESULTS
# ============================================================

search_results = pd.DataFrame(
    random_search.cv_results_
)

search_results = search_results.sort_values(
    by="rank_test_score"
)

search_results.to_excel(
    "XGBoost_Tuning_Results.xlsx",
    index=False
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print("XGBoost_Tuning_Results.xlsx")
print("XGBoost_Feature_Importance.xlsx")

# Save the trained model
joblib.dump(best_xgb, "dry_bean_xgboost.pkl")

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model and label encoder saved successfully!")
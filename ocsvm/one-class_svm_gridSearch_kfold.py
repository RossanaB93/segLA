# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, precision_recall_curve, \
    accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix

# ==========================================
# Support functions
# ==========================================

def calculate_la_expansion_index(data):
    return (np.max(data) - np.min(data)) / np.min(data)

def calculate_la_emptying_fraction(data):
    return (np.max(data) - np.min(data)) / np.max(data)

def hyperplane_distance_score(estimator, X):
    distances = np.abs(estimator.decision_function(X))
    return -np.inf if np.isnan(distances).any() else distances.mean()

# ==========================================
# Load and preprocess data
# ==========================================

volumes = pd.read_csv('volumes_all_phases.csv')
volumes.columns.values[0] = "Patient_ID"
features_data = volumes.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()

diameters = pd.read_csv('diameters_ap.csv')
diameters_list = diameters['Diametro_AP_LA_mm'].tolist()

expansion_list = []
emptying_fraction_list = []

for _, row in features_data.iterrows():
    volume_data = row.values
    expansion_list.append(calculate_la_expansion_index(volume_data))
    emptying_fraction_list.append(calculate_la_emptying_fraction(volume_data))

features = pd.DataFrame({
    'expansion': expansion_list,
    'emptying': emptying_fraction_list,
    'diameter': diameters_list
})

feature_array = np.array(features)

# ==========================================
# Model training with Grid Search
# ==========================================

oc_svm = OneClassSVM(kernel='linear', gamma='scale', nu=0.1)

param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'nu': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1, 0.2, 0.3],
    'degree': [2, 3, 4, 5]
}

grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring=hyperplane_distance_score, n_jobs=-1, verbose=2)
grid_search.fit(feature_array)

print("Best parameters found:", grid_search.best_params_)
print("Best score found:", grid_search.best_score_)

best_oc_svm = grid_search.best_estimator_

# ==========================================
# Cross-validation predictions
# ==========================================

patient_ids = volumes['Patient_ID']
group_kf = GroupKFold(n_splits=5)

decision_scores = []
predictions_best = []
current_test = []

for train_index, test_index in group_kf.split(feature_array, groups=patient_ids):
    X_train, X_test = feature_array[train_index], feature_array[test_index]
    current_test.append(list(test_index))
    
    best_oc_svm.fit(X_train)
    predictions_best.append(list(best_oc_svm.predict(X_test)))
    scores = best_oc_svm.decision_function(X_test)
    decision_scores.append(scores)

# ==========================================
# Build prediction DataFrame
# ==========================================

results = []

for test_indices, predictions in zip(current_test, predictions_best):
    fold_ids = patient_ids.iloc[test_indices].values
    fold_labels = ['Atrial Fibrillation' if p == 1 else 'Sinus Rhythm' for p in predictions]
    fold_df = pd.DataFrame({'Patient_ID': fold_ids, 'Predicted_Label': fold_labels})
    results.append(fold_df)

final_results_df = pd.concat(results).sort_values(by='Patient_ID').reset_index(drop=True)

# ==========================================
# Build feature+prediction DataFrame
# ==========================================

data_with_predictions = pd.DataFrame(feature_array, columns=['Expansion', 'Emptying', 'Diameter'])
data_with_predictions['Predicted_Label'] = np.where(
    final_results_df['Predicted_Label'] == 'Atrial Fibrillation',
    'Atrial Fibrillation', 'Sinus Rhythm'
)
data_with_predictions['Patient_Id'] = volumes['Patient_ID']

print(data_with_predictions)
print("Prediction Summary:")
print(data_with_predictions['Predicted_Label'].value_counts())

# ==========================================
# Feature distribution plots
# ==========================================

for feature in ['Expansion', 'Emptying', 'Diameter']:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Predicted_Label', y=feature, data=data_with_predictions)
    plt.title(f'Distribution of {feature} by Predicted Label')
    plt.show()

# ==========================================
# Normalized distribution plots
# ==========================================

data_normalized = data_with_predictions.copy()
scaler = MinMaxScaler()
data_normalized[['Expansion', 'Emptying', 'Diameter']] = scaler.fit_transform(
    data_normalized[['Expansion', 'Emptying', 'Diameter']]
)

data_melted = data_normalized.melt(
    id_vars="Predicted_Label",
    value_vars=['Expansion', 'Emptying', 'Diameter'],
    var_name="Feature", value_name="Value"
)

plt.figure(figsize=(12, 8))
sns.boxplot(x="Predicted_Label", y="Value", hue="Feature", data=data_melted, palette="pastel")
plt.title("Normalized Feature Distribution by Cluster")
plt.xlabel("Predicted Label")
plt.ylabel("Normalized Value (0-1)")
plt.tight_layout()
plt.show()

# ==========================================
# Evaluation with ground truth
# ==========================================

predictions_df = data_with_predictions[['Predicted_Label', 'Patient_Id']].copy()
predictions_df['Predicted_Label'] = predictions_df['Predicted_Label'].map({
    'Atrial Fibrillation': 0,
    'Sinus Rhythm': 1
})

ground_truth_df = pd.read_csv('ground_truth_df.csv')
evaluation_df = predictions_df.merge(ground_truth_df, on="Patient_Id", suffixes=('_pred', '_true'))

y_pred = evaluation_df['Predicted_Label']
y_true = evaluation_df['True_label']

avg_acc = accuracy_score(y_true, y_pred)
avg_ba = balanced_accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)
specificity = recall_score(~y_true.astype(bool), ~y_pred.astype(bool))

print(f"Accuracy: {avg_acc:.4f}")
print(f"Balanced Accuracy: {avg_ba:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# ==========================================
# Distance from hyperplane
# ==========================================

all_decision_scores = np.concatenate(decision_scores)
labels = ['Inlier' if s >= 0 else 'Outlier' for s in all_decision_scores]

scores_df = pd.DataFrame({'Decision_Score': all_decision_scores, 'Label': labels})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='Decision_Score', data=scores_df)
plt.title('Distance from Hyperplane (Inliers vs Outliers)')
plt.xlabel('Label')
plt.ylabel('Distance')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# Optional: Evaluate every model in grid
# ==========================================

for model_idx, model_score in enumerate(grid_search.cv_results_['mean_test_score']):
    params = grid_search.cv_results_['params'][model_idx]
    model = grid_search.best_estimator_

    model.fit(feature_array, ground_truth_df['True_label'].values)
    predictions = model.predict(feature_array)
    decision_scores = model.decision_function(feature_array)

    acc = accuracy_score(ground_truth_df['True_label'], predictions)
    bal_acc = balanced_accuracy_score(ground_truth_df['True_label'], predictions)
    fpr, tpr, _ = roc_curve(ground_truth_df['True_label'], decision_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(ground_truth_df['True_label'], decision_scores)

    print(f"Model params: {params}")
    print(f"Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}")
    print("-" * 50)

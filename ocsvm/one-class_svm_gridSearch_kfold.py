#%% Importazione delle Librerie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import make_scorer
import seaborn as sns

#%% Definizione delle Funzioni di Supporto
# Calcolo dell'indice di espansione atriale sinistro
def calculate_la_expansion_index(data):
    return (np.max(data) - np.min(data)) / np.min(data)

# Calcolo della frazione di svuotamento atriale sinistro
def calculate_la_emptying_fraction(data):
    return (np.max(data) - np.min(data)) / np.max(data)

# Funzione di scoring personalizzata per la distanza dall'iperpiano
def hyperplane_distance_score(estimator, X):
    distances = np.abs(estimator.decision_function(X))
    if np.isnan(distances).any():
        return -np.inf  # o un valore basso come -1e6
    return distances.mean()

#%% Caricamento e Preprocessing dei Dati
# Carica i dati segmentati dei volumi
data = pd.read_csv('volumes_all_phases.csv')
data.columns.values[0] = "Patient_ID"  # Rinominare la prima colonna come 'Patient_ID'
features_data = data.iloc[:, 1:]  # Presumendo che la prima colonna sia un indice o non rilevante
features_data = features_data.apply(pd.to_numeric, errors='coerce')
features_data = features_data.dropna()

# Carica i diametri
diameters = pd.read_csv('diameters_ap.csv')
diameters_list = diameters['Diametro_AP_Atrio_Sinistro_mm'].tolist()

# Calcolo dell'indice di espansione e della frazione di svuotamento per ogni riga
expansion_list = []
emptying_fraction = []

for index, row in features_data.iterrows():
    volume_data = row.values
    expansion_list.append(calculate_la_expansion_index(volume_data))
    emptying_fraction.append(calculate_la_emptying_fraction(volume_data))

# Creazione di un DataFrame per le feature
features = pd.DataFrame({
    'expansion': expansion_list,
    'emptying': emptying_fraction,
    'diameter': diameters_list
})

# Conversione delle feature in array numpy
feature_array = np.array(features)

#%% Configurazione e Addestramento del Modello con Grid Search e Cross-Validation
# Configurazione di One-Class SVM
oc_svm = OneClassSVM(kernel='linear', gamma='scale', nu=0.1)

# Griglia dei parametri per Grid Search
param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'nu': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'gamma': ['scale', 'auto'],
    'coef0' : [0.0, 0.1, 0.2, 0.3],
    'degree' : [2, 3, 4, 5]
}

# Esegui Grid Search con 5-fold cross-validation
grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring=hyperplane_distance_score, n_jobs=-1, verbose=2)
grid_search.fit(feature_array)

# Mostra i migliori parametri e punteggio
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

# Recupera il miglior modello addestrato
best_oc_svm = grid_search.best_estimator_

#%% Calcolo dei Punteggi di Decisione tramite Cross-Validation
# Configurazione di GroupKFold per evitare la sovrapposizione dei pazienti nei fold
patient_ids = data['Patient_ID']  # Supponiamo che tu abbia una colonna 'Patient_ID' che identifica univocamente ogni paziente

# Configurazione di GroupKFold con 5 split
group_kf = GroupKFold(n_splits=5)

# Calcolo dei punteggi di decisione tramite cross-validation
decision_scores = []
predictions_best = []
current_test = []

for train_index, test_index in group_kf.split(feature_array, groups=patient_ids):
    # Divisione in dati di addestramento e test
    X_train, X_test = feature_array[train_index], feature_array[test_index]
    current_test.append(list(test_index))
    
    # Addestra il miglior modello
    best_oc_svm.fit(X_train)
    predictions_best.append(list(best_oc_svm.predict(X_test)))  # Predizioni per il fold corrente
    
    # Ottiene i punteggi di decisione (distanza dall'iperpiano)
    scores = best_oc_svm.decision_function(X_test)
    decision_scores.append(scores)

#%% Elaborazione dei Risultati della Cross-Validation
# Lista per memorizzare i risultati di ogni fold
results = []

# Combina gli ID dei pazienti e le predizioni in un unico DataFrame
for test_indices, predictions in zip(current_test, predictions_best):
    # Estrai gli ID dei pazienti e le etichette predette per il fold corrente
    fold_patient_ids = patient_ids.iloc[test_indices].values  # Ricava gli ID dei pazienti
    fold_predictions = ['Atrial Fibrillation' if pred == 1 else 'Sinus Rhythm' for pred in predictions]  # Converte le predizioni
    
    # Crea un DataFrame per i risultati del fold corrente
    fold_df = pd.DataFrame({
        'Patient_ID': fold_patient_ids,
        'Predicted_Label': fold_predictions
    })
    
    # Aggiungi il DataFrame del fold alla lista dei risultati
    results.append(fold_df)

# Concatenazione dei risultati di tutti i fold in un unico DataFrame
final_results_df = pd.concat(results).reset_index(drop=True)
# Ordina il DataFrame in base a 'Patient_ID'
final_results_df = final_results_df.sort_values(by='Patient_ID').reset_index(drop=True)

# Visualizza i risultati ordinati
print(final_results_df)


#%% Predizione e Conversione delle Etichette in Valori Interpretabili
# Predizioni utilizzando il miglior modello

# Converte le predizioni in etichette interpretabili (0 per Ritmo Sinusale, 1 per Fibrillazione Atriale)
data_with_predictions = pd.DataFrame(feature_array, columns=['Expansion', 'Emptying', 'Diameter'])
data_with_predictions['Predicted_Label'] = np.where(final_results_df['Predicted_Label'] == 'Atrial Fibrillation', 'Atrial Fibrillation', 'Sinus Rhythm')
data_with_predictions['Patient_Id'] = data['Patient_ID'] 
# Mostra il DataFrame finale con le predizioni
print(data_with_predictions)


# Riepilogo delle predizioni
print("Prediction Summary:")
print(data_with_predictions['Predicted_Label'].value_counts())

#%% Visualizzazione delle Distribuzioni delle Feature
# Boxplot di distribuzione per ogni feature in base all'etichetta predetta
plt.figure(figsize=(12, 6))
sns.boxplot(x='Predicted_Label', y='Expansion', data=data_with_predictions)
plt.title('Distribution of Expansion by Predicted Label')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Predicted_Label', y='Emptying', data=data_with_predictions)
plt.title('Distribution of Emptying by Predicted Label')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Predicted_Label', y='Diameter', data=data_with_predictions)
plt.title('Distribution of Diameter by Predicted Label')
plt.show()

#%% Normalizzazione dei Dati e Visualizzazione della Distribuzione Normalizzata
# Crea una copia del DataFrame originale per non modificare i dati originali
data_normalized = data_with_predictions.copy()

# Inizializza lo scaler e applicalo alle colonne da normalizzare
scaler = MinMaxScaler()
data_normalized[['Expansion', 'Emptying', 'Diameter']] = scaler.fit_transform(data_normalized[['Expansion', 'Emptying', 'Diameter']])

# Fonde i dati normalizzati in formato lungo
data_melted = data_normalized.melt(id_vars="Predicted_Label", value_vars=['Expansion', 'Emptying', 'Diameter'], 
                                   var_name="Feature", value_name="Value")

# Imposta la figura per il boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x="Predicted_Label", y="Value", hue="Feature", data=data_melted, palette="pastel", width=0.6)
plt.title("Distribuzione Normalizzata di Expansion, Emptying e Diameter nei Cluster", fontsize=16, weight='bold')
plt.xlabel("Predicted_Label", fontsize=14)
plt.ylabel("Valore Normalizzato (0-1)", fontsize=14)
plt.legend(title="Feature")
#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% Visualizzazione della Distribuzione dei Punteggi di Decisione
# Concatenazione di tutti i punteggi di decisione per visualizzare la distribuzione
# all_scores = np.concatenate(decision_scores)

# # Traccia il grafico dei punteggi di decisione
# plt.figure(figsize=(10, 6))
# plt.hist(all_scores, bins=50, alpha=0.7)
# plt.title("Distribuzione dei Punteggi di Decisione (Distanza dall'Iperpiano)")
# plt.xlabel("Punteggio di Decisione")
# plt.ylabel("Frequenza")
# plt.show()

# #%%
# # Visualize the actual and predicted anomalies
# fig = plt.figure(figsize=(20, 6))

# # One-Class SVM Predictions
# ax2 = fig.add_subplot(111, projection='3d')  # 111 specifica che stiamo aggiungendo un unico subplot
# ax2.set_title('One-Class SVM Predictions')
# ax2.scatter(data_with_predictions['Expansion'], data_with_predictions['Emptying'], data_with_predictions['Diameter'], 
#             c=predictions_best, cmap='rainbow')

# ax2.set_xlabel('Expansion')
# ax2.set_ylabel('Emptying')
# ax2.set_zlabel('Diameter')
# ax2.legend()
# plt.title('3D Decision Boundary of One-Class SVM with Poly Kernel on Atrial Features')
# plt.show()

# # %%
# from sklearn.svm import OneClassSVM
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Assuming 'best_oc_svm' is already fitted on your 3D feature array 'feature_array'
# # Extract feature columns
# X = feature_array  # [Expansion, Emptying, Diameter]

# # Predict to identify inliers and outliers
# y_pred = best_oc_svm.predict(X)

# # Separate inliers (label = 1) and outliers (label = -1)
# inliers = X[y_pred == 1]
# outliers = X[y_pred == -1]

# # Create a 3D grid of points in feature space
# x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
# y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
# z_range = np.linspace(X[:, 2].min() - 1, X[:, 2].max() + 1, 50)
# xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
# grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# # Evaluate the decision function for each point on the grid
# decision_values = best_oc_svm.decision_function(grid)
# decision_values = decision_values.reshape(xx.shape)

# # Set up 3D plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot inliers and outliers
# ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='blue', marker='o', label='Inliers')
# ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='red', marker='s', label='Outliers')

# # Plot the approximate decision boundary (isosurface where decision_function ≈ 0)
# # We’ll plot grid points with decision function values close to zero
# boundary_indices = np.abs(decision_values) < 0.06  # Adjust threshold as needed for a clearer boundary
# ax.scatter(xx[boundary_indices], yy[boundary_indices], zz[boundary_indices], color='orange', alpha=0.1, s=1)

# # Set labels and legend
# ax.set_xlabel('Expansion')
# ax.set_ylabel('Emptying')
# ax.set_zlabel('Diameter')
# ax.legend()
# plt.title('3D Decision Boundary of One-Class SVM with Poly Kernel on Atrial Features')
# ax.view_init(elev=30, azim=60)  # Adjust viewing angle
# plt.show()

# #%%
# from sklearn.svm import OneClassSVM
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Assuming `best_oc_svm` is the fitted OneClassSVM instance and `X` is your 3D feature array
# # Separate the inliers and outliers based on the predictions
# y_pred = best_oc_svm.predict(X)
# inliers = X[y_pred == 1]
# outliers = X[y_pred == -1]

# # Define grid ranges for x, y, and z based on data
# x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
# y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
# z_range = np.linspace(X[:, 2].min() - 1, X[:, 2].max() + 1, 50)
# xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

# # Flatten the grid and stack to create a 3D array of points for evaluation
# grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# # Evaluate the decision function on each point in the 3D grid
# decision_values = best_oc_svm.decision_function(grid)
# decision_values = decision_values.reshape(xx.shape)

# # Set up 3D plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot inliers and outliers
# ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='blue', marker='o', label='Inliers')
# ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='red', marker='s', label='Outliers')

# # Plot the approximate decision boundary by plotting grid points where the decision function is near zero
# # This simulates the boundary surface
# boundary_indices = np.abs(decision_values) < 0.01  # Adjust threshold for clearer boundary if needed
# ax.scatter(xx[boundary_indices], yy[boundary_indices], zz[boundary_indices], color='orange', alpha=0.1, s=1)

# # Set labels and legend
# ax.set_xlabel('Expansion')
# ax.set_ylabel('Emptying')
# ax.set_zlabel('Diameter')
# ax.legend()
# plt.title('3D Decision Boundary of One-Class SVM with Poly Kernel on Atrial Features')
# ax.view_init(elev=30, azim=60)  # Adjust viewing angle
# plt.show()


# %%

predictions_df = data_with_predictions[['Predicted_Label', 'Patient_Id']].copy()

# Conversione delle etichette: "Atrial Fibrillation" -> 0 e "Sinus Rhythm" -> 1
predictions_df['Predicted_Label'] = predictions_df['Predicted_Label'].map({
    'Atrial Fibrillation': 0,
    'Sinus Rhythm': 1
})

# Visualizza il DataFrame risultante
print("Predictions DataFrame:")
print(predictions_df)

#%%

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, balanced_accuracy_score, recall_score
import matplotlib.pyplot as plt
ground_truth_df = pd.read_csv('ground_truth_df.csv')

# Supponiamo che 'ground_truth_df' contenga le etichette vere dei pazienti (colonne: 'Patient_Id' e 'True_Label')
# Assicurati che le etichette di ground truth siano mappate come: 'Atrial Fibrillation' -> 0 e 'Sinus Rhythm' -> 1
# Unisci i DataFrame per avere una corrispondenza tra ID pazienti e predizioni/verità
evaluation_df = predictions_df.merge(ground_truth_df, on="Patient_Id", suffixes=('_pred', '_true'))

# Estrai i vettori delle etichette predette e quelle vere
y_pred = evaluation_df['Predicted_Label']
y_true = evaluation_df['True_label']

# Calcolo delle metriche principali
auroc = roc_auc_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_pred)
avg_acc = accuracy_score(y_true, y_pred)
avg_ba = balanced_accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true , y_pred)
specificity = recall_score(np.logical_not(y_true) , np.logical_not(y_pred))

print("AUROC:", auroc)
print("Accuracy:", avg_acc)
print("Balanced accuracy", avg_ba)
print("Sensitivity", sensitivity)
print("Specificity", specificity)

#%% Visualizzazione della Curva AUROC
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='AUROC (area = %0.2f)' % auroc, color='b')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (AUROC) Curve")
plt.legend(loc="lower right")
#plt.grid(True)
plt.show()

#%% Visualizzazione della Curva Precision-Recall
plt.figure()
plt.plot(recall, precision, label='Precision-Recall Curve', color='g')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
#plt.grid(True)
plt.show()

# %%

all_decision_scores = np.concatenate(decision_scores)

# Determine if each score is an inlier or outlier based on the threshold of the SVM
# Here, distances >= 0 are typically inliers (label 1), distances < 0 are outliers (label -1)
labels = ['Inlier' if score >= 0 else 'Outlier' for score in all_decision_scores]

# Create a DataFrame to hold the scores and their respective labels
scores_df = pd.DataFrame({
    'Decision_Score': all_decision_scores,
    'Label': labels
})

# Plot the box plot for the inliers and outliers based on their distance scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='Decision_Score', data=scores_df)
plt.title('Box Plot of Distances from the Hyperplane for Inliers and Outliers')
plt.xlabel('Class Label')
plt.ylabel('Distance from Hyperplane')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Supponiamo che 'feature_array' contenga i dati e 'ground_truth_df' contenga le etichette vere

# Step 1: Itera su ogni modello trovato da grid_search
for model_idx, model_score in enumerate(grid_search.cv_results_['mean_test_score']):
    # Parametri del modello corrente
    params = grid_search.cv_results_['params'][model_idx]
    
    # Recupera il miglior modello (ovvero il modello che ha dato il miglior score)
    model = grid_search.best_estimator_

    # Fitting del modello sull'intero dataset
    model.fit(feature_array, ground_truth_df['True_label'].values)

    # Predizioni sul dataset completo
    predictions = model.predict(feature_array)
    decision_scores = model.decision_function(feature_array)  # Punteggi di decisione (per AUROC, PR)

    # Calcolare le metriche
    accuracy = accuracy_score(ground_truth_df['True_label'], predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth_df['True_label'], predictions)
    #sensitivity = recall_score(ground_truth_df['True_label'], predictions)
    #specificity = recall_score(1 - ground_truth_df['True_label'], 1 - predictions)
    
    # Calcolare la curva AUROC
    fpr, tpr, _ = roc_curve(ground_truth_df['True_label'], decision_scores)
    roc_auc = auc(fpr, tpr)

    # Calcolare la curva Precision-Recall
    precision, recall, _ = precision_recall_curve(ground_truth_df['True_label'], decision_scores)

    # Visualizzare i risultati
    plt.figure(figsize=(12, 6))

    # Subplot 1: Curva AUROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='b', label=f'AUROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC - {params}')
    plt.legend(loc="lower right")

    # Subplot 2: Curva Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='g', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {params}')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Output delle metriche per il modello corrente
    print(f"\nModel with parameters {params}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUROC: {roc_auc:.4f}")
    print("-" * 50)


# %%

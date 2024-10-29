import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, fbeta_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Caricamento del dataset
df = pd.read_csv("/Users/rossanabarile/Desktop/ICON_BARILE_ROSSANA/Dataset/cuore.csv")

# Rimuovi duplicati, se presenti
df = df.drop_duplicates()

# Separiamo le feature dalla variabile target
X = df.drop('output', axis=1)
y = df['output'].values  # Converti in array NumPy per evitare problemi di indicizzazione

# Standardizziamo le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Numero di fold per la cross-validation
n_splits = 10

# Inizializziamo StratifiedKFold per mantenere la distribuzione delle classi bilanciata
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Definizione degli iperparametri KNN
knn_params = {
    'n_neighbors': 5,      # Numero di vicini considerati
    'metric': 'minkowski',  # Metrica di distanza, minkowski è la default e include l'euclidea
    'p': 2,                # Parametro della metrica Minkowski, 2 corrisponde alla distanza euclidea
    'weights': 'uniform'    # Pesi uniformi per i vicini (alternativa: 'distance' per pesare in base alla distanza)
}

# Liste per salvare i risultati di ogni fold
accuracies = []
f2_scores = []

# Ciclo per eseguire la cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    # Suddividiamo il dataset in training e testing set in base agli indici del fold
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Definizione del modello KNN con iperparametri
    knn = KNeighborsClassifier(**knn_params)
    
    # Addestriamo il modello con i dati di training
    knn.fit(X_train, y_train)
    
    # Effettuiamo le predizioni sui dati di test
    y_pred = knn.predict(X_test)
    
    # Calcoliamo l'accuratezza e l'F2-score per ogni fold
    accuracies.append(accuracy_score(y_test, y_pred))
    f2_scores.append(fbeta_score(y_test, y_pred, beta=2))

    print(f"Fold {fold + 1} completed")

# Calcoliamo media e deviazione standard di accuratezza e F2-score
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_f2_score = np.mean(f2_scores)
std_f2_score = np.std(f2_scores)

# Stampa dei risultati
print(f"Mean Accuracy over {n_splits} folds: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean F2-score over {n_splits} folds: {mean_f2_score:.4f}")
print(f"Standard Deviation of F2-score: {std_f2_score:.4f}")

# Creiamo i grafici per visualizzare la distribuzione delle metriche

# Grafico delle accuratezze
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_splits + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy for each fold')
plt.axhline(mean_accuracy, color='r', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.4f}')
plt.fill_between(range(1, n_splits + 1), mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, color='r', alpha=0.2, label=f'Std: ±{std_accuracy:.4f}')
plt.title('Accuracy for each fold with Mean and Std Deviation')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Grafico degli F2-score
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_splits + 1), f2_scores, marker='o', linestyle='-', color='g', label='F2-score for each fold')
plt.axhline(mean_f2_score, color='r', linestyle='--', label=f'Mean F2-score: {mean_f2_score:.4f}')
plt.fill_between(range(1, n_splits + 1), mean_f2_score - std_f2_score, mean_f2_score + std_f2_score, color='r', alpha=0.2, label=f'Std: ±{std_f2_score:.4f}')
plt.title('F2-score for each fold with Mean and Std Deviation')
plt.xlabel('Fold')
plt.ylabel('F2-score')
plt.legend()
plt.grid(True)
plt.show()

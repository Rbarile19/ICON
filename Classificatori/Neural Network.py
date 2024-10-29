import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, fbeta_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Funzione per costruire e addestrare il modello con iperparametri regolabili
def build_and_train_model(X_train, y_train, X_test, y_test, hidden_layers, hidden_units, activation, learning_rate, epochs, batch_size):
    # Definizione del modello Neural Network
    model = Sequential()
    # Aggiunta del primo livello nascosto con input dimensionale
    model.add(Dense(hidden_units[0], input_dim=X_train.shape[1], activation=activation))
    
    # Aggiunta dei livelli nascosti successivi in base agli iperparametri
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))
    
    # Aggiunta del livello di output
    model.add(Dense(1, activation='sigmoid'))

    # Compilazione del modello con il tasso di apprendimento
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Addestramento del modello
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Effettuiamo le predizioni sui dati di test
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Calcolo delle metriche
    accuracy = accuracy_score(y_test, y_pred)
    f2_score = fbeta_score(y_test, y_pred, beta=2)

    return accuracy, f2_score

# Caricamento del dataset
df = pd.read_csv("/Users/rossanabarile/Desktop/ICON_BARILE_ROSSANA/Dataset/cuore.csv")

# Rimuovi duplicati, se presenti
df = df.drop_duplicates()

# Separazione delle feature dalla variabile target
X = df.drop('output', axis=1)
y = df['output'].values

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Numero di fold per la cross-validation
n_splits = 10

# Inizializzazione StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Liste per salvare i risultati
accuracies = []
f2_scores = []

# Iperparametri per il modello
hidden_layers = 2  # Numero di strati nascosti
hidden_units = [32, 16]  # Numero di neuroni in ciascun strato nascosto
activation = 'relu'  # Funzione di attivazione
learning_rate = 0.001  # Tasso di apprendimento
epochs = 50  # Numero di epoche
batch_size = 10  # Dimensione del batch

# Ciclo di cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    # Suddivisione in training e test set
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Addestriamo il modello per ogni fold
    accuracy, f2_score = build_and_train_model(X_train, y_train, X_test, y_test, hidden_layers, hidden_units, activation, learning_rate, epochs, batch_size)

    # Salviamo i risultati
    accuracies.append(accuracy)
    f2_scores.append(f2_score)

    print(f"Fold {fold + 1} completed")

# Calcolo delle statistiche
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_f2_score = np.mean(f2_scores)
std_f2_score = np.std(f2_scores)

# Stampa dei risultati
print(f"Mean Accuracy over {n_splits} folds: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean F2-score over {n_splits} folds: {mean_f2_score:.4f}")
print(f"Standard Deviation of F2-score: {std_f2_score:.4f}")

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

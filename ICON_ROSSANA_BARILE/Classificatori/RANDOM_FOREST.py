import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Caricamento del dataset
df = pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di BarI/Desktop/ICON_ROSS/DATASET/cuore.csv")

# Rimuovi duplicati, se presenti
df = df.drop_duplicates()

# Separiamo le feature dalla variabile target
X = df.drop('output', axis=1)
y = df['output']

# Standardizziamo le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Numero di run
n_runs = 10

# Inizializziamo liste per salvare i risultati di ogni run
accuracies = []
f2_scores = []

# Ciclo per eseguire il training e testing su diversi run
for i in range(n_runs):
    # Suddivisione del dataset in training e testing set (shuffle=True per mescolare i dati ad ogni iterazione)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=i)
    
    # Definizione del modello Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Addestriamo il modello con i dati di training
    rf.fit(X_train, y_train)
    
    # Effettuiamo le predizioni sui dati di test
    y_pred = rf.predict(X_test)
    
    # Calcoliamo l'accuratezza e l'F2-score per ogni run
    accuracies.append(accuracy_score(y_test, y_pred))
    f2_scores.append(fbeta_score(y_test, y_pred, beta=2))

# Calcoliamo media e deviazione standard di accuratezza e F2-score
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)


# Stampa dei risultati
print(f"Mean Accuracy over {n_runs} runs: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

# Valutazione complessiva sul set di test finale (ultimo run)
print("\nClassification Report (last run):\n", classification_report(y_test, y_pred))

# Creiamo i grafici per visualizzare la distribuzione delle metriche

# Grafico delle accuratezze
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_runs + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy for each run')
plt.axhline(mean_accuracy, color='r', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.4f}')
plt.fill_between(range(1, n_runs + 1), mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, color='r', alpha=0.2, label=f'Std: ±{std_accuracy:.4f}')
plt.title('Accuracy for each run with Mean and Std Deviation')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


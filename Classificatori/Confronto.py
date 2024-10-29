import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Carica il dataset da un file CSV
data = pd.read_csv("/Users/rossanabarile/Desktop/ICON_BARILE_ROSSANA/Dataset")


y = data['output']  
X = data.drop(columns=['output'])  # Rimuovi la colonna output dalle caratteristiche

# Suddividi in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inizializza i modelli da testare
modelli = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Networks': MLPClassifier(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Dizionario per memorizzare le performance dei modelli
modelli_performance = {}

# Addestra i modelli e calcola le metriche
for nome_modello, modello in modelli.items():
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)
    
    # Calcola accuratezza e F2-score
    accuracy = accuracy_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
    
    # Memorizza i risultati
    modelli_performance[nome_modello] = {
        'accuracy': accuracy,
        'f2_score': f2
    }

# Estrazione dei nomi dei modelli, delle accuratezze e degli F2-score
modelli = list(modelli_performance.keys())
mean_accuracies = [performance['accuracy'] for performance in modelli_performance.values()]
f2_scores = [performance['f2_score'] for performance in modelli_performance.values()]

# Converti le accuratezze in percentuali
mean_accuracies = [x * 100 for x in mean_accuracies]

# Creazione del grafico a barre per le accuratezze
plt.figure(figsize=(12, 7))

# Grafico per Accuratezze
plt.subplot(1, 2, 1)
plt.barh(modelli, mean_accuracies, color='skyblue')
plt.xlabel('Mean Accuracy (%)')
plt.title('Confronto tra Modelli di Classificazione - Accuratezza')
plt.xlim(0, 100)

# Aggiungi valori sopra le barre
for i, v in enumerate(mean_accuracies):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', color='black', fontweight='bold')

# Grafico per F2-scores
plt.subplot(1, 2, 2)
plt.barh(modelli, f2_scores, color='lightcoral')
plt.xlabel('F2 Score')
plt.title('Confronto tra Modelli di Classificazione - F2 Score')
plt.xlim(0, 1)

# Aggiungi valori sopra le barre
for i, v in enumerate(f2_scores):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center', color='black', fontweight='bold')

# Mostra entrambi i grafici
plt.tight_layout()
plt.show()

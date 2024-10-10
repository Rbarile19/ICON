import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Caricamento del dataset
df = pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di BarI/Desktop/ICON_ROSS/DATASET/cuore.csv")

# Rimuovi duplicati, se presenti
df = df.drop_duplicates()

# Separazione delle feature dalla variabile target
X = df.drop('output', axis=1)
y = df['output']

# Standardizziamo le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddividiamo il dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Dizionario per raccogliere i risultati
model_results = {}

# 1. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Valutazione
accuracy_knn = accuracy_score(y_test, y_pred_knn)
model_results['KNN'] = {'accuracy': accuracy_knn}

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("-" * 40)

# 2. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Valutazione
accuracy_nb = accuracy_score(y_test, y_pred_nb)
model_results['Naive Bayes'] = {'accuracy': accuracy_nb}

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("-" * 40)

# 3. Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

# Valutazione
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
model_results['Decision Tree'] = {'accuracy': accuracy_dtc}

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dtc))
print("-" * 40)

# 4. Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Valutazione
accuracy_rf = accuracy_score(y_test, y_pred_rf)
model_results['Random Forest'] = {'accuracy': accuracy_rf}

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("-" * 40)

# Report complessivo
print("\n=== Report Complessivo ===")
for model_name, metrics in model_results.items():
    print(f"Modello: {model_name}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 40)

# Creazione di un grafico comparativo delle accuratezze e degli F2 score
modelli = list(model_results.keys())
accuracies = [metrics['accuracy'] * 100 for metrics in model_results.values()]  # Percentuali


# Grafico delle accuratezze
plt.figure(figsize=(10, 5))
plt.barh(modelli, accuracies, color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Confronto tra Modelli di Classificazione - Accuracy')
plt.xlim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', color='black', fontweight='bold')
plt.show()


# Visualizzazione dei dati con seaborn
fig, axes = plt.subplots(2, 3, sharex=True, figsize=(10, 5))

sns.countplot(data=df, x='cp', ax=axes[0, 0])
sns.countplot(data=df, x='fbs', ax=axes[0, 1])
sns.countplot(data=df, x='restecg', ax=axes[0, 2])
sns.countplot(data=df, x='exng', ax=axes[1, 0])
sns.countplot(data=df, x='slp', ax=axes[1, 1])
sns.countplot(data=df, x='caa', ax=axes[1, 2])

plt.tight_layout()
plt.show()
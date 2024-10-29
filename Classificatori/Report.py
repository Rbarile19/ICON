import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, fbeta_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb

# Caricamento del dataset
df = pd.read_csv("/Users/rossanabarile/Desktop/ICON_BARILE_ROSSANA/Dataset/cuore.csv")  # Assicurati che il file sia corretto

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

# Funzione per allenare e valutare i modelli
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    model_results[model_name] = {'accuracy': accuracy, 'f2': f2}

    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

# 1. K-Nearest Neighbors (KNN)
train_and_evaluate_model(KNeighborsClassifier(), 'KNN')

# 2. Naive Bayes
train_and_evaluate_model(GaussianNB(), 'Naive Bayes')

# 3. Decision Tree Classifier
train_and_evaluate_model(DecisionTreeClassifier(random_state=42), 'Decision Tree')

# 4. Random Forest Classifier
train_and_evaluate_model(RandomForestClassifier(random_state=42), 'Random Forest')

# 5. Support Vector Machine
train_and_evaluate_model(SVC(kernel='linear', random_state=42), 'Support Vector Machine')

# 6. Neural Networks
nn_model = Sequential()
nn_model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
f2_nn = fbeta_score(y_test, y_pred_nn, beta=2)
model_results['Neural Networks'] = {'accuracy': accuracy_nn, 'f2': f2_nn}

print("Neural Networks Classification Report:")
print(classification_report(y_test, y_pred_nn))
print("-" * 40)

# 7. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f2_xgb = fbeta_score(y_test, y_pred_xgb, beta=2)
model_results['XGBoost'] = {'accuracy': accuracy_xgb, 'f2': f2_xgb}

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("-" * 40)

# Report complessivo
print("\n=== Report Complessivo ===")
for model_name, metrics in model_results.items():
    print(f"Modello: {model_name}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F2 Score: {metrics['f2']:.4f}")
    print("-" * 40)

# Creazione di un grafico comparativo delle accuratezze e degli F2 score
modelli = list(model_results.keys())
accuracies = [metrics['accuracy'] * 100 for metrics in model_results.values()]  # Percentuali
f2_scores = [metrics['f2'] for metrics in model_results.values()]

# Grafico delle accuratezze
plt.figure(figsize=(10, 5))
plt.barh(modelli, accuracies, color='skyblue', label='Accuracy')
plt.xlabel('Score (%)')
plt.title('Confronto tra Modelli di Classificazione - Accuracy')
plt.xlim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', color='black', fontweight='bold')
plt.legend()
plt.show()

# Grafico degli F2 score
plt.figure(figsize=(10, 5))
plt.barh(modelli, f2_scores, color='salmon', label='F2 Score')
plt.xlabel('F2 Score')
plt.title('Confronto tra Modelli di Classificazione - F2 Score')
for i, v in enumerate(f2_scores):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center', color='black', fontweight='bold')
plt.legend()
plt.show()

# Visualizzazione dei dati con seaborn
fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 8))
sns.countplot(data=df, x='cp', ax=axes[0, 0])
sns.countplot(data=df, x='fbs', ax=axes[0, 1])
sns.countplot(data=df, x='restecg', ax=axes[0, 2])
sns.countplot(data=df, x='exng', ax=axes[1, 0])
sns.countplot(data=df, x='slp', ax=axes[1, 1])
sns.countplot(data=df, x='caa', ax=axes[1, 2])

plt.tight_layout()
plt.show()

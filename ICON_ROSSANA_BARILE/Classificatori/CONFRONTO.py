import matplotlib.pyplot as plt

# Nomi dei modelli
modelli = ['KNN', 'Naive Bayes', 'Decision Tree', 'Random Forest']

# Accuratezze medie dei modelli
mean_accuracies = [0.8774, 0.8613, 0.8161, 0.8645]  # Accuratezze in formato decimale (0-1)

# Converti le accuratezze in percentuali
mean_accuracies = [x * 100 for x in mean_accuracies]

# Creazione del grafico a barre
plt.figure(figsize=(10, 6))
plt.barh(modelli, mean_accuracies, color='skyblue')
plt.xlabel('Mean Accuracy (%)')
plt.title('Confronto tra Modelli di Classificazione')
plt.xlim(0, 100)

# Aggiungi valori sopra le barre
for i, v in enumerate(mean_accuracies):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', color='black', fontweight='bold')

# Visualizza il grafico
plt.show()

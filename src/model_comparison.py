import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Naive Bayes",
    "LSTM",
    "Improved SVM"
]

accuracy = [
    48.74,
    40.74,
    37.39,
    61.18
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, accuracy)

plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Model Performance Comparison", fontsize=14)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.2f}%",
        ha='center',
        fontsize=11
    )

plt.ylim(0, 70)
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()
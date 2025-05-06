import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import random
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 🆕 Import for confusion matrix
import matplotlib.pyplot as plt  # 🆕 Needed to show the plot
from tensorflow.python.keras.callbacks import TensorBoard

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the dataset
with open("1.json", "r") as file:
    data = json.load(file)

# Preprocessing
words = []
labels = []
documents = []
ignore_words = ["?", "!", ".", ","]

for entry in data:
    question = entry["question"]
    tag = entry["tags"][0]
    
    word_list = nltk.word_tokenize(question)
    words.extend(word_list)
    documents.append((word_list, tag))
    
    if tag not in labels:
        labels.append(tag)

# Lemmatize, remove duplicates, and sort words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
labels = sorted(labels)

# Ensure input feature size is exactly 389
required_input_size = 389
while len(words) < required_input_size:
    words.append("placeholder_word")  # Add dummy words if fewer
words = words[:required_input_size]  # Trim excess words

# Create training data
training = []
output_empty = [0] * len(labels)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model with updated input size (389)
model = Sequential([
    Dense(128, input_shape=(389,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
# Save model and data
model.save("chatbot_model.hdf5")

with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Training complete. Model saved.")

# Predict on training data
y_pred = model.predict(train_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(train_y, axis=1)

# After predicting on the training data
y_pred = model.predict(train_x)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels
y_true = np.argmax(train_y, axis=1)  # Get the true class labels

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)  # Use y_pred_classes here
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Calculate and print the True Positive, False Positive, False Negative, and True Negative
for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    
    print(f"Class '{label}':")
    print(f"  TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    




import numpy as np
import matplotlib.pyplot as plt

# Assuming you have already computed the confusion matrix (cm) and labels are defined
# Initialize cumulative counts for TP, FP, FN, and TN
total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

# Iterate over all classes (assuming cm is the confusion matrix from your previous code)
for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    total_tp += TP
    total_fp += FP
    total_fn += FN
    total_tn += TN

# Now create a 2x2 plot with the total TP, FP, FN, TN
fig, ax = plt.subplots(figsize=(6, 6))

# Data to plot
data = np.array([[total_tp, total_fp], [total_fn, total_tn]])

# Create a heatmap plot
cax = ax.matshow(data, cmap="Blues")

# Add color bar for the heatmap
fig.colorbar(cax)

# Annotate the plot with the values
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{data[i, j]}', ha='center', va='center', color='black', fontsize=12)

# Set labels and title
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
ax.set_yticklabels(['Actual Positive', 'Actual Negative'])
ax.set_title("Confusion Matrix Summary (All Classes)")

plt.tight_layout()
plt.show()

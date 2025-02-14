import json
import os
import numpy as np
import nltk
from nltk import LancasterStemmer, WordNetLemmatizer
import random
import pickle
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd

# Initialize the stemmer and lemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Path to the dataset
path = "dataset/medical-question-answer-data"

# Function to load the JSON files
def load_doc(jsonFile):
    with open(jsonFile) as file:
        Json_data = json.loads(file.read())
    return Json_data

# Load the JSON files into Python dictionaries
file1 = load_doc(os.path.abspath(os.path.join(path, "ehealthforumQAs.json")))
file2 = load_doc(os.path.abspath(os.path.join(path, "healthtapQAs.json")))
file3 = load_doc(os.path.abspath(os.path.join(path, "icliniqQAs.json")))
file4 = load_doc(os.path.abspath(os.path.join(path, "questionDoctorQAs.json")))
file5 = load_doc(os.path.abspath(os.path.join(path, "webmdQAs.json")))
file6 = load_doc(os.path.abspath(os.path.join(path, "medical_intent.json")))

# Select the files to be used for training and concatenate them
all_Files = [file1, file3, file4, file6]

words = []
labels = []
documents = []
ignore_words = ['?', '!']

# Process each file and extract questions and their respective tags
for data in all_Files:
    for intent in data:
        if len(intent['tags']) == 0:
            tag = "unspecified"
        else:     
            tag = intent['tags'][0]  # Use the first tag as it is the most relevant
            question = intent["question"]
            # Paraphrased questions for better variety
            paraphrases = [
                question,
                f"What is the issue with {question}?",
                f"Could you explain {question} to me?",
                f"Can you tell me about {question}?",
            ]
            
            # Add paraphrased questions
            for q in paraphrases:
                wrds = nltk.word_tokenize(q)
                words.extend(wrds)
                documents.append((wrds, tag))
                
            if tag not in labels:
                labels.append(tag)

# Lemmatize words and remove unwanted characters
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

print(len(documents), "documents")
print(len(labels), "labels", labels)
print(len(words), "unique lemmatized words", words)

# Save the processed words and labels to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

# Prepare training data
training = []
out_empty = [0 for _ in range(len(labels))]

# Convert each document to a bag of words and its corresponding output label
for doc in documents:
    bag = []
    
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create the bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Create the output row with the correct label position
    output_row = out_empty[:]
    output_row[labels.index(doc[1])] = 1

    # Ensure that bag and output_row are of consistent length
    if len(bag) == len(words) and len(output_row) == len(labels):
        training.append([bag, output_row])

# Shuffle the training data to ensure randomness
random.shuffle(training)

# Convert the training data to a NumPy array
training = np.array(training, dtype=object)

# Split into input features and labels
train_x = list(training[:, 0])  # Features (bags)
train_y = list(training[:, 1])  # Labels (one-hot encoded)

print("Training data created")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=10, verbose=1)

# Save the trained model
model.save('chatbot_model.hdf5')

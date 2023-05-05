import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
# Define the data as a list of dictionaries
data = [
    {"question": "hi im a prospective student can you tell me about the admission requirements", 
     "response": "sure for undergraduate programs we require a high school diploma and sat or act scores for graduate programs we require a bachelors degree and gre scores is there anything else youd like to know"},
    {"question": "how much is the tuition for the business program", 
     "response": "the tuition for the business program is x per year you may also be eligible for scholarships or financial aid"},
    {"question": "thats expensive are there any payment plans available", 
     "response": "yes we offer payment plans that allow you to spread out the cost over several months would you like more information on this"},
    {"question": "yes please", 
     "response": "great i can send you a link to our payment plan options whats your email address"},
    {"question": "myemailexamplecom", 
     "response": "okay ill send that over shortly do you have any other questions"}
]
# Open a CSV file for writing and write the data to it
with open("data.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["question", "response"])
    writer.writeheader()
    writer.writerows(data)
# Load the preprocessed data from the CSV file into a Pandas DataFrame
data = pd.read_csv('data.csv')
#Split the data into training and testing sets.
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#Prepare the data for training by tokenizing the text and converting it to numerical data.
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['question'])
train_sequences = tokenizer.texts_to_sequences(train_data['question'])
train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=50)
test_sequences = tokenizer.texts_to_sequences(test_data['question'])
test_padded = pad_sequences(test_sequences, padding='post', truncating='post', maxlen=50)
train_labels = pd.get_dummies(train_data['response']).values
test_labels = pd.get_dummies(test_data['response']).values
#Define a suitable machine learning model for training the chatbot.
#second

print(train_labels.shape)
print(test_labels.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(train_labels.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#train the bot on training data
model.fit(train_padded, train_labels, epochs=10, batch_size=32, validation_data=(test_padded, test_labels))
#save the model
model.save('trained_model.h5')


import os
import pandas as pd
import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the preprocessed data from the CSV file into a Pandas DataFrame.
data = pd.read_csv('preprocessed_data.csv')

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


# Define start and restart sequences
start_sequence = "\nAI:"
restart_sequence = "\nHuman:"


def openai_create(prompt):
    model = tf.keras.models.load_model('trained_model.h5')
    input_sequence = tokenizer.texts_to_sequences([prompt])
    padded_sequence = pad_sequences(input_sequence, padding='post', truncating='post', maxlen=50)
    output = model.predict(padded_sequence)
    return tokenizer.sequences_to_texts([output.argmax(axis=1)])[0]

# # Define function to generate responses using your trained model
# def generate_response(model, input_text):
#     input_seq = tokenizer.texts_to_sequences([input_text])
#     input_seq = pad_sequences(input_seq, maxlen=max_seq_len, padding='post')
#     predicted_label = np.argmax(model.predict(input_seq), axis=-1)
#     response = label_tokenizer.index_word[predicted_label[0]]
#     return response

# Define function to handle chat messages
def chat(message, chat_history):
    # Combine the chat history and new message
    chat_history.append(message)
    prompt = start_sequence + restart_sequence.join(chat_history) + restart_sequence
    
#function to generae response  
# def chatgpt_clone(input, history):
#     history = history or []
#     s = list(sum(history,()))
#     s.append(input)
#     inp = ' '.join(s)
#     output = generate_response(model, inp)
#     history.append((input, output))
#     return history, history


# def chatgpt_clone(input, history):
#     history = history or []
#     s = list(sum(history,()))
#     s.append(input)
#     inp = ' '.join(s)
#     output = generate_response(model, inp)
#     history.append((input, output))
#     return history, history
   

# Set up Gradio user interface
block = gr.Interface(
    fn=chat,
    inputs=[
        gr.inputs.Textbox(placeholder="Enter your message here."),
        gr.inputs.Hidden(initial=[]),
    ],
    outputs=[
        gr.outputs.Textbox(placeholder="AI's response here."),
        gr.outputs.Hidden(),
    ],
    title="University Chatbot",
    description="A chatbot to answer questions about the university.",
)

# Launch the Gradio interface
block.launch()

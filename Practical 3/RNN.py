import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# 2. Tokenize Text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# 3. Create N-gram Sequences
input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 4. Build & Train Simple RNN Model
model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len - 1),
    SimpleRNN(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# 5. User Input Prediction
def predict_next_words(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted_idx = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_idx:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print("\n--- RNN Model Ready ---")
user_seed = input("Enter starting text/seed: ")
num_words = int(input("Enter number of words to predict: "))
result = predict_next_words(user_seed, num_words)
print("\nGenerated Output:")
print(result)

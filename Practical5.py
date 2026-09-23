import numpy as np
print("---Simple Attention mechanism Explanation---")
encoder_output = np.array([
    [0.1,0.2],
    [0.5,0.3],
    [0.2,0.8],
    [0.9,0.1]
])
print(f"\nEncoder Outputs (4 words, 2 Features each):\n{encoder_output}")

decoder_hidden_state = np.array([0.7,0.4])
print(f"\nDecoder Hidden state (what we are looking for): {decoder_hidden_state}")

alignment_scores = np.dot(encoder_output, decoder_hidden_state)
print(f"\nAlignment Scores (similarity of each encoder output to decoder state):{alignment_scores}")

def softmax(x):
  e_x = np.exp(x-np.max(x))
  return e_x / e_x.sum(axis=0)

attention_weights = softmax(alignment_scores)
print(f"\nAttention Weights (how much to 'pay attention' to each word): {attention_weights}")
print(f"(Notice that the word 'student' (index 3) got higher weight, meaning its more relevant to 'learning')")

context_vector = np.sum(encoder_output * attention_weights[:, np.newaxis], axis=0)
print(f"\nContext Vector (weighted sum of encoder output, 'paying attention'): {context_vector}")

print("---How it works---")

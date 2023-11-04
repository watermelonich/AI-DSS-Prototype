from autocorrect import Speller
import string
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizer import (correct_spelling, preprocess_text, tokenize_text,
                       text_to_indices, save_tokens, load_tokens, save_vocab, load_vocab)
import matplotlib.pyplot as plt

# Load input data from file
with open('input_data.txt', 'r') as file:
    lines = file.readlines()

phrases = [line.strip().split(',')[0] for line in lines]
corrected_phrases = [correct_spelling(phrase) for phrase in phrases]
tokenized_phrases = [tokenize_text(phrase) for phrase in corrected_phrases]
preprocessed_phrases = [preprocess_text(phrase) for phrase in corrected_phrases]
tokens = tokenize_text(' '.join(preprocessed_phrases))
vocab = {token: index for token, index in zip(tokens, [i/10 for i in range(1, len(tokens)+1)])}
indices = text_to_indices(tokens, vocab)

# Save tokenized phrases and vocabulary
save_tokens(phrases, 'tokens.txt')
save_vocab(vocab, 'vocab.txt', tokenized_phrases)

# Load tokens and vocabulary
loaded_tokens = load_tokens('tokens.txt')
loaded_vocab = load_vocab('vocab.txt')

# For demonstration, let's generate some random y_test and y_pred
np.random.seed(0)
y_test = np.random.randint(0, 2, 100)  # Assuming binary classification
y_pred = np.random.randint(0, 2, 100)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Plot actual vs. predicted values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

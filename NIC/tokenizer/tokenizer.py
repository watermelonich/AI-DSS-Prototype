from autocorrect import Speller
import string
import numpy as np

spell = Speller()

# Define a mapping from labels to numerical values
label_to_numeric = {"positive": 0, "negative": 1, "neutral": 2}

# python3 tokenizer.py

def encode_sentiments(sentiments):
    label_to_numeric = {"positive": 0, "negative": 1, "neutral": 2}
    encoded_sentiments = []

    for sentiment in sentiments:
        encoded = [0, 0, 0]  # Initialize with all zeros
        encoded[label_to_numeric[sentiment]] = 1  # Set the corresponding index to 1
        encoded_sentiments.append(encoded)

    return np.array(encoded_sentiments)

# Function to correct spelling in a given text
def correct_spelling(text):
    words = text.split()
    corrected_words = []

    for word in words:
        corrected_word = spell(word)
        corrected_words.append(corrected_word)

    return ' '.join(corrected_words)

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Lowercase the text
    text = text.lower()

    return text

# Function to tokenize text
def tokenize_text(text):
    tokens = text.split()
    return tokens

# Function to build vocabulary
def build_vocab(tokens):
    vocab = {}
    index = 0.1

    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 0.1

    return vocab

# Function to convert text to indices
def text_to_indices(text, vocab):
    indices = [vocab[token] for token in text if token in vocab]
    return indices

# Function to save tokens to a file
def save_tokens(tokens, file_path):
    with open(file_path, 'a') as file:
        file.write('\n\n'.join(tokens))

# Function to load tokens from a file
def load_tokens(file_path):
    with open(file_path, 'r') as file:
        tokens = file.read().splitlines()
    return tokens

# Saving vocabulary
def save_vocab(vocab, file_path, tokenized_phrases):
    with open(file_path, 'a') as file:
        for word, index in vocab.items():
            file.write(f'{word}: {index:.2f}\n')


# Function to load vocabulary from a file
def load_vocab(file_path):
    vocab = {}
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                word, index = parts
                vocab[word] = float(index)
    return vocab

# Read input from file
with open('input_data.txt', 'r') as file:
    lines = file.readlines()

    print(lines)

# Extract phrases and sentiments
phrases = [line.strip().split(',')[0] for line in lines]
sentiments = [line.strip().split(',')[1] for line in lines]
print(sentiments)

def compute_qkv(embeddings):
    Q = embeddings
    K = embeddings
    V = embeddings
    return Q, K, V

def self_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose()) / np.sqrt(d_k)  # Transpose K here
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    output = np.matmul(attention_weights, V)
    return output

import numpy as np

class FeedForwardNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.zeros((1, output_dim))
        self.a1 = None

    # Add a method for training the network
    def train(self, x, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(len(x)):
                output = self.forward(x[i])
                target = y[i]
                loss = self.backward(x[i], output, target, learning_rate)
                total_loss += loss
            average_loss = total_loss / len(x)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    def forward(self, x):
        z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = np.tanh(z1)
        z2 = np.dot(self.a1, self.weights2) + self.bias2
        output = z2
        return output

    def backward(self, x, output, y, learning_rate):
        loss = np.mean((output - y) ** 2)
        dz2 = 2 * (output - y) / x.shape[0]
        dw2 = np.dot(np.transpose(self.a1), dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, np.transpose(self.weights2))
        dz1 = da1 * (1 - np.tanh(np.dot(x, self.weights1) + self.bias1)**2)
        dw1 = np.dot(np.transpose(x), dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2

        return loss
    
def compute_loss(output, target):
    # Ensure that the shapes of output and target are compatible
    print(output.shape)
    print(target.shape)
    assert output.shape == target.shape, "Output and target shapes must match"
    
    # Apply softmax to the output to get class probabilities
    probs = softmax(output)
    
    # Compute the cross-entropy loss
    loss = -np.sum(target * np.log(probs + 1e-10)) / len(output)
    
    return loss

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Apply spelling correction to each phrase
corrected_phrases = [correct_spelling(phrase) for phrase in phrases]

# Tokenize the corrected phrases
tokenized_phrases = [tokenize_text(phrase) for phrase in corrected_phrases]

# Preprocess the corrected phrases
preprocessed_phrases = [preprocess_text(phrase) for phrase in corrected_phrases]

# Extract tokens from preprocessed phrases
tokens = tokenize_text(' '.join(preprocessed_phrases))

# Build vocabulary with word values
vocab = {token: index for token, index in zip(tokens, [i/10 for i in range(1, len(tokens)+1)])}

num_epochs = 100
learning_rate = 0.01

input_dim = 100
hidden_dim = 64
output_dim = 3  # (positive, negative, neutral)

target = encode_sentiments(sentiments)

network = FeedForwardNetwork(input_dim, hidden_dim, output_dim)

num_epochs = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    for token, sentiment in zip(tokens, sentiments):
        if token in vocab:
            numeric_input = np.zeros((1, input_dim))
            numeric_input[0, :] = vocab[token]
            output = network.forward(numeric_input)
            target_sentiment = label_to_numeric[sentiment]  # Convert sentiment label to numeric
            target_encoded = np.zeros((1, output_dim))  # Initialize target as all zeros
            target_encoded[0, target_sentiment] = 1  # Set the corresponding index to 1
            loss = network.backward(numeric_input, output, target_encoded, learning_rate)
            print(f'Epoch [{epoch+1}/{num_epochs}], Token [{token}], Loss: {loss:.4f}')

# Now, you can perform the forward pass
output = network.forward(numeric_input)

# Convert tokens to indices
indices = text_to_indices(tokens, vocab)

# Convert tokens to embeddings (random for demonstration)
embeddings = np.random.rand(len(tokens), 4)  # Assuming 4-dimensional embeddings

# Compute Q, K, and V matrices
Q, K, V = compute_qkv(embeddings)

# Apply self-attention
output = self_attention(Q, K, V)

# Save phrases with sentiments to a file
def save_phrases_with_sentiments(phrases, sentiments, file_path):
    with open(file_path, 'w') as file:
        for phrase, sentiment in zip(phrases, sentiments):
            file.write(f'{phrase},{sentiment}\n')

# Load phrases with sentiments from a file
def load_phrases_with_sentiments(file_path):
    phrases = []
    sentiments = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                phrase, sentiment = parts
                phrases.append(phrase)
                sentiments.append(sentiment)
    return phrases, sentiments

# Save the tokenized phrases with two empty lines between them
save_tokens(phrases, 'tokens.txt')

# Save vocabulary with two double new lines after each phrase
save_vocab(vocab, 'vocab.txt', tokenized_phrases)

# Build the input data in a suitable format
numeric_inputs = []
for token, sentiment in zip(tokens, sentiments):
    if token in vocab:
        numeric_input = np.zeros((1, input_dim))
        numeric_input[0, :] = vocab[token]
        numeric_inputs.append(numeric_input)

# Save Q, K, and V matrices
with open('qkv.txt', 'a') as file:
    file.write(f'Q Matrix:\n{Q}\n\n')
    file.write(f'K Matrix:\n{K}\n\n')
    file.write(f'V Matrix:\n{V}\n\n')
    file.write(f'Output Matrix:\n{output}\n\n')

# Loading tokens
loaded_tokens = load_tokens('tokens.txt')

# Loading vocabulary
loaded_vocab = load_vocab('vocab.txt')

# Save phrases with sentiments
save_phrases_with_sentiments(phrases, sentiments, 'phrases_with_sentiments.txt')

# Load phrases with sentiments
loaded_phrases, loaded_sentiments = load_phrases_with_sentiments('phrases_with_sentiments.txt')

print("Loaded Phrases:", loaded_phrases)
print("Loaded Sentiments:", loaded_sentiments)

print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Indices:", indices)
print("Loaded Tokens:", loaded_tokens)
print("Loaded Vocabulary:", loaded_vocab)
network.train(numeric_inputs, target, num_epochs, learning_rate)

from autocorrect import Speller
import string

spell = Speller()

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
    with open(file_path, 'w') as file:
        file.write('\n\n'.join(tokens))

# Function to load tokens from a file
def load_tokens(file_path):
    with open(file_path, 'r') as file:
        tokens = file.read().splitlines()
    return tokens

# Saving vocabulary
def save_vocab(vocab, file_path, tokenized_phrases):
    with open(file_path, 'w') as file:
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
with open('/../src/sentimanalysis/testdata.txt', 'r') as file:
    lines = file.readlines()

# Extract phrases and sentiments
phrases = [line.strip().split(',')[0] for line in lines]

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

# Convert tokens to indices
indices = text_to_indices(tokens, vocab)

# Save the tokenized phrases with two empty lines between them
save_tokens(phrases, 'tokens.txt')

# Save vocabulary with two double new lines after each phrase
save_vocab(vocab, 'vocab.txt', tokenized_phrases)

# Loading tokens
loaded_tokens = load_tokens('tokens.txt')

# Loading vocabulary
loaded_vocab = load_vocab('vocab.txt')

print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Indices:", indices)
print("Loaded Tokens:", loaded_tokens)
print("Loaded Vocabulary:", loaded_vocab)

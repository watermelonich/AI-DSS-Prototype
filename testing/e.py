import string
from autocorrect import Speller

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Lowercase the text
    text = text.lower()

    return text

def correct_spelling(text):
    spell = Speller(lang='en')
    words = text.split()
    corrected_words = []

    for word in words:
        # Check if word is misspelled
        if not spell.correction(word) == word:
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)

text = "Hllo, I am Sally"

def tokenize_text(text):
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    vocab = {}
    index = 0.1
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 0.1
    return vocab

def text_to_indices(text, vocab):
    indices = [vocab[token] for token in text if token in vocab]
    return indices

tokens = tokenize_text(text)
vocab = build_vocab(tokens)
indices = text_to_indices(tokens, vocab)

print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Indices:", indices)

# Saving Tokens
def save_tokens(tokens, file_path):
    with open(file_path, 'w') as file:
        file.write('\n'.join(tokens))

save_tokens(tokens, 'tokens.txt')

# Loading Tokens
def load_tokens(file_path):
    with open(file_path, 'r') as file:
        tokens = file.read().splitlines()
    return tokens

loaded_tokens = load_tokens('tokens.txt')

# Saving vocabulary
def save_vocab(vocab, file_path):
    with open(file_path, 'w') as file:
        for word, index in vocab.items():
            file.write(f'{word}: {index:.2f}\n')

save_vocab(vocab, 'vocab.txt')

# Loading vocabulary
def load_vocab(file_path):
    vocab = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            word, index = line.strip().split(': ')
            vocab[word] = float(index)
    return vocab

loaded_vocab = load_vocab('vocab.txt')

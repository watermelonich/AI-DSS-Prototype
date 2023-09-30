import warnings
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.visualization import plot_histogram
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import difflib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define sentiment score ranges
sentiments = {
    "pure bad": range(-4, -5),
    "bad": range(-2, -3),
    "medium bad": range(-1, -2),
    "neutral": range(0, 1),
    "medium good": range(2, 3),
    "good": range(3, 4),
    "pure good": range(4, 6)
}

warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit-aer")
warnings.resetwarnings()

# Define sentiment intentions based on probabilities
sentiment_general = {
    "pure bad": "negative",
    "bad": "negative",
    "medium bad": "negative",
    "neutral": "neutral",
    "medium good": "positive",
    "good": "positive",
    "pure good": "positive"
}


# Encode sentiment scores onto qubits
def encode_sentiment(score):
    qc = QuantumCircuit(1)
    if score < 0:
        qc.x(0)
    elif score > 0:
        for _ in range(score):
            qc.h(0)
    qc.measure_all()
    return qc

# Get word meaning using NLTK's WordNet
def get_word_meaning(word):
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "Meaning not found"

# Calculate sentiment probabilities based on familiarity with reference sentences
def analyze_similar_sentences(test_sentence, reference_filename):
    _, similar_sentences = find_similar_sentence(reference_filename, test_sentence)
    
    sentiment_probabilities = {sentiment: 0 for sentiment in sentiment_general}
    reference_data = {}  # Dictionary to store reference data
    
    # Read the entire reference data into memory
    with open(reference_filename, 'r') as reference_file:
        lines = reference_file.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()[len("Sentence: "):]
            labels = lines[i + 1].strip()[len("Correct Label: "):].split(", ")
            reference_data[sentence] = labels

    for similar_sentence in similar_sentences:
        if similar_sentence in reference_data:
            labels = reference_data[similar_sentence]
            for label in labels:
                if label in sentiment_probabilities:
                    sentiment_probabilities[label] += 1

    total_sentiments = sum(sentiment_probabilities.values())
    
    if total_sentiments > 0:
        sentiment_probabilities = {sentiment: count / total_sentiments * 100 for sentiment, count in sentiment_probabilities.items()}
    else:
        sentiment_probabilities = {sentiment: 0 for sentiment in sentiment_general}
        
    return sentiment_probabilities

# Process the test data and write output to a file
def process_test_data(filename, reference_filename):
    with open(filename, 'r') as file:
        test_data = file.readlines()

    with open('testdata_output.txt', 'w') as output_file, open('missing_meanings.txt', 'w') as missing_file:
        for sentence in test_data:
            words = sentence.strip().split()
            score = len(words)

            # Analyze sentiment based on the score ranges
            sentiment_probabilities = {sentiment: 0 for sentiment in sentiment_general}

            for sentiment, score_range in sentiments.items():
                for s in sentiment_general:
                    if s in sentiment_probabilities:
                        sentiment_probabilities[s] += 1

                if score in score_range:
                    sentiment_probabilities[sentiment] += 1

            # Calculate sentiment probabilities based on similar sentences
            similar_labels, similar_sentences = find_similar_sentence(reference_filename, sentence)
            sentiment_probabilities = analyze_similar_sentences(similar_sentences, reference_filename)

            # Calculate sentiment intention based on matched sentiment label
            sentiment_intention = sentiment_general[similar_labels[0]] if similar_labels else "neutral"

            # Determine sentiment classification based on sentiment intention
            sentiment_classification = sentiment_intention

            # Calculate sentiment intention based on matched sentiment label
            similar_labels, _ = find_similar_sentence(reference_filename, sentence)
            if similar_labels:
                sentiment_intention = sentiment_general[similar_labels[0]]
            else:
                sentiment_intention = "neutral"  # Default to neutral if no match

            # Determine sentiment classification based on sentiment intention
            sentiment_classification = sentiment_intention

            # Check for similar sentence in reference data
            similar_labels, similar_sentences = find_similar_sentence(reference_filename, sentence)
            if similar_labels:
                label = similar_labels[0]

            # Perform part-of-speech tagging using NLTK
            tagged_words = pos_tag(word_tokenize(sentence))
            nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
            verbs = [word for word, pos in tagged_words if pos.startswith('VB')]

            # Determine the main word for analysis
            main_word = nouns[0] if nouns else (verbs[0] if verbs else words[0])

            meaning = get_word_meaning(main_word)

            if meaning == "Meaning not found":
                missing_file.write(f"Sentence: {sentence}\n")
            else:
                noun_meaning = get_word_meaning(nouns[0]) if nouns else "Noun meaning not found"
                verb_meaning = get_word_meaning(verbs[0]) if verbs else "Verb meaning not found"

                similar_labels = find_similar_sentence(reference_filename, sentence)
                if similar_labels:
                    label = similar_labels[0]

                if isinstance(label, list) and len(label) > 0:
                    label_value = label[0]
                else:
                    label_value = label

                qc = encode_sentiment(score)
                sim_backend = Aer.get_backend('aer_simulator')
                transpiled_qc = transpile(qc, backend=sim_backend)
                job = assemble(transpiled_qc, backend=sim_backend, shots=1024)
                result = sim_backend.run(job).result()
                counts = result.get_counts(transpiled_qc)
                total_shots = sum(counts.values())
                probabilities = {outcome: count / total_shots for outcome, count in counts.items()}

                output_file.write(f"Sentence: {sentence}\n")
                output_file.write(f"Main Word: {main_word}\n")
                output_file.write(f"Noun Meaning: {noun_meaning}\n")
                output_file.write(f"Verb Meaning: {verb_meaning}\n")

                # Check for similar sentence in reference data
                output_file.write(f"Label: {label_value}\n")
                output_file.write(f"Score: {score}\n")
                output_file.write(f"Sentiment Probabilities: {sentiment_probabilities}\n")
                output_file.write(f"Sentiment Intention: {sentiment_intention}\n")
                output_file.write(f"Sentiment Classification: {sentiment_classification}\n\n")

                if similar_sentences:
                    output_file.write("Source of Familiarity:\n")
                    for similar_sentence in similar_sentences:
                        output_file.write(f" - {similar_sentence}\n")
                output_file.write(f"Probabilities: {probabilities}\n\n")

def display_reference_data(filename, test_data, actual_labels):
    with open(filename, 'a') as reference_file:
        for sentence, label in zip(test_data, actual_labels):
            words = sentence.strip().split()
            score = len(words)
            
            # Identify important words by selecting nouns and verbs
            tagged_words = pos_tag(word_tokenize(sentence))
            important_words = []
            for word, pos in tagged_words:
                if pos.startswith('NN') or pos.startswith('VB'):
                    important_words.append(word)
                    if len(important_words) >= 3:
                        break
            
            if not important_words:
                main_word = words[0]
            else:
                main_word = " ".join(important_words)

            meaning = get_word_meaning(main_word)

            reference_file.write(f"Sentence: {sentence}")
            reference_file.write(f"Correct Label: {label}\n")
            reference_file.write(f"Sentence Meaning: {meaning}\n\n")

# Read reference data from reference_data.txt
def read_reference_data(reference_filename):
    reference_data = {}
    with open(reference_filename, 'r') as reference_file:
        lines = reference_file.readlines()
        current_main_word = None
        for line in lines:
            if line.startswith("Main Word:"):
                current_main_word = line.strip().split(": ")[1]
                current_label = None
            elif line.startswith("Correct Label:"):
                current_label = line.strip().split(": ")[1]
                if current_main_word not in reference_data:
                    reference_data[current_main_word] = []
                reference_data[current_main_word].append(current_label)
    return reference_data

# Function to update patterns with suggested labels from reference data
def update_patterns_with_reference_data(patterns_filename, reference_data_filename):
    reference_data = read_reference_data(reference_data_filename)
    updated_patterns = []

    with open(patterns_filename, 'r') as pattern_file:
        lines = pattern_file.readlines()
        current_pattern = {}

        for i, line in enumerate(lines):
            if line.startswith("Sentence:"):
                if current_pattern:
                    updated_patterns.append(current_pattern)
                current_pattern = {"sentence": line.strip().split(":")[1].strip()}
                current_pattern["pattern_lines"] = []
            current_pattern["pattern_lines"].append(line)

    # Now, integrate suggestion in the loop
    for pattern in updated_patterns:
        sentence = pattern["sentence"]
        words = sentence.strip().split()
        suggested_label = None
        
        for word in words:
            if word in reference_data:
                suggested_label = reference_data[word][0]
                break
        
        if suggested_label:
            pattern["pattern_lines"].append(f"Suggested Label: {suggested_label}\n\n")

    # Write the updated patterns back to the file
    with open(patterns_filename, 'w') as pattern_file:
        for pattern in updated_patterns:
            pattern_file.writelines(pattern["pattern_lines"])

    print("Patterns updated with suggested labels from reference data.")

# Process test data
with open('testdata.txt', 'r') as file:
    test_data = file.readlines()

def read_actual_labels(labels_filename):
    with open(labels_filename, 'r') as labels_file:
        actual_labels = labels_file.read().splitlines()
    return actual_labels

# Evaluate model accuracy
def evaluate_accuracy(actual_labels):
    with open('testdata_output.txt', 'r') as predictions_file:
        prediction_entries = predictions_file.read().split('\n\n')
        predictions = []

        for entry in prediction_entries:
            lines = entry.strip().split('\n')
            predicted_label = None
            
            for line in lines:
                if line.startswith("Label:"):
                    predicted_label = line.split(": ")[1]
                    break
            
            if predicted_label is not None:
                predictions.append(predicted_label)

    correct_predictions = sum(1 for actual, predicted in zip(actual_labels, predictions) if actual == predicted)
    total_predictions = len(actual_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy


def calculate_sentence_similarity(sentence1, sentence2):
    similarity = difflib.SequenceMatcher(None, sentence1, sentence2).ratio()
    return similarity

def find_similar_sentence(reference_filename, sentence):
    similar_labels = []
    similar_sentences = []

    with open(reference_filename, 'r') as reference_file:
        reference_lines = reference_file.readlines()
        for i in range(0, len(reference_lines), 4):
            ref_sentence = reference_lines[i].strip()[len("Sentence: "):]
            similarity = calculate_sentence_similarity(sentence, ref_sentence)
            if similarity > 0.8:  # Adjust the similarity threshold as needed
                label = reference_lines[i + 1].strip()[len("Correct Label: "):]
                similar_labels.append(label)
                similar_sentences.append(ref_sentence)

    return similar_labels, similar_sentences

# Process test data
with open('testdata.txt', 'r') as file:
    test_data = file.readlines()

# Read reference data
reference_data = read_reference_data('reference_data.txt')

# Process test data and identify similar sentences
for sentence in test_data:
    similar_labels, similar_sentences = find_similar_sentence('reference_data.txt', sentence)
    
    if similar_labels:
        print(f"Found similar sentence for '{sentence.strip()}': Using label '{similar_labels[0]}'")
        if similar_sentences:
            print("Source of familiarity:", similar_sentences[0])

# Process test data
with open('testdata.txt', 'r') as file:
    test_data = file.readlines()

process_test_data('testdata.txt', 'reference_data.txt')
print("Test data processed and output written to testdata_output.txt")

# Display reference data for the model
actual_labels = read_actual_labels('labels.txt')
display_reference_data('reference_data.txt', test_data, actual_labels)
print("Reference data written to reference_data.txt")

# Update patterns based on reference data
update_patterns_with_reference_data('testdata_output.txt', 'reference_data.txt')

# Read actual model predictions from a file or another source
with open('testdata_output.txt', 'r') as predictions_file:
    predictions = predictions_file.read().splitlines()

# Call the evaluate_accuracy function
accuracy = evaluate_accuracy(actual_labels)
print("Number of actual labels:", len(actual_labels))
print("Number of predictions:", len(predictions))
print(f"Model Accuracy: {accuracy:.2%}")

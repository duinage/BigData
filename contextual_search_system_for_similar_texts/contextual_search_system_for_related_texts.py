# The task is to create a system that offers five reviews that are similar to the given one.
# Author: Vadym Tunik.

import os
import re
import time
import string
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import pairwise_distances

# config
FOLDER_PATH = r'C:\Users\duina\repo\DA\task3\aclImdb\train\unsup'
DATA_FRACTION = 1.
CHOSEN_TEXT_INDEX = 42
NUM_RELATED_TO_FIND = 5
CHAR_LIMIT_FOR_TEXT = 250
USE_BIGRAMS = False

def clean_text(text: str) -> str:
    """Cleans text data by removing HTML tags, punctuation, and converting to lowercase."""
    try:
        if not isinstance(text, str):
             print(f"Warning: Expected string input for clean_text, got {type(text)}. Skipping HTML removal.")
        else:
             soup = BeautifulSoup(text, "html.parser")
             text = soup.get_text()
    except Exception as e:
        # Fallback for potential BeautifulSoup errors or if input wasn't string initially
        print(f"Warning: Error parsing HTML or processing input: {e}. Using regex fallback for HTML removal.")
        text = re.sub(r'<.*?>', '', text)

    chars_to_remove = string.punctuation + '—–―…’“”‘’'
    translation_table = str.maketrans('', '', chars_to_remove)
    text = text.translate(translation_table) # Remove punctuation.

    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()
    return text


class BagOfWords:
    """
    Creates a Bag of Words representation from a corpus of texts.
    Can generate representations based on unigrams or bigrams.
    """
    def __init__(self, use_bigrams: bool = False):
        """
        Initializes the BagOfWords model.

        Args:
            use_bigrams (bool): If True, use bigrams as tokens. Otherwise, use unigrams.
        """
        self.use_bigrams = use_bigrams
        self.vocabulary = []
        self.word_to_index = {}
        self.bow_matrix = np.array([])
        self._tokens_per_text = [] # Internal storage for tokens before matrix creation

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes text into unigrams or bigrams based on the flag."""
        tokens = text.split()
        if not self.use_bigrams:
            return tokens
        else:
            if len(tokens) < 2:
                return [] # Not enough words to form any bigrams
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            return bigrams

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        """
        Cleans texts, builds vocabulary, creates token lists, and generates the BoW matrix.

        Args:
            corpus (list[str]): A list of text documents.

        Returns:
            np.ndarray: The Bag of Words matrix (documents x vocabulary size).
        """
        print(f"Starting BoW process (use_bigrams={self.use_bigrams})...")
        start_time = time.time()

        print("Step 1/4: Cleaning and tokenizing texts...")
        vocabulary_set = set()
        self._tokens_per_text = []
        for text in corpus:
            cleaned_text = clean_text(text)
            tokens = self._tokenize(cleaned_text)
            self._tokens_per_text.append(tokens)
            vocabulary_set.update(tokens)

        print("Step 2/4: Building vocabulary...")
        self.vocabulary = sorted(list(vocabulary_set))
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}
        vocabulary_len = len(self.vocabulary)
        if vocabulary_len == 0:
             print("Warning: Vocabulary is empty. No tokens were generated.")
             self.bow_matrix = np.array([])
             return self.bow_matrix

        print(f"    vocabulary size: {vocabulary_len}")

        print("Step 3/4: Creating BoW matrix...")
        bow_matrix_list = []
        for tokens in self._tokens_per_text:
            vector = [0] * vocabulary_len
            for token in tokens:
                if token in self.word_to_index: # Check if token exists (it should)
                    index = self.word_to_index[token]
                    vector[index] += 1
            bow_matrix_list.append(vector)

        self.bow_matrix = np.array(bow_matrix_list)

        end_time = time.time()
        print(f"Step 4/4: BoW process finished. Shape: {self.bow_matrix.shape}")
        print(f"    total time: {end_time - start_time:.2f} seconds")

        return self.bow_matrix


if __name__ == "__main__":
    print(f"\nLoading data from: {FOLDER_PATH}")
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: Folder not found at {FOLDER_PATH}")
        exit()

    files = os.listdir(FOLDER_PATH)
    num_files_to_load = int(len(files) * DATA_FRACTION)
    files_to_load = files[:num_files_to_load]
    print(f"Loading {num_files_to_load} files ({DATA_FRACTION*100:.1f}% of total)...")

    texts = []
    for file in files_to_load:
        file_path = os.path.join(FOLDER_PATH, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    print(f"Loaded {len(texts)} texts.")

    if not texts:
        print("Error: No texts were loaded. Exiting.")
        exit()

    bow_model = BagOfWords(use_bigrams=USE_BIGRAMS)
    bow_matrix = bow_model.fit_transform(texts)

    if bow_matrix.size == 0:
        print("Error: BoW matrix is empty. Cannot proceed with similarity analysis.")
        exit()

    if CHOSEN_TEXT_INDEX < 0 or CHOSEN_TEXT_INDEX >= len(texts):
        print(f"Error: CHOSEN_TEXT_INDEX ({CHOSEN_TEXT_INDEX}) is out of bounds for the loaded texts (0-{len(texts)-1}).")
        exit()


    print(f"\nFinding {NUM_RELATED_TO_FIND} texts similar to text #{CHOSEN_TEXT_INDEX} using Euclidean distance...")

    chosen_vector = bow_matrix[CHOSEN_TEXT_INDEX].reshape(1, -1)
    distances = pairwise_distances(chosen_vector, bow_matrix, metric='euclidean')[0]
    sorted_indices = np.argsort(distances)

    chosen_text = texts[CHOSEN_TEXT_INDEX]
    print(f"\nChosen Text #{CHOSEN_TEXT_INDEX}")
    print(clean_text(chosen_text)[:CHAR_LIMIT_FOR_TEXT] + ('...' if len(clean_text(chosen_text)) > CHAR_LIMIT_FOR_TEXT else '')) # Show cleaned snippet

    print(f"\nTop {NUM_RELATED_TO_FIND} Most Similar Texts")

    found_count = 0
    for index in sorted_indices:
        if index == CHOSEN_TEXT_INDEX:  continue # Skip the chosen text itself
        if found_count >= NUM_RELATED_TO_FIND:  break

        related_text = texts[index]
        distance = distances[index]

        print(f"\nText #{index} with distance: {distance:.2f}")
        print(clean_text(related_text)[:CHAR_LIMIT_FOR_TEXT] + ('...' if len(clean_text(related_text)) > CHAR_LIMIT_FOR_TEXT else ''))
        found_count += 1
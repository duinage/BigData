# PROBLEM:
# Create a system that offers five IMDB reviews that are similar to the given one.
# (we intuitively believe that the user who wrote the review will be interested in finding a movie that evokes similar impressions)
# Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
# Author: Vadym Tunik.

# general
import os
import time
import numpy as np
from typing import List, Set, Optional
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize

# text/vocabulary cleaning
import re
import string
import collections
# import nltk
# nltk.download('stopwords')
from nltk.stem import SnowballStemmer # type: ignore
from nltk.corpus import stopwords # type: ignore
STOPWORDS_SET = set(stopwords.words("english"))
STEMMER = SnowballStemmer('english')

# config
FOLDER_PATH = r'C:\Users\duina\repo\DA\contextual_search_system_for_similar_texts_for_imdb_reviews\aclImdb\train\unsup'
DATA_FRACTION = 0.05
CHOSEN_TEXT_INDEX = 2025
NUM_RELATED_TO_FIND = 5
CHAR_LIMIT_FOR_TEXT = 500
USE_BIGRAMS = False
VOCAB_MIN_FREQUENCY = 5

def clean_text(text: str) -> str:
    """Cleans text data by removing HTML tags, punctuation, and converting to lowercase."""
    if not isinstance(text, str):
        raise TypeError(f"Expected string input for clean_text, got {type(text)}.")
    
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text) # remove general html tags
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    text = re.sub(r'\d+', '', text) # remove all numbers
    text = re.sub(r'\s+', ' ', text).strip()

    chars_to_remove = string.punctuation + '—–―…’“”‘’'
    translation_table = str.maketrans('', '', chars_to_remove)
    text = text.translate(translation_table) # Remove punctuation.
    return text

def postprocess_vocabulary(
    initial_vocabulary: list[str] | set[str],
    tokens_per_clean_text: list[str],
    min_frequency: int = 3,
    stopwords_set: Optional[Set[str]] = STOPWORDS_SET,
    stemmer: Optional[SnowballStemmer] = STEMMER
) -> list[str]:
    """
    Post-processes a vocabulary list by removing rare words, removing stop words, and applying stemming.
    lang: english.
    """
    word_counts = collections.Counter()
    for tokens in tokens_per_clean_text:
        word_counts.update(tokens)

    initial_vocab_set = set(initial_vocabulary)
    frequent_words = {word for word in initial_vocab_set if word_counts.get(word, 0) >= min_frequency}

    if stopwords_set:
        non_stop_words = {word for word in frequent_words if word not in stopwords_set}
    else:
        non_stop_words = frequent_words

    if stemmer:
        stemmed_vocabulary = set()
        for word in non_stop_words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word:
                stemmed_vocabulary.add(stemmed_word)
    else:
        stemmed_vocabulary = non_stop_words

    return sorted(list(stemmed_vocabulary))


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

    def _preprocess_corpus(self, corpus: List[str]) -> None:
        """Cleans and tokenizes the corpus, building an initial vocabulary set."""
        print("Cleaning and tokenizing texts...")
        initial_vocab_set: Set[str] = set()
        self._tokens_per_text = []

        cleaned_corpus = [clean_text(text) for text in corpus]

        for cleaned_text in cleaned_corpus:
            tokens = self._tokenize(cleaned_text)
            self._tokens_per_text.append(tokens)
            initial_vocab_set.update(tokens)

        print(f"____ Initial token count (unique): {len(initial_vocab_set)}")
        self._initial_vocabulary_set = initial_vocab_set

    def _build_final_vocabulary(self) -> None:
        """Builds the final vocabulary using postprocessing steps."""
        print("Building final vocabulary...")
        self.vocabulary = postprocess_vocabulary(
            initial_vocabulary=self._initial_vocabulary_set,
            tokens_per_clean_text=self._tokens_per_text,
            min_frequency=VOCAB_MIN_FREQUENCY,
            stopwords_set=STOPWORDS_SET,
            stemmer=STEMMER
        )
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}
        print(f"____ Final vocabulary size: {len(self.vocabulary)}")

    def _create_bow_matrix(self, num_texts: int) -> None:
        print("Creating BoW matrix...")
        vocabulary_len = len(self.vocabulary)
        if vocabulary_len == 0:
            print("Warning: Vocabulary is empty. No tokens were generated.")
            self.bow_matrix = np.array([])
            return None
        
        self.bow_matrix = np.zeros((num_texts, vocabulary_len))

        for text_index, tokens in enumerate(self._tokens_per_text):
            for token in tokens:
                processed_token = STEMMER.stem(token) if STEMMER else token
                if processed_token in self.word_to_index:
                    index = self.word_to_index[processed_token]
                    self.bow_matrix[text_index, index] += 1

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

        self._preprocess_corpus(corpus)
        self._build_final_vocabulary()
        self._create_bow_matrix(num_texts=len(corpus))

        end_time = time.time()
        print(f"BoW process finished. Shape: {self.bow_matrix.shape}")
        print(f"____ Total time: {end_time - start_time:.2f} seconds")
        return self.bow_matrix
    
    def save_vocabulary_to_csv(self, filepath: str = "vocabulary.txt"):
        np.savetxt(filepath, self.vocabulary, fmt='%s')


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

    normalized_bow_matrix = normalize(bow_matrix, norm='l2', axis=1)

    if CHOSEN_TEXT_INDEX < 0 or CHOSEN_TEXT_INDEX >= len(texts):
        print(f"Error: CHOSEN_TEXT_INDEX ({CHOSEN_TEXT_INDEX}) is out of bounds for the loaded texts (0-{len(texts)-1}).")
        exit()


    print(f"\nFinding {NUM_RELATED_TO_FIND} texts similar to text #{CHOSEN_TEXT_INDEX} using Euclidean distance...")

    chosen_vector_normalized = normalized_bow_matrix[CHOSEN_TEXT_INDEX].reshape(1, -1)
    distances = pairwise_distances(chosen_vector_normalized, normalized_bow_matrix, metric='euclidean')[0]
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
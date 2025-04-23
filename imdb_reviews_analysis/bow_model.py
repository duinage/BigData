# Contains the BagOfWords class and text processing utilities.
# Author: Vadym Tunik.

import re
import string
import collections
import time
import numpy as np
from typing import List, Set, Optional, Tuple, Dict

# Attempt to import NLTK resources, provide guidance if missing
try:
    from nltk.stem import SnowballStemmer # type: ignore
    from nltk.corpus import stopwords # type: ignore
    STOPWORDS_SET = set(stopwords.words("english"))
    STEMMER = SnowballStemmer('english')
except ImportError:
    print("NLTK not found or stopwords/stemmer resource missing.")
    print("Please install NLTK: pip install nltk")
    print("Then download required resources: python -m nltk.downloader stopwords")
    STOPWORDS_SET = set()
    STEMMER = None
    print("Proceeding without stopwords removal and stemming.")

def clean_text(text: str) -> str:
    """Cleans text data by removing HTML tags, punctuation, and converting to lowercase."""
    if not isinstance(text, str):
        raise TypeError(f"Expected string input for clean_text, got {type(text)}.")

    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text) # remove general html tags
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text) # remove control characters
    text = re.sub(r'\d+', '', text) # remove all numbers
    text = re.sub(r'\s+', ' ', text).strip() # replace multiple spaces with single and strip ends

    chars_to_remove = string.punctuation + '—–―…’“”‘’'
    translation_table = str.maketrans('', '', chars_to_remove)
    text = text.translate(translation_table) # Remove punctuation.
    return text

def postprocess_vocabulary(
    initial_vocabulary: Set[str],
    tokens_per_clean_text: List[List[str]],
    min_frequency: int = 3,
    stopwords_set: Optional[Set[str]] = STOPWORDS_SET,
    stemmer: Optional[SnowballStemmer] = STEMMER
) -> Tuple[List[str], Dict[str, int]]:
    """
    Post-processes a vocabulary set by removing rare words, removing stop words,
    and applying stemming. Returns the final sorted vocabulary list and a word-to-index mapping.
    lang: english.
    """
    print(f"____ Postprocessing: Min Freq={min_frequency}, Stopwords={'Yes' if stopwords_set else 'No'}, Stemming={'Yes' if stemmer else 'No'}")
    word_counts = collections.Counter()
    for tokens in tokens_per_clean_text:
        word_counts.update(tokens)

    frequent_words = {word for word, count in word_counts.items() if count >= min_frequency and word in initial_vocabulary}
    print(f"____ Words after frequency filter: {len(frequent_words)}")

    if stopwords_set:
        non_stop_words = {word for word in frequent_words if word not in stopwords_set}
        print(f"____ Words after stopword filter: {len(non_stop_words)}")
    else:
        non_stop_words = frequent_words

    final_vocabulary_set = set()
    if stemmer:
        for word in non_stop_words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word:
                final_vocabulary_set.add(stemmed_word)
        print(f"____ Unique stemmed words: {len(final_vocabulary_set)}")
    else:
        final_vocabulary_set = non_stop_words

    final_vocabulary_list = sorted(list(final_vocabulary_set))
    word_to_index = {word: i for i, word in enumerate(final_vocabulary_list)}

    return final_vocabulary_list, word_to_index


class BagOfWords:
    """
    Creates a Bag of Words representation from a corpus of texts.
    Can generate representations based on unigrams or bigrams.
    Includes text cleaning, vocabulary building with frequency filtering,
    stopword removal, and stemming.
    """
    def __init__(self, use_bigrams: bool = False):
        """
        Initializes the BagOfWords model.

        Args:
            use_bigrams (bool): If True, use bigrams as tokens. Otherwise, use unigrams.
        """
        self.use_bigrams = use_bigrams
        self.vocabulary: List[str] = []
        self.word_to_index: Dict[str, int] = {}
        self.bow_matrix: np.ndarray = np.array([])
        self._tokens_per_text: List[List[str]] = [] # Stores tokens for each doc after tokenization but before stemming/filtering
        self._initial_vocabulary_set: Set[str] = set()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into unigrams or bigrams based on the flag."""
        # Assumes text is already cleaned (lowercase, no punctuation/HTML)
        tokens = text.split()
        if not self.use_bigrams:
            return tokens
        else:
            if len(tokens) < 2:
                return [] # Not enough words to form any bigrams
            # Create bigrams like "word1_word2"
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            return bigrams

    def _preprocess_corpus(self, corpus: List[str]) -> None:
        """Cleans and tokenizes the corpus, building an initial vocabulary set."""
        print("Step 1: Cleaning and tokenizing texts...")
        self._tokens_per_text = []
        initial_vocab_set: Set[str] = set()

        for i, text in enumerate(corpus):
            if i % 1000 == 0 and i > 0:
                 print(f"____ Processed {i} texts...")
            cleaned_text = clean_text(text)
            tokens = self._tokenize(cleaned_text)
            self._tokens_per_text.append(tokens)
            initial_vocab_set.update(tokens)

        print(f"____ Initial token count (unique): {len(initial_vocab_set)}")
        self._initial_vocabulary_set = initial_vocab_set

    def _build_final_vocabulary(self, min_frequency: int) -> None:
        """Builds the final vocabulary using postprocessing steps."""
        print("Step 2: Building final vocabulary...")
        self.vocabulary, self.word_to_index = postprocess_vocabulary(
            initial_vocabulary=self._initial_vocabulary_set,
            tokens_per_clean_text=self._tokens_per_text,
            min_frequency=min_frequency,
            # Uses STOPWORDS_SET and STEMMER defined at the module level
            stopwords_set=STOPWORDS_SET,
            stemmer=STEMMER
        )
        print(f"____ Final vocabulary size: {len(self.vocabulary)}")

    def _create_bow_matrix(self) -> None:
        """Creates the BoW matrix based on the final vocabulary and processed tokens."""
        print("Step 3: Creating BoW matrix...")
        num_texts = len(self._tokens_per_text)
        vocabulary_len = len(self.vocabulary)

        if vocabulary_len == 0:
            print("Warning: Final vocabulary is empty. BoW matrix cannot be created.")
            self.bow_matrix = np.array([])
            return # Early exit

        self.bow_matrix = np.zeros((num_texts, vocabulary_len), dtype=np.float32)

        stemming_active = STEMMER is not None

        for text_index, doc_tokens in enumerate(self._tokens_per_text):
            if text_index % 1000 == 0 and text_index > 0:
                 print(f"____ Vectorized {text_index} texts...")

            token_counts = collections.Counter()
            for token in doc_tokens:
                processed_token = STEMMER.stem(token) if stemming_active and STEMMER else token
                token_counts[processed_token] += 1

            for token, count in token_counts.items():
                if token in self.word_to_index:
                    index = self.word_to_index[token]
                    self.bow_matrix[text_index, index] = count

        print(f"____ BoW matrix created. Shape: {self.bow_matrix.shape}")


    def fit_transform(self, corpus: List[str], vocab_min_frequency: int = 5) -> np.ndarray:
        """
        Processes the corpus: cleans texts, builds vocabulary, creates the BoW matrix.

        Args:
            corpus (list[str]): A list of text documents.
            vocab_min_frequency (int): Minimum frequency for a word to be included
                                       in the final vocabulary.

        Returns:
            np.ndarray: The Bag of Words matrix (documents x vocabulary size).
        """
        print(f"\nStarting BoW process (use_bigrams={self.use_bigrams})...")
        start_time = time.time()

        if not corpus:
            print("Warning: Input corpus is empty.")
            self.bow_matrix = np.array([])
            return self.bow_matrix

        self._preprocess_corpus(corpus)
        self._build_final_vocabulary(min_frequency=vocab_min_frequency)
        self._create_bow_matrix()

        end_time = time.time()
        print(f"\nBoW process finished.")
        if self.bow_matrix.size > 0 :
            print(f"____ Final Matrix Shape: {self.bow_matrix.shape}")
        else:
             print("____ Result: Empty BoW matrix (check vocabulary size and input).")
        print(f"____ Total time: {end_time - start_time:.2f} seconds")
        return self.bow_matrix

    def save_vocabulary(self, filepath: str = "vocabulary.txt") -> None:
        """Saves the final vocabulary list to a text file."""
        if not self.vocabulary:
            print("Vocabulary is empty, nothing to save.")
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for word in self.vocabulary:
                    f.write(word + '\n')
            print(f"Vocabulary saved to {filepath}")
        except IOError as e:
            print(f"Error saving vocabulary to {filepath}: {e}")

    def get_vocabulary(self) -> List[str]:
        """Returns the final vocabulary list."""
        return self.vocabulary

    def get_word_to_index_map(self) -> Dict[str, int]:
         """Returns the word-to-index dictionary."""
         return self.word_to_index


if __name__ == "__main__":
    print("Testing bow_model module...")
    sample_corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
        "<br>This contains HTML tags and numbers 123."
    ]

    # Test with unigrams
    print("\n--- Testing Unigrams ---")
    bow_unigram = BagOfWords(use_bigrams=False)
    matrix_unigram = bow_unigram.fit_transform(sample_corpus, vocab_min_frequency=1)
    print("Vocabulary (Unigram):", bow_unigram.get_vocabulary())
    # print("Matrix (Unigram):\n", matrix_unigram)
    # bow_unigram.save_vocabulary("vocab_unigram_test.txt")

    # Test with bigrams
    print("\n--- Testing Bigrams ---")
    bow_bigram = BagOfWords(use_bigrams=True)
    matrix_bigram = bow_bigram.fit_transform(sample_corpus, vocab_min_frequency=1)
    print("Vocabulary (Bigram):", bow_bigram.get_vocabulary())
    # print("Matrix (Bigram):\n", matrix_bigram)
    # bow_bigram.save_vocabulary("vocab_bigram_test.txt")

    print("\nModule test complete.")
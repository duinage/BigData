import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from typing import Tuple


# --- Configuration ---
FOLDER_PATH = r'C:\Users\duina\repo\DA\imdb_reviews_analysis\aclImdb\train\unsup'
DATA_FRACTION = 0.05 # Load 5% of the data for faster processing. Increase for more data.
CHOSEN_TEXT_INDEX = 2025 # Index of the review to find similar ones for (relative to loaded fraction)
NUM_RELATED_TO_FIND = 5   # How many similar reviews to display
CHAR_LIMIT_FOR_TEXT = 400 # Limit the displayed snippet length for readability

# BagOfWords Model Configuration
USE_BIGRAMS = False        # Use unigrams (False) or bigrams (True)
VOCAB_MIN_FREQUENCY = 5    # Minimum times a word must appear to be in the vocabulary
# --- End Configuration ---


def load_texts_from_folder(folder_path: str, fraction: float = 1.0) -> list[str]:
    """Loads text files from a specified folder."""
    print(f"\nLoading data from: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []

    try:
        files = os.listdir(folder_path)
        files = [f for f in files if f.endswith('.txt')]
    except OSError as e:
        print(f"Error accessing folder contents: {e}")
        return []

    if not files:
        print("No '.txt' files found in the specified folder.")
        return []

    num_files_to_load = int(len(files) * fraction)
    if num_files_to_load == 0 and fraction > 0 and len(files) > 0:
        num_files_to_load = 1
    files_to_load = files[:num_files_to_load]

    print(f"Attempting to load {num_files_to_load} files ({fraction*100:.1f}% of total {len(files)} text files)...")

    texts = []
    loaded_count = 0
    for file in files_to_load:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                loaded_count += 1
        except Exception as e:
            print(f"Warning: Error reading file {file}: {e}")

    print(f"Successfully loaded {loaded_count} out of {num_files_to_load} attempted texts.")
    return texts


def find_similar_texts(
    bow_matrix: np.ndarray,
    chosen_index: int,
    num_similar: int,
    metric: str = 'euclidian'
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds texts similar to the one at chosen_index using pairwise distances.

    Args:
        bow_matrix (np.ndarray): The Bag of Words matrix (docs x vocab).
        chosen_index (int): Index of the text to compare against.
        num_similar (int): Number of top similar texts to find (excluding itself).
        metric (str): Distance metric ('cosine', 'euclidean', etc.).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - indices: Indices of the most similar texts (sorted by similarity).
            - distances: Corresponding distances/similarities.
    """
    if chosen_index < 0 or chosen_index >= bow_matrix.shape[0]:
        raise IndexError(f"chosen_index {chosen_index} is out of bounds for the matrix with {bow_matrix.shape[0]} rows.")

    if bow_matrix.ndim != 2 or bow_matrix.shape[0] < 2:
        print("Warning: BoW matrix is not suitable for pairwise distance calculation (needs >= 2 documents).")
        return np.array([]), np.array([])

    print(f"\nCalculating similarity using '{metric}' distance...")
    if metric == 'cosine':
        distances = pairwise_distances(bow_matrix[chosen_index].reshape(1, -1), bow_matrix, metric=metric)[0]
    elif metric == 'euclidean':
        print("Normalizing vectors (L2 norm) for Euclidean distance...")
        normalized_bow_matrix = normalize(bow_matrix, norm='l2', axis=1)
        chosen_vector_normalized = normalized_bow_matrix[chosen_index].reshape(1, -1)
        distances = pairwise_distances(chosen_vector_normalized, normalized_bow_matrix, metric='euclidean')[0]
    else:
        print(f"Warning: Unsupported metric '{metric}'. Defaulting to 'cosine'.")
        distances = pairwise_distances(bow_matrix[chosen_index].reshape(1, -1), bow_matrix, metric='cosine')[0]

    sorted_indices = np.argsort(distances)
    similar_indices = sorted_indices[1 : num_similar + 1]
    similar_distances = distances[similar_indices]
    return similar_indices, similar_distances
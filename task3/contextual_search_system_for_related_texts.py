# The task is to create a system that offers the reader five texts that are similar in topic to the given one.
# Author: Vadym Tunik.
import os
import re
import string
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import pairwise_distances

# Dataset
# let's use Large Movie Review Dataset v1.0 (reviews from IMDB): https://ai.stanford.edu/~amaas/data/sentiment/.
# I will use only [train/unsup] reviews, because this will be enough for our task.

folder_path = r'C:\Users\duina\repo\DA\task3\aclImdb\train\unsup'
files = os.listdir(folder_path)

# use 5% of texts for develop
files = files[:int(len(files)*0.05)]

# load text
texts = []
for file in files:
    file_path = os.path.join(folder_path, file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    except Exception as e:
            print(f"Error reading file {file}: {e}")

#################################################################

def clean_text(text: str):
    """Cleans text data by removing HTML tags, punctuation, and converting to lowercase."""
    # Remove HTML tags using BeautifulSoup.
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
    except Exception as e:
        print(f"Error parsing HTML with BeautifulSoup: {e}. Using regex fallback.")
        text = re.sub(r'<.*?>', '', text)

    chars_to_remove = string.punctuation + '—–―…’“”‘’'
    text = text.translate(str.maketrans('', '', chars_to_remove)) # Remove punctuation.
    text = text.lower() # Convert to lowercase.
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace.
    return text

######## Bag of Words model (BoW) - by single word.      ########

# create vocabulary
vocabulary = set()
tokens_per_text = []
for text in texts:
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    vocabulary.update(tokens)
    tokens_per_text.append(tokens)

print(f"\nFinished cleaning and tokenizing. Vocabulary size: {len(vocabulary)}")

vocabulary = sorted(list(vocabulary))
vocabulary_len = len(vocabulary)

# create map "index by word"
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# create bow matrix that consist of bow vectors per text
bow_matrix = []
for tokens in tokens_per_text:
    vector = [0] * vocabulary_len
    for word in tokens:
        index = word_to_index[word]
        vector[index] += 1

    bow_matrix.append(vector)

print(f"\nFinished BoW matrix (single word). Shape: ({len(bow_matrix)}, {vocabulary_len})")

######## Bag of Words model (BoW) - by bigram.           ########


######## look for 5 related reviews to the selected one. ########
bow_matrix = np.array(bow_matrix)

chosen_text_index = 0
chosen_text = texts[chosen_text_index]

chosen_vector = bow_matrix[chosen_text_index]
chosen_vector = chosen_vector.reshape(1, -1)

distances = pairwise_distances(chosen_vector, bow_matrix, metric='euclidean')[0]
sorted_indices = np.argsort(distances)


# outputs the results
char_limit_for_text = 500
print(f"\n *** chosen text #{chosen_text_index}")
print(chosen_text[:char_limit_for_text] + '...')

num_related_to_find = 5
print(f"\n *** {num_related_to_find} most related reviews by euclid dist.")

found_count = 0
for index in sorted_indices:
    if index == chosen_text_index: continue # skip the chosen text itself
    if found_count >= num_related_to_find: break

    related_text = texts[index]
    distance = distances[index]

    print(f"\n *** related text #{index} with distances: {distance:.4f}")
    print(related_text[:char_limit_for_text] + '...')
    found_count += 1

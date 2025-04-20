# The task is to create a system that offers the reader five texts that are similar in topic to the given one.
# Author: Vadym Tunik.
import os
import re
import string
from mpi4py import MPI
from bs4 import BeautifulSoup

# Dataset
# let's use Large Movie Review Dataset v1.0 (reviews from IMDB): https://ai.stanford.edu/~amaas/data/sentiment/.
# I will use only [train/unsup] reviews, because this will be enough for our task.

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

folder_path = r'C:\Users\duina\repo\DA\task3\aclImdb\train\unsup'
files = os.listdir(folder_path)

# use 5% of texts for develop
files = files[:int(len(files)*0.05)]

if rank == 0:
    # load text
    texts = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
        except Exception as e:
             print(f"Rank {rank}: Error reading file {file}: {e}")

    
    # split texts
    if texts:
        texts = [texts[i::size] for i in range(size)]
    else:
        print(f"Rank {rank}: No texts were successfully loaded.")
        texts = [[] for _ in range(size)]
else:
    texts = None

# pass texts to each process
texts = comm.scatter(texts, root=0)

print(f"{rank=}, #texts={len(texts)}")
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

######## Bag of Words model (BoW) - by bigram.           ########



######## look for 5 similar reviews to the selected one. ########

# print the result
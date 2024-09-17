#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I want to create a 3D visualization of the word embeddings that we have learned.
I will use the t-SNE algorithm to reduce the dimensionality of the word embeddings to 3D.

I shall pass __sentences and the algorithm will produce a embedding width of 3, the second
word is the word itself, the first word is the word before and the third word is the word after.
We then create a 3D scatter plot of the word embeddings, and as we map new sets of 3 words we
also map the variations in the embeddings. For example, the words "cat eat fish" and "cat eat meat",
should have a similar embedding because the word "cat" is the same in both __sentences. But more we
should not be duplicating any word embeddings, so we should have a unique embedding for each word.
The map is also dynamic, so we can see the embeddings change as we change the __sentences. Thus, our
intelligence of the corpus map of our embeddings vocabulary is dynamic and can be visualized in 3D.
"""

import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from typing import List, Tuple


def create_word_embeddings(
    __sentences: List[str],
) -> Tuple[pandas.DataFrame, go.Figure]:
    """Create word embeddings for the given sentences and visualize them in 3D.

    :param __sentences: List[str]: The list of sentences to create word embeddings for.
    :return: Tuple[pandas.DataFrame, go.Figure]: The word embeddings and the 3D visualization.
    """
    # Split the __sentences into words
    words = [sentence.split() for sentence in __sentences]

    # Create a vocabulary of unique words
    vocabulary = set(word for sentence in words for word in sentence)

    # Create a mapping of words to unique integers
    word_to_int = {word: i for i, word in enumerate(vocabulary)}

    # Create a mapping of integers to words
    int_to_word = {i: word for word, i in word_to_int.items()}

    # Create a list of word sequences
    sequences = [[word_to_int[word] for word in sentence] for sentence in words]

    # Create a list of word embeddings
    embeddings = []
    for sequence in sequences:
        for i in range(1, len(sequence) - 1):
            embeddings.append([sequence[i - 1], sequence[i], sequence[i + 1]])

    # Convert the embeddings to a numpy array
    embeddings = numpy.array(embeddings)

    # Apply t-SNE to reduce the dimensionality of the embeddings to 3D
    n_samples = embeddings.shape[0]
    perplexity = min(
        30, n_samples - 1
    )  # Ensure perplexity is less than the number of samples
    tsne = TSNE(n_components=3, perplexity=perplexity)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create a DataFrame of the 3D embeddings
    __df = pandas.DataFrame(embeddings_3d, columns=["x", "y", "z"], dtype=float)

    # Create a 3D scatter plot of the embeddings
    unique_words = [int_to_word[int(i)] for i in numpy.unique(embeddings.flatten())]
    __fig = px.scatter_3d(__df, x="x", y="y", z="z", text=unique_words)
    __fig.update_traces(marker=dict(size=5))
    __fig.update_layout(title="Word Embeddings in 3D", title_font_size=30)

    return __df, __fig


if __name__ == "__main__":
    # Example usage
    sentences = [
        "cat eat fish",
        "cat eat meat",
        "dog eat meat",
        "dog eat bone",
        "fish eat worm",
        "fish eat insect",
        "bird eat worm",
        "bird eat insect",
    ]
    df, fig = create_word_embeddings(sentences)
    fig.show()

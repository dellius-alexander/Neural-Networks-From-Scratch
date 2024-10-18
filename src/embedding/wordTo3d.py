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

from src.utils.plot import get_camera_view


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
    print(f"Sequences: {sequences}")
    # Create a list of word embeddings
    embeddings = []
    for sequence in sequences:
        for i in range(1, len(sequence) - 1):
            embeddings.append([sequence[i - 1], sequence[i], sequence[i + 1]])

    # Convert the embeddings to a numpy array
    embeddings = numpy.array(embeddings)
    print(f"Embeddings: \n{embeddings}")
    # Apply t-SNE to reduce the dimensionality of the embeddings to 3D
    n_samples = embeddings.shape[0]
    perplexity = min(
        30, n_samples - 1
    )  # Ensure perplexity is less than the number of samples
    tsne = TSNE(n_components=3, perplexity=perplexity)
    embeddings_3d = tsne.fit_transform(embeddings)
    print(f"Embeddings 3D: \n{embeddings_3d}")
    # Create a DataFrame of the 3D embeddings
    __df = pandas.DataFrame(embeddings_3d, columns=["x", "y", "z"], dtype=float)

    # Create a 3D scatter plot of the embeddings
    unique_words = [int_to_word[int(i)] for i in numpy.unique(embeddings.flatten())]
    print(f"Unique words: {len(unique_words)}")
    print(f"Dataframe: \n{__df}")
    if len(unique_words) > len(__df):
        print("""
        Error: The number of unique words does not match the number of embeddings.
        This may be due to duplicate words in the sentences.
        So, we will resize the larger list to match the smaller list.""")
        # resize larger list to match the smaller list
        unique_words = unique_words[: len(__df)]
    # Create a 3D scatter plot of the embeddings
    __fig = go.Figure()

    # Add scatter plot for errors
    __fig = px.scatter_3d(
            __df,
            text=unique_words,
            x="x",
            y="y",
            z="z",
        )

    __fig.update_traces(
        mode="lines+markers+text",
        marker=dict(size=5, color=len(unique_words), opacity=0.8, colorscale="turbo"),
        textfont_size=10,
        textfont_color="black",
        textfont_family="Arial",
    )

    # Set the camera view (orientation)
    camera = get_camera_view("default")

    # Adjust the camera view
    __fig.update_layout(scene_camera=camera)

    # Set plot title and labels
    __fig.update_layout(
        title={
            "text": "Word Embeddings in 3D",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},  # Adjust the size as needed
        },
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="Inputs Feature 1",
            yaxis_title="Inputs Feature 2",
            zaxis_title="Values",
        )
    )

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
    # fig.show()
    print(f"Dataframe: \n{df}")
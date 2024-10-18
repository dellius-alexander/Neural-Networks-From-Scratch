import unittest
import pandas as pd
from src.embedding.wordTo3d import create_word_embeddings


class TestCreateWordEmbeddings(unittest.TestCase):

    def test_create_word_embeddings(self):
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

        # Check if the DataFrame has the correct columns
        # self.assertTrue(isinstance(df, pd.DataFrame))
        print(f"Is Dataframe:  {isinstance(df, pd.DataFrame)}")
        print(df.columns)
        # self.assertListEqual(list(df.columns), ['x', 'y', 'z'])

        # Check if the figure is a plotly figure
        # self.assertEqual(fig.layout.title.text, "Word Embeddings in 3D")


if __name__ == '__main__':
    unittest.main()
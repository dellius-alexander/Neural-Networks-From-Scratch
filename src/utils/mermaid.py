"""Utilities for working with Mermaid graphs in Jupyter notebooks.

Credit: https://gist.github.com/MLKrisJohnson/2d2df47879ee6afd3be9d6788241fe99

This module provides functions for working with Mermaid graphs in Jupyter notebooks. The functions allow you to:

- Display a Mermaid graph in a Jupyter notebook cell.
- Generate a URL that will display the graph in a web browser.
- Save the graph as a PNG file.
- Load a graph from a file and display it in a Jupyter notebook cell.

"""

import base64
from typing import Annotated

from IPython.core.display_functions import DisplayHandle
from IPython.display import Image, display

from src.utils.logger import getLogger

log = getLogger(__name__)

MermaidGraph = Annotated[str, bytes, "A string containing a Mermaid-format graph"]
Bytes = Annotated[bytes, "A bytes object"]
Path = Annotated[str, bytes, "A path to a file"]


def mm_ink(graphbytes: Bytes) -> str:
    """Given a bytes object holding a Mermaid-format graph, return a
    URL that will generate the image.

    :param graphbytes: (bytes): The Mermaid-format graph
    :return: (str): The URL for displaying the graph
    """
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url_link = "https://mermaid.ink/img/" + base64_string
    return url_link


def mm_display(graphbytes: Bytes) -> DisplayHandle:
    """Given a bytes object holding a Mermaid-format graph, display it.

    :param graphbytes: (bytes): The Mermaid-format graph
    :return: (DisplayHandle): The display handle for the graph
    """
    return display(Image(url=mm_ink(graphbytes)))


def mm(graph: MermaidGraph) -> DisplayHandle:
    """Given a string containing a Mermaid-format graph, display it.

    :param graph: (str): The Mermaid-format graph
    :return: (DisplayHandle): The display handle for the graph
    """
    graphbytes: bytes = graph.encode("ascii")
    return mm_display(graphbytes)


def mm_link(graph: Bytes) -> MermaidGraph:
    """Given a string containing a Mermaid-format graph, return URL for display.

    :param graph: (str): The Mermaid-format graph
    :return: (str): The URL for displaying the graph
    """
    graphbytes = graph.encode("ascii")
    return mm_ink(graphbytes)


def mm_from_file(path: str) -> DisplayHandle:
    """Given a path to a file containing a Mermaid-format graph, display
    the graph in a Jupyter notebook cell or IPython display.

    :param path: (str): The path to the file containing the Mermaid graph
    :return: (DisplayHandle): The display handle for the graph
    """
    with open(path, 'rb') as f:
        graphbytes = f.read()
    return display(Image(graphbytes))


def mm_save_as_png(graph: MermaidGraph, output_path: str,  mode: str = "w",) -> str:
    """
    Save a Mermaid graph as a PNG file
    :param graph: (MermaidGraph): The Mermaid graph
    :param output_path: (str): The path to save the PNG file
    :param mode: (str): The mode to open the file; default is "w" for write/overwrite and "a" for append/create new
    :return: (str): The path to the saved PNG file
    """
    import requests, os
    # Generate the Mermaid graph and get the DisplayHandle
    graphbytes = graph.encode("ascii")
    url = mm_ink(graphbytes)

    # Fetch the image from the URL
    response = requests.get(url)
    assert response.status_code == 200, f"Failed to fetch image: {response.status_code}" # Ensure we get a good response
    # response.raise_for_status()  # Ensure we notice bad responses

    # Ensure the output path is a PNG file
    image_filename = output_path.split("/")[-1].split(".")[0]
    output_path = "/".join(output_path.split("/")[0:-1])
    output_file_path = output_path + "/" + image_filename + ".png"

    # check if path exists and throw error if it does
    if not os.path.exists(output_path):
        raise FileExistsError(f"Path does not exist: {output_path}")

    # check for mode and create a new file if it does not exist
    if mode == "a":
        # check if file exists and create a new file if it does
        output_file_path = create_file_if_exists(output_file_path)
    elif mode == "w":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Save the image as a PNG file
    with open(output_file_path, 'wb') as f:
        f.write(response.content)
        f.close()
    return output_file_path


def create_file_if_exists(path: str, **kwargs) -> str:
    """
    Create a new file if the file already exists
    :param path: (str): The path to the file
    :param kwargs: (dict): Additional keyword arguments
    :return: (str): The path to the new file
    """
    import os
    index = kwargs.get("index", 1)

    try:
        path = os.path.abspath(path)
        if os.path.exists(path):
            raise FileExistsError(f"File already exists: {path}")
    except FileExistsError as e:
        if path.__contains__("_"):
            __prefix = "".join(path.split("_")[:-1])
            __suffix = path.split(".")[0].split("_")[-1]
            if __suffix.isdigit():
                path = __prefix + f"_{int(__suffix) + 1}.png"
            else:
                path = __prefix + f"_1.png"
            return create_file_if_exists(path, index=index + 1)
        else:
            __prefix = ".".join(path.split(".")[:-1])
            __suffix = path.split(".")[-1]
            path = __prefix + f"_{index}.png"
            return create_file_if_exists(path, index=index + 1)
    return path


def mm_encode(graph: MermaidGraph) -> Bytes:
    """Given a string containing a Mermaid-format graph, return bytes.

    :return: (Bytes): A bytes object holding the Mermaid-format graph.
    """
    return graph.encode("ascii")


def mm_decode(graphbytes: Bytes) -> MermaidGraph:
    """Given a bytes object holding a Mermaid-format graph, return the string.

    :return: (MermaidGraph): A string containing the Mermaid-format graph.
    """
    base64_bytes = base64.b64decode(graphbytes)
    return base64_bytes.decode("ascii")


# Example usage
if __name__ == "__main__":
    mermaid_diagram = '''
    %%{
      init: {
        'theme': 'forest',
        'themeVariables': {
          'primaryColor': '#BB2528',
          'primaryTextColor': '#fff',
          'primaryBorderColor': '#7C0000',
          'lineColor': '#F8B229',
          'secondaryColor': '#006100',
          'secondaryBorderColor': '#003700',
          'secondaryTextColor': '#fff000',
          'tertiaryColor': '#fff999',
          'tertiaryBorderColor': '#000999',
          'orientation': 'landscape'
        }
      }
    }%%

    graph TD
        subgraph Input Layer
            direction LR
            I1((1))
            I2((2))
            I3((3))
            I4((4))
        end

        subgraph Hidden Layer 1
            direction LR
            H1_1((H1_1))
            H1_2((H1_2))
            H1_3((H1_3))
            B1_1["B1_1"]
            B1_2["B1_2"]
            B1_3["B1_3"]
        end

        subgraph Hidden Layer 2
            direction LR
            H2_1((H2_1))
            H2_2((H2_2))
            H2_3((H2_3))
            H2_4((H2_4))
            B2_1["B2_1"]
            B2_2["B2_2"]
            B2_3["B2_3"]
            B2_4["B2_4"]
        end

        subgraph Output Layer
            direction LR
            O((O))
            B3["B3"]
        end

        I1 -->|W1_1| H1_1
        I1 -->|W1_2| H1_2
        I1 -->|W1_3| H1_3

        I2 -->|W2_1| H1_1
        I2 -->|W2_2| H1_2
        I2 -->|W2_3| H1_3

        I3 -->|W3_1| H1_1
        I3 -->|W3_2| H1_2
        I3 -->|W3_3| H1_3

        I4 -->|W4_1| H1_1
        I4 -->|W4_2| H1_2
        I4 -->|W4_3| H1_3

        H1_1 -->|W5_1| H2_1
        H1_1 -->|W5_2| H2_2
        H1_1 -->|W5_3| H2_3
        H1_1 -->|W5_4| H2_4

        H1_2 -->|W6_1| H2_1
        H1_2 -->|W6_2| H2_2
        H1_2 -->|W6_3| H2_3
        H1_2 -->|W6_4| H2_4

        H1_3 -->|W7_1| H2_1
        H1_3 -->|W7_2| H2_2
        H1_3 -->|W7_3| H2_3
        H1_3 -->|W7_4| H2_4

        H2_1 -->|W8_1| O
        H2_2 -->|W8_2| O
        H2_3 -->|W8_3| O
        H2_4 -->|W8_4| O

        B1_1 --> H1_1
        B1_2 --> H1_2
        B1_3 --> H1_3

        B2_1 --> H2_1
        B2_2 --> H2_2
        B2_3 --> H2_3
        B2_4 --> H2_4

        B3 --> O
       '''

    # Save the mermaid diagram
    mm_save_as_png(mermaid_diagram, '../../assets/images/hidden-layer-forward-pass.png')
    # Generate the mermaid diagram
    mm_from_file('../../assets/images/hidden-layer-forward-pass.png')

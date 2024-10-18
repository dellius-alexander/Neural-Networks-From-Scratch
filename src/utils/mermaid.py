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
import requests, os
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
    if isinstance(graph, str):
        graphbytes = graph.encode("ascii")
    else:
        graphbytes = graph
    return mm_ink(graphbytes)


def display_image_from_file(path: str) -> DisplayHandle:
    """Given a path to a file containing a Mermaid-format graph, display
    the graph in a Jupyter notebook cell or IPython display.

    :param path: (str): The path to the file containing the Mermaid graph
    :return: (DisplayHandle): The display handle for the graph
    """
    with open(path, "rb") as f:
        graphbytes = f.read()
    return display(Image(graphbytes))


def mm_save_as_png(
    graph: MermaidGraph,
    output_file_path: str,
    mode: str = "w",
) -> Path:
    """
    Save a Mermaid graph as a PNG file
    :param graph: (MermaidGraph): The Mermaid graph
    :param output_file_path: (str): The path to save the PNG file
    :param mode: (str): The mode to open the file; default is "w" for write/overwrite and "a" for append/create new
    :return: (Path): The path to the saved PNG file
    """
    # Generate the Mermaid graph and get the DisplayHandle
    graph_bytes = graph.encode("ascii")
    url = mm_ink(graph_bytes)

    # Fetch the image from the URL
    response = requests.get(
        url=url,
        stream=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Host": "mermaid.ink",
        },
    )
    assert (
        response.status_code == 200
    ), f"Failed to fetch image: {response.status_code}"  # Ensure we get a good response
    # response.raise_for_status()  # Ensure we notice bad responses
    log.debug(f"Response status code: {response.status_code}")

    # Ensure the output path is a PNG file
    image_filename = output_file_path.split("/")[-1].split(".")[0]
    output_dir = "/".join(output_file_path.split("/")[0:-1])
    output_file_path = os.path.abspath(output_dir + "/" + image_filename + ".png")
    log.debug(f"Output file path: \n{output_file_path}")

    # check if path exists and throw error if it does
    if not os.path.exists(output_dir):
        raise FileExistsError(f"Path does not exist: {output_dir}")

    # check for mode and create a new file if it does not exist
    if mode == "a":
        # check if file exists and create a new file if it does
        output_file_path = create_file_if_exists(output_file_path)
    elif mode == "w":
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Save the image as a PNG file
    with open(output_file_path, "wb") as f:
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
    mermaid_diagram = """---
title: Neural Network with 2 Hidden Layers
---

%%{
  init: {
    'theme': 'forest',
    'themeVariables': {
      'primaryColor': '#BB2528',
      'primaryTextColor': '#fff',
      'primaryBorderColor': '#28bb25',
      'lineColor': '#28bb25',
      'secondaryColor': '#006100',
      'secondaryBorderColor': '#003700',
      'secondaryTextColor': '#fff000',
      'tertiaryColor': '#fff999',
      'tertiaryBorderColor': '#fff000',
      'orientation': 'landscape'
    }
  }
}%%

flowchart LR
    subgraph subGraph0["Output Layer D1"]
      direction TB
        D1("$$\\text{activation}_{D_1} (\\sum_{i=1}^{n} w_i \\cdot x_i + b )$$")
    end
    
    subgraph subGraph1["Hidden Layer 2"]
      direction TB
        C1("$$\\text{activation}_{C_1}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        C2("$$\\text{activation}_{C_2}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        C3("$$\\text{activation}_{C_3}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        C4("$$\\text{activation}_{C_4}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
    end
    
    subgraph subGraph2["Hidden Layer 1"]
      direction TB
        B1("$$\\text{activation}_{B_1}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        B2("$$\\text{activation}_{B_2}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        B3("$$\\text{activation}_{B_3}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
        B4("$$\\text{activation}_{B_4}\\left(\\sum_{i=1}^{n} w_i \\cdot x_i + b \\right)$$")
    end
    
    subgraph subGraph3["Input Layer"]
      direction TB
        A1["$$\\text{input}_{1}$$"]
        A2["$$\\text{input}_{2}$$"]
        A3["$$\\text{input}_{3}$$"]
        A4["$$\\text{input}_{4}$$"]
    end
    
    %% Weights: Layer 1
    subgraph WeightsLayer1["Layer 1 Weights"]
      direction TB
        W11["$$\\text{w}_{1_1}$$"]
        W12["$$\\text{w}_{1_2}$$"]
        W13["$$\\text{w}_{1_3}$$"]
        W14["$$\\text{w}_{1_4}$$"]
        W21["$$\\text{w}_{2_1}$$"]
        W22["$$\\text{w}_{2_2}$$"]
        W23["$$\\text{w}_{2_3}$$"]
        W24["$$\\text{w}_{2_4}$$"]
        W31["$$\\text{w}_{3_1}$$"]
        W32["$$\\text{w}_{3_2}$$"]
        W33["$$\\text{w}_{3_3}$$"]
        W34["$$\\text{w}_{3_4}$$"]
        W41["$$\\text{w}_{4_1}$$"]
        W42["$$\\text{w}_{4_2}$$"]
        W43["$$\\text{w}_{4_3}$$"]
        W44["$$\\text{w}_{4_4}$$"]
        Bias11["$$\\text{b}_{1_1}$$"]
        Bias12["$$\\text{b}_{1_2}$$"]
        Bias13["$$\\text{b}_{1_3}$$"]
        Bias14["$$\\text{b}_{1_4}$$"]
    end

    subgraph WeightsLayer2["Layer 2 Weights"]
      direction TB
        W211["$$\\text{w}_{1_1}$$"]
        W212["$$\\text{w}_{1_2}$$"]
        W213["$$\\text{w}_{1_3}$$"]
        W214["$$\\text{w}_{1_4}$$"]
        W221["$$\\text{w}_{2_1}$$"]
        W222["$$\\text{w}_{2_2}$$"]
        W223["$$\\text{w}_{2_3}$$"]
        W224["$$\\text{w}_{2_4}$$"]
        W231["$$\\text{w}_{3_1}$$"]
        W232["$$\\text{w}_{3_2}$$"]
        W233["$$\\text{w}_{3_3}$$"]
        W234["$$\\text{w}_{3_4}$$"]
        W241["$$\\text{w}_{4_1}$$"]
        W242["$$\\text{w}_{4_2}$$"]
        W243["$$\\text{w}_{4_3}$$"]
        W244["$$\\text{w}_{4_4}$$"]
        Bias21["$$\\text{b}_{2_1}$$"]
        Bias22["$$\\text{b}_{2_2}$$"]
        Bias23["$$\\text{b}_{2_3}$$"]
        Bias24["$$\\text{b}_{2_4}$$"]
    end
    
    subgraph WeightsLayer3["Output Layer Weights"]
      direction TB
        W311["$$\\text{w}_{1_1}$$"]
        W321["$$\\text{w}_{1_2}$$"]
        W331["$$\\text{w}_{1_3}$$"]
        W341["$$\\text{w}_{1_4}$$"]
        Bias31["$$\\text{b}_{3_1}$$"]
    end
    
    %% Layers
    A1 --- W11 --> B1
    A1 --- W12 --> B2
    A1 --- W13 --> B3
    A1 --- W14 --> B4
    A2 --- W21 --> B1
    A2 --- W22 --> B2
    A2 --- W23 --> B3
    A2 --- W24 --> B4
    A3 --- W31 --> B1
    A3 --- W32 --> B2
    A3 --- W33 --> B3
    A3 --- W34 --> B4
    A4 --- W41 --> B1
    A4 --- W42 --> B2
    A4 --- W43 --> B3
    A4 --- W44 --> B4
    
    %% Layer 2
    B1 --- W211 --> C1
    B1 --- W212 --> C2
    B1 --- W213 --> C3
    B1 --- W214 --> C4
    B2 --- W221 --> C1
    B2 --- W222 --> C2
    B2 --- W223 --> C3
    B2 --- W224 --> C4
    B3 --- W231 --> C1
    B3 --- W232 --> C2
    B3 --- W233 --> C3
    B3 --- W234 --> C4
    B4 --- W241 --> C1
    B4 --- W242 --> C2
    B4 --- W243 --> C3
    B4 --- W244 --> C4
    
    %% Output Layer
    C1 --- W311 --> D1
    C2 --- W321 --> D1
    C3 --- W331 --> D1
    C4 --- W341 --> D1
    
    %% Bias
    Bias11 --> B1
    Bias12 --> B2
    Bias13 --> B3
    Bias14 --> B4
    Bias21 --> C1
    Bias22 --> C2
    Bias23 --> C3
    Bias24 --> C4
    Bias31 --> D1
    
    %% Style
    style Bias11 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias12 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias13 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias14 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias21 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias22 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias23 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias24 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style Bias31 fill:#42f569,stroke-width:1px,stroke-dasharray: 1
    style subGraph3 color:#000000,fill:none
    style subGraph2 color:#000000,fill:none
    style subGraph1 color:#000000,fill:none
    style subGraph0 color:#000000,fill:none
    style D1 fill:#baace6,stroke:#003700,stroke-width:1px
    style C1 fill:#fff000,stroke:#003700,stroke-width:1px
    style C2 fill:#fff000,stroke:#003700,stroke-width:1px
    style C3 fill:#fff000,stroke:#003700,stroke-width:1px
    style C4 fill:#fff000,stroke:#003700,stroke-width:1px
    style B1 fill:#fff000,stroke:#003700,stroke-width:1px
    style B2 fill:#fff000,stroke:#003700,stroke-width:1px
    style B3 fill:#fff000,stroke:#003700,stroke-width:1px
    style B4 fill:#fff000,stroke:#003700,stroke-width:1px
    style A1 fill:#42a7f5,stroke:#003700,stroke-width:1px
    style A2 fill:#42a7f5,stroke:#003700,stroke-width:1px
    style A3 fill:#42a7f5,stroke:#003700,stroke-width:1px
    style A4 fill:#42a7f5,stroke:#003700,stroke-width:1px
    style WeightsLayer1 color:#000000,fill:none,stroke:#003700,stroke-width:1px
    style WeightsLayer2 color:#000000,fill:none,stroke:#003700,stroke-width:1px
    style WeightsLayer3 color:#000000,fill:none,stroke:#003700,stroke-width:1px
"""

    # Save the mermaid diagram
    __file_path = mm_save_as_png(
        mermaid_diagram, "../../assets/images/hidden-layer-forward-pass.png"
    )
    # Generate the mermaid diagram
    display_image_from_file(__file_path)

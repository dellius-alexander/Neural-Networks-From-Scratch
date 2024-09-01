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
    """Given a bytes object holding a Mermaid-format graph, return a URL that will generate the image."""
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url_link = "https://mermaid.ink/img/" + base64_string
    return url_link


def mm_display(graphbytes: Bytes) -> DisplayHandle:
    """Given a bytes object holding a Mermaid-format graph, display it."""
    return display(Image(url=mm_ink(graphbytes)))


def mm(graph: MermaidGraph) -> DisplayHandle:
    """Given a string containing a Mermaid-format graph, display it."""
    graphbytes: bytes = graph.encode("ascii")
    return mm_display(graphbytes)


def mm_link(graph: Bytes) -> MermaidGraph:
    """Given a string containing a Mermaid-format graph, return URL for display."""
    graphbytes = graph.encode("ascii")
    return mm_ink(graphbytes)


def mm_path(path: Bytes) -> Bytes:
    """Given a path to a file containing a Mermaid-format graph, display it"""
    with open(path, 'rb') as f:
        graphbytes = f.read()
    mm_display(graphbytes)


def mm_from_file(path: Bytes) -> Bytes:
    """Given a path to a file containing a Mermaid-format graph, return URL for display.

    :return: (Bytes): A bytes object holding the URL for the image.
    """
    with open(path, 'rb') as f:
        graphbytes = bytes(f.read())
    return mm_ink(graphbytes).encode("ascii")


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



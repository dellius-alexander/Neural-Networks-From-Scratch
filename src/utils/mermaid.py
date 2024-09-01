import base64
from typing import Annotated

from IPython.display import Image, display


MermaidGraph = Annotated[str, bytes, "A string containing a Mermaid-format graph"]
Bytes = Annotated[bytes, "A bytes object"]
Path = Annotated[str, bytes, "A path to a file"]


def mm_ink(graphbytes: Bytes) -> str:
    """Given a bytes object holding a Mermaid-format graph, return a URL that will generate the image."""
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url_link = "https://mermaid.ink/img/" + base64_string
    return url_link


def mm_display(graphbytes: Bytes) -> None:
    """Given a bytes object holding a Mermaid-format graph, display it."""
    display(Image(url=mm_ink(graphbytes)))


def mm(graph: MermaidGraph) -> None:
    """Given a string containing a Mermaid-format graph, display it."""
    graphbytes: bytes = graph.encode("ascii")
    mm_display(graphbytes)


def mm_link(graph: Bytes) -> MermaidGraph:
    """Given a string containing a Mermaid-format graph, return URL for display."""
    graphbytes = graph.encode("ascii")
    return mm_ink(graphbytes)


def mm_path(path: Bytes) -> Bytes:
    """Given a path to a file containing a Mermaid-format graph, display it"""
    with open(path, 'rb') as f:
        graphbytes = f.read()
    mm_display(graphbytes)


def mm_file(path: Bytes) -> Bytes:
    """Given a path to a file containing a Mermaid-format graph, return URL for display.

    :return: (Bytes): A bytes object holding the URL for the image.
    """
    with open(path, 'rb') as f:
        graphbytes = bytes(f.read())
    return mm_ink(graphbytes)


def mm_save(graph: MermaidGraph, path: Bytes) -> Path:
    """Given a string containing a Mermaid-format graph, save it to a file.

    :return: (Path): The path to the saved file.
    """
    graphbytes = graph.encode("ascii")
    with open(path, 'wb') as f:
        f.write(graphbytes)
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
    mm_graph = """
    graph TD
        A[Christmas] -->|Get money| B(Go shopping)
        B --> C{Let me think}
        C -->|One| D[Laptop]
        C -->|Two| E[iPhone]
        C -->|Three| F[Car]
    """
    mm(mm_graph)


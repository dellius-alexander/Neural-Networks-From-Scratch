from typing import Union, Any, List, Annotated
import numpy as np


def encode(
        raw_labels_or_words: Annotated[Union[np.ndarray[Any], List[Any]], "The raw labels or words to encode"],
        dtype: Annotated[np.dtype, "The data type of the encoded labels or words to use"] = np.float32
) -> np.ndarray:
    """Encode the raw labels or words into ASCII values.

    :param raw_labels_or_words: (np.ndarray[Any] or List): The raw labels or words to encode.
    :param dtype: (np.dtype): The data type of the encoded labels or words.
    :return: (np.ndarray): The encoded labels or words in the form of 2D numpy matrix.
    """
    encodings = []
    encoding = []
    ndim = np.ndim(raw_labels_or_words) + 1
    for label in raw_labels_or_words:
        # base case for str
        if isinstance(label, str):
            label = label.lower()
            encoding.clear()
            for char in label:
                if len(char.encode('utf-8')) > 1:
                    encoding.extend(encode([char], dtype))
                elif char == " ":
                    encoding.append(" ".encode('utf-8').__hash__())
                elif char == "\n":
                    encoding.append("\n".encode('utf-8').__hash__())
                elif char == "\t":
                    encoding.append("\t".encode('utf-8').__hash__())
                elif char == "\r":
                    encoding.append("\r".encode('utf-8').__hash__())
                else:
                    encoding.append(ord(char))
            normalized_mean = np.mean(encoding)
            encodings.append(normalized_mean)
        # base case for int
        elif isinstance(label, int):
            encodings.append(label)
        # base case for float
        elif isinstance(label, float):
            encodings.append(label)
        # recursive case for list or np.ndarray
        elif isinstance (label, (np.ndarray, list)):
            encodings.extend(encode(label, dtype))
    return np.array([encodings], dtype=dtype, ndmin=ndim)



if __name__ == "__main__":
    labels = ["cat", "dog", "fish", "elephant", "lion", "tiger", "bear"]
    encoded_labels = encode(labels)
    print(f"Encoded labels: \n{encoded_labels}\n")
    print(f"Data type: \n{encoded_labels.dtype}\n")
    print(f"Shape: \n{encoded_labels.shape}\n")
    print(f"Size: \n{encoded_labels.size}\n")
    print(f"Number of dimensions: \n{encoded_labels.ndim}\n")
    print(f"Item size: \n{encoded_labels.itemsize}\n")
    print(f"Total bytes: \n{encoded_labels.nbytes}\n")
    print(f"Strides: \n{encoded_labels.strides}\n")
    print(f"Flags: \n{encoded_labels.flags}\n")
    print(f"Ctypes: \n{encoded_labels.ctypes}\n")
    print(f"Base: \n{encoded_labels.base}\n")
    print(f"Data: \n{encoded_labels.data}\n")
    print(f"Transpose: \n{encoded_labels.T}\n")
    print(f"Real part: \n{encoded_labels.real}\n")
    print(f"Imaginary part: \n{encoded_labels.imag}\n")
    print(f"Flat: \n{encoded_labels.flat}\n")
    print(f"Item: \n{encoded_labels.item}\n")
    print(f"List: \n{encoded_labels.tolist()}\n")
    print(f"Bytes: {encoded_labels.tobytes()}\n")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from _weakref import ReferenceType
import numpy as np
from datetime import datetime, date, timedelta, time
from typing import Any
from uuid import UUID
from src.utils.logger import getLogger

log = getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for serializing objects."""
    def default(self, obj: Any) -> Any:
        """Custom JSON encoder for serializing objects.

        :param obj: Any: the object to serialize
        :return: Any: the serialized object
        """
        try:
            # Handle Custom Classes serialization
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, date):
                return obj.isoformat()  # Call the method here
            if isinstance(obj, timedelta):
                return (datetime.min + obj).time()
            # Check if the object is of type 'ReferenceType'
            if isinstance(obj, ReferenceType):
                # Add your desired custom serialization logic here
                # For example, you could return only the referenced object's ID:
                return obj.__dict__  # Replace '_id' with the actual attribute containing the ID.
            if isinstance(obj, timedelta):
                # convert time delta to datetime
                return (datetime.min + obj).time()
            if isinstance(obj, time):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.float64):
                return float(obj)
            # Handle Built-in types serialization
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, bytes):
                return obj.decode()
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                return obj.__dict__
            if isinstance(obj, list):
                return obj
            if isinstance(obj, int):
                return obj
            if isinstance(obj, float):
                return obj
            if isinstance(obj, bool):
                return obj
            if isinstance(obj, type):
                return obj.__name__
            # Handle objects has attribute "<attribute>" serialization
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, '__dict__'):
                return dict(obj.__dict__)
            if hasattr(obj, 'body'):
                return obj.body
            if hasattr(obj, "__iter__"):
                return list(obj)
            if hasattr(obj, "__str__"):
                return str(obj)
            if hasattr(obj, "__repr__"):
                return repr(obj)
            if hasattr(obj, "int"):
                return int(obj)

            # Handle default serialization
            log.debug(f"Default: Type not handled - \n{type(obj)} \n{obj}")
            return super().default(obj)
        except Exception as e:
            return f"Error encoding object: {type(e)} \n{e}"
# --------------------------------------------------------------

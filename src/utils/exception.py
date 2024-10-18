#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This class is dedicated to custom exceptions for the project."""


class RetryException(Exception):
    """Exception raised when a retry limit is reached."""

    pass


class InvalidShapeException(Exception):
    """Exception raised when a shape mismatch occurs."""

    pass


class InvalidTypeException(Exception):
    """Exception raised when an invalid type is encountered."""

    pass


class InvalidInputException(Exception):
    """Exception raised when an invalid input is encountered."""

    pass


class InvalidValueException(Exception):
    """Exception raised when an invalid value is encountered."""

    pass


class InvalidDimensionException(Exception):
    """Exception raised when an invalid dimension is encountered."""

    pass

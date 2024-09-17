#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

from jupyter_server.auth import passwd
from typing_extensions import Union, Dict, Annotated

from src.utils.logger import getLogger

log = getLogger(__name__)
JSONString = Annotated[str, json.loads, json.dumps, "A JSON string"]
ConfigurationOptions = Annotated[
    dict, json.loads, json.dumps, "A dictionary of configuration options"
]
ConfigurationFile = Annotated[
    str,
    dict,
    JSONString,
    ConfigurationOptions,
    "Defaults to system config file, found in user home directory (~).",
]


def generate_password_hash(__passwd_str: str):
    """
    Generate a password hash for the Jupyter notebook server.

    :param __passwd_str: str: The password to hash
    :return: str: The hashed password
    """
    return passwd(__passwd_str)


def verify_password(__passwd_hash: str, __passwd_str: str):
    """
    Decode a password hash for the Jupyter notebook server.
    Tests the password string against the hash to verify if it is correct.

    :param __passwd_hash: str: The hashed password to decode
    :param __passwd_str: str: The password to verify against the hash
    :return: bool: True if the password is correct, False otherwise
    """
    from jupyter_server.auth import passwd_check

    return passwd_check(__passwd_hash, __passwd_str)


def write_password_hash_to_config(
    __passwd_hash: str,
    __config_file: Annotated[
        Union[str, JSONString],
        "Defaults to system config file, found in user home directory (~).",
    ] = "~/.jupyter/jupyter_notebook_config.json",
    config_options: ConfigurationOptions = None,
) -> None:
    """
    Write the password hash to the Jupyter notebook server configuration file.

    :param __passwd_hash: str: The hashed password to write to the configuration file
    :param __config_file: str: The configuration file to write the hashed password to
    :param config_options: ConfigurationOptions: Additional configuration options to write to the configuration file.
    This should be a dictionary of dictionaries, where the key is the section name and the value is a dictionary of
    configuration options for that section.
    :return: None
    """
    import pandas as pd
    import os

    # Check if configuration options are provided and merge them with the configuration file
    try:
        config_options = pd.DataFrame(config_options)
        if not config_options.empty:
            print(f"Configuration options provided: {config_options}")
            __config_file = pd.concat([__config_file, config_options], axis=1)
            print(f"Successfully merged configuration options: {__config_file}")
    except:
        raise ValueError("Invalid configuration options provided")

    # Get user home directory path
    __config_file_location = str(
        os.path.expanduser("~") + "/.jupyter/" + __config_file.split("/")[-1]
    )
    log.debug(f"Config file in User home directory: {__config_file_location}")
    __config_file = __config_file_location

    # Check if the configuration file exists
    if isinstance(__config_file, str) and os.path.exists(__config_file):
        __config_file = os.path.expanduser(__config_file)
        __config_file = pd.read_json(__config_file)
    elif isinstance(__config_file, dict) and __config_file.__len__() > 0:
        __config_file = pd.DataFrame(__config_file)
    else:
        raise ValueError("Invalid configuration file")
    log.debug(f"Configuration file: \n{__config_file.to_dict()}")
    __config_file["ServerApp"].add([{"hashed_password": __passwd_hash}])
    log.debug(f"Updated configuration file: \n{__config_file.to_dict()}")
    # Write the configuration file to the user home directory
    # __config_file.to_json(__config_file_location, orient="records")


if __name__ == "__main__":
    log.info("Starting the application")
    # Create a password hash
    new_passwd = generate_password_hash("password")
    log.info(f"New password hash: {new_passwd}")
    # Verify the password hash
    log.info(f"Password verification: {verify_password(new_passwd, 'password')}")
    # Write the password hash to the configuration file
    write_password_hash_to_config(new_passwd)

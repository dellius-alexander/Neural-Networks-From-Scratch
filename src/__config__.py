#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define the directory structure of this project. Even though
currently the project is small, it is a good practice to define
the directory structure at the beginning of the project. This
will help you to organize your code and assets in a structured
way. The directory structure is as follows:
# --------------------------------------------------------------
# root/
# ├── assets/
# │   ├── images/
# │   ├── docs/
# ├── datasets/
# │   ├── tesla/
# ├── logs/
# ├── models/
# ├── src/
# │   ├── __config__.py
# │   ├── functions/
# │   ├── encoder/
# │   ├── decoder/
# │   ├── repository/
# │   ├── main.py
# ├── notebooks/
# │   ├── encoder/
# │   ├── decoder/
# │   ├── repository/
# ├── tests/
# │   ├── test_encoder/
# │   ├── test_decoder/
# │   ├── test_repository/
# ├── .env
# ├── .gitignore
# ├── .dockerignore
# ├── .git/
# ├── .vscode/
# ├── .devcontainer/
# ├── .prodcontainer/
# ├── .github/
# ├── .circleci/
# ├── .gitlab-ci.yml
# ├── README.md
# ├── LICENSE
# ├── requirements.txt
# ├── setup.py
# ├── Dockerfile
# ├── docker-compose.yml
# ├── Jenkinsfile
# ├── Makefile
# ├── .travis.yml
# ├── CODE_OF_CONDUCT.md
# ├── CONTRIBUTING.md
# ├── ISSUE_TEMPLATE.md
# ├── PULL_REQUEST_TEMPLATE.md
# --------------------------------------------------------------
"""
# --------------------------------------------------------------
import time
import os
import json
from dotenv import load_dotenv, find_dotenv, dotenv_values
# --------------------------------------------------------------
# Define and Create project environment
# --------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__).split("src")[0])
print(f"ROOT_DIR: {ROOT_DIR}")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
print(f"LOG_DIR: {LOG_DIR}")
LOG_FILE = f'{LOG_DIR}/dev_{time.strftime("%Y%m%d%H%M%S")}.log'
print(f"LOG_FILE: {LOG_FILE}")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
print(f"DATASETS_DIR: {DATASETS_DIR}")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
print(f"MODELS_DIR: {MODELS_DIR}")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
print(f"LOG_LEVEL: {LOG_LEVEL}")
# --------------------------------------------------------------
# Create directories if they do not exist
for directory in [LOG_DIR, DATASETS_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True, mode=0o777)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")
# --------------------------------------------------------------
# Define log colors
log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}
# --------------------------------------------------------------
# Logging configuration
log_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "class": "logging.Formatter",
            "format": "[%(asctime)s][%(levelname)s][%(name)s][%(lineno)s]: \n%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
        },
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s[%(asctime)s][%(levelname)s][%(name)s][%(lineno)s]: "
            "\n%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": log_colors_config,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored",
        },
        "file_handler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": f"{LOG_FILE}",
            "when": "midnight",
            "interval": 1,
            "encoding": "utf-8",
            "delay": False,
            "utc": False,
            "level": "DEBUG",
            "formatter": "standard",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "file_handler"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}
# --------------------------------------------------------------
# Load environment variables from .env file
try:
    for filename in [".env"]:
        result = load_dotenv(
            find_dotenv(
                filename=filename,
                raise_error_if_not_found=True,
                usecwd=True,
            )
        )
        if result:
            print(f"Loaded environment variables from: {filename}")
            continue
        else:
            print(f"Environment variables not loaded from: {filename}")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    exit(1)
# --------------------------------------------------------------
# Set environment variables
# --------------------------------------------------------------
os.environ.setdefault("ROOT_DIR", ROOT_DIR)
os.environ.setdefault("LOG_FILE", LOG_FILE)
os.environ.setdefault("LOG_LEVEL", LOG_LEVEL)
os.environ.setdefault("LOG_DIR", LOG_DIR)
# --------------------------------------------------------------
print(f"Loaded Environment Variables: \n{
json.dumps(
    dotenv_values(), 
    indent=2, 
    sort_keys=True,
)}")
# --------------------------------------------------------------



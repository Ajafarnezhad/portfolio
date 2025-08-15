import logging
import typer
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("envsecure.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def prompt_password(prompt: str) -> str:
    return typer.prompt(prompt, hide_input=True, confirmation_prompt=True)

def validate_file_extension(filename: str, expected_ext: str) -> bool:
    return filename.endswith(expected_ext)
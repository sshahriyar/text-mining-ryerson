import os
from pathlib import Path
from getpass import getpass

from langchain_groq import ChatGroq


# src/config.py

import os
from pathlib import Path
from getpass import getpass

from langchain_groq import ChatGroq


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Make sure key folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)


def setup_api_key(env_var: str = "GROQ_API_KEY") -> str:
    """
    Ensure the Groq API key is available in the environment.
    Prompts the user only if the key is not already set.

    Args:
        env_var: Environment variable name for the API key.

    Returns:
        The API key string.
    """
    api_key = os.environ.get(env_var)

    if not api_key:
        api_key = getpass("Enter your Groq API key: ").strip()
        if not api_key:
            raise ValueError("Groq API key was not provided.")
        os.environ[env_var] = api_key

    return api_key


def get_basic_model(
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
) -> ChatGroq:
    """
    Return the basic Groq model used for baseline runs.
    """
    setup_api_key()
    return ChatGroq(
        model=model_name,
        temperature=temperature,
    )


def get_reasoning_model(
    model_name: str = "deepseek-r1-distill-llama-70b",
    temperature: float = 0.0,
) -> ChatGroq:
    """
    Return the reasoning-focused Groq model used for comparison runs.
    """
    setup_api_key()
    return ChatGroq(
        model=model_name,
        temperature=temperature,
    )
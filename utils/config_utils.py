"""
Configuration Utilities Module.

This module is responsible for loading and managing configuration files such as `models.yaml` 
and `params.yaml` located in the config directory. It provides specific helper functions to load 
different configuration topics, retrieve environment variables (like API keys), and validate the 
presence of these necessary API keys.

Overall, it serves as the central configuration management point for the Agentic RAG system, 
ensuring that components have easy access to the parameters and keys they need to function.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from the .env file located at the project root
load_dotenv()


def load_yaml_configuration(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the specified file path and returns it as a dictionary.

    Args:
        file_path (str): The absolute or relative path to the YAML file to be loaded.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML configuration. Returns an empty 
        dictionary if the file is not found or is empty.
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, "r", encoding="utf-8") as file:
        # yaml.safe_load parses the YAML file securely without executing arbitrary code
        configuration = yaml.safe_load(file)
        
    # Return an empty dictionary if the file was completely empty (which makes yaml.safe_load return None)
    return configuration if configuration is not None else {}


def get_model_configuration() -> Dict[str, Any]:
    """
    Retrieves the models configuration by loading the 'models.yaml' file.

    Args:
        None

    Returns:
        Dict[str, Any]: The configuration details related to the LLMs defined in 'models.yaml'.
    """
    # Construct the path to the models.yaml file assuming it's in the 'config' folder
    # one level up from the 'utils' folder.
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_file_path = os.path.join(base_directory, "config", "models.yaml")
    
    return load_yaml_configuration(models_file_path)


def get_parameter_configuration() -> Dict[str, Any]:
    """
    Retrieves the parameter configuration by loading the 'params.yaml' file.

    Args:
        None

    Returns:
        Dict[str, Any]: The general system parameters and configuration from 'params.yaml'.
    """
    # Construct the path to the params.yaml file
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parameters_file_path = os.path.join(base_directory, "config", "params.yaml")
    
    return load_yaml_configuration(parameters_file_path)


def get_api_key(service_name: str) -> Optional[str]:
    """
    Retrieves the API key for a specified service from the environment variables.

    Args:
        service_name (str): The name of the service (e.g., 'OPENAI', 'OPENROUTER', 'TAVILY'). 
                            The function will automatically append '_API_KEY' if not included.

    Returns:
        Optional[str]: The API key string if found, otherwise None.
    """
    # Standardize the environment variable name to be uppercase
    environment_variable_name = service_name.upper()
    
    # Check if the provided service name already ends with '_API_KEY'
    # If not, append it to match the standard environment variable naming convention
    if not environment_variable_name.endswith("_API_KEY"):
        environment_variable_name = f"{environment_variable_name}_API_KEY"
        
    return os.environ.get(environment_variable_name)


def is_api_key_valid(service_name: str) -> bool:
    """
    Checks whether a valid API key exists in the environment variables for a given service.

    Args:
        service_name (str): The name of the service to check (e.g., 'OPENAI').

    Returns:
        bool: True if the API key exists and is not an empty string, False otherwise.
    """
    api_key = get_api_key(service_name)
    
    # A valid key must not be None and must have a length greater than 0 after stripping whitespace
    if api_key is not None and len(api_key.strip()) > 0:
        return True
        
    return False

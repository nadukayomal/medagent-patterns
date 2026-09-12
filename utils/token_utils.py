"""
Token Utilities Module.

This module provides necessary functionalities to count and manage tokens for Large Language Models (LLMs).
It utilizes the 'tiktoken' library, which is standard for counting tokens for OpenAI models, to help 
estimate and monitor the number of tokens used in text strings and structured chat messages.

Overall, it helps in ensuring that the prompts do not exceed the context window limits of the models,
aids in estimating the cost of API calls, and supports the 'Reflection' and 'ReAct' design patterns 
by allowing the system to monitor its token budget.
"""

import tiktoken
from typing import List, Dict, Any


def get_token_counter(model_name: str = "gpt-3.5-turbo") -> tiktoken.Encoding:
    """
    Retrieves the appropriate tiktoken encoding (tokenizer) for a specific model.

    Args:
        model_name (str): The identifier of the model for which to retrieve the tokenizer.
                          Defaults to "gpt-3.5-turbo".

    Returns:
        tiktoken.Encoding: The tiktoken encoding object capable of encoding text to tokens.
    """
    try:
        # tiktoken provides a function to get the exact encoding used by a specific model
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If the specific model is not found in tiktoken's mapping, we fall back to 'cl100k_base'
        # which is the standard encoding used by most modern OpenAI models (GPT-3.5 and GPT-4)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens_in_text(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Calculates the number of tokens present in a simple string of text.

    Args:
        text (str): The text string to be tokenized and counted.
        model_name (str): The model name used to select the correct tokenizer. 
                          Defaults to "gpt-3.5-turbo".

    Returns:
        int: The total number of tokens in the text.
    """
    tokenizer = get_token_counter(model_name)
    # The encode method converts the text string into a list of integer token IDs
    encoded_tokens = tokenizer.encode(text)
    
    return len(encoded_tokens)


def count_tokens_in_messages(messages: List[Dict[str, Any]], model_name: str = "gpt-3.5-turbo") -> int:
    """
    Calculates the total number of tokens used in a list of chat messages.
    
    This function accounts for the special tokens and formatting overhead that chat-based LLMs 
    use to structure conversations (e.g., tokens indicating roles like 'system' or 'user').

    Args:
        messages (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents 
                                         a message (e.g., {"role": "user", "content": "hello"}).
        model_name (str): The model name used to select the correct tokenizer. 
                          Defaults to "gpt-3.5-turbo".

    Returns:
        int: The total estimated number of tokens required to process the messages.
    """
    tokenizer = get_token_counter(model_name)
    
    # Every message follows a specific format (e.g., <im_start>role\ncontent<im_end>\n)
    # This overhead typically costs about 3 tokens per message for modern models.
    tokens_per_message_overhead = 3
    
    # The entire conversation is usually bounded by a final <im_start>assistant to prime the model
    # which costs an additional 3 tokens overall.
    total_token_count = 3 
    
    for message in messages:
        total_token_count += tokens_per_message_overhead
        
        # We iterate over the keys in the message (e.g., 'role', 'content', 'name')
        for key, value in message.items():
            # Sometimes values might be None or not a string, we ensure we only encode strings
            if isinstance(value, str):
                total_token_count += len(tokenizer.encode(value))
                
            # If a 'name' key is provided, the role overhead changes slightly,
            # costing 1 additional token in standard OpenAI formatting
            if key == "name":
                total_token_count += 1
                
    return total_token_count

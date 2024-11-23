"""
utils for language models
"""
import ast
import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from ..utils import f_mkdir


def get_model_params(
        model_name,
        temperature=0,
        request_timeout=120,
        verbose=True,
        callbacks=None,
        max_retries=5
):
    """
    Constructs a dictionary of parameters for initializing a language model.

    This function prepares the necessary parameters for a language model, including handling specific configurations
    for Azure OpenAI services if applicable.

    Parameters:
        model_name (str): The name of the model to be used.
        temperature (float): The temperature setting for the model, controlling randomness.
        request_timeout (int): The timeout in seconds for the model request.
        verbose (bool): Flag to enable verbose logging.
        callbacks (list): A list of callback functions to be invoked during model interaction.
        max_retries (int): The maximum number of retries for a request.

    Returns:
        dict: A dictionary containing the prepared model parameters.
    """
    model_params = {
        'model_name': model_name,
        'temperature': temperature,
        'request_timeout': request_timeout,
        'verbose': verbose,
        'callbacks': callbacks,
    }

    if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        # AZURE_OPENAI_DEPLOYMENT_NAME is a string representing a dict
        deployment_names = ast.literal_eval(os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"])
        model_params['deployment_name'] = deployment_names[model_name]
    if max_retries:
        model_params['max_retries'] = max_retries

    return model_params


def get_chat_model():
    """
    Determines the chat model class based on the environment configuration.

    This function selects the appropriate chat model class to use based on whether the Azure OpenAI endpoint
    is configured in the environment variables.

    Returns:
        class: The chat model class, either AzureChatOpenAI or ChatOpenAI.
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI
    else:
        return ChatOpenAI


def get_embedding_fn(store_location="./lm_cache/embeddings"):
    """
    Determines the embedding function based on the environment configuration.

    This function selects the appropriate embedding function to use based on whether the Azure OpenAI endpoint
    is configured in the environment variables.

    Returns:
        function: The embedding function, either AzureOpenAIEmbeddings or OpenAIEmbeddings.
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        underlying_embeddings = AzureOpenAIEmbeddings()
    else:
        underlying_embeddings = OpenAIEmbeddings()
    
    f_mkdir(store_location)
    store = LocalFileStore(store_location)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model, query_embedding_cache=True
    )

    return cached_embedder


def construct_chat_model(
        model_name: str,
        temperature=0,
        request_timeout: int = 120,
        verbose: bool = True,
        callbacks=None,
        max_retries: int = 5
):
    """
    Constructs and initializes a chat model with the specified parameters.

    This function combines the functionality of get_model_params() and get_chat_model() to construct and
    initialize a chat model with the provided parameters.

    Parameters:
        model_name (str): The name of the model to be used.
        temperature (float): The temperature setting for the model, controlling randomness.
        request_timeout (int): The timeout in seconds for the model request.
        verbose (bool): Flag to enable verbose logging.
        callbacks (list): A list of callback functions to be invoked during model interaction.
        max_retries (int): The maximum number of retries for a request.

    Returns:
        object: An instance of the constructed chat model.
    """
    model_params = get_model_params(
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
        verbose=verbose,
        callbacks=callbacks,
        max_retries=max_retries,
    )
    chat_model = get_chat_model()
    constructed_chat_model = chat_model(**model_params)
    return constructed_chat_model

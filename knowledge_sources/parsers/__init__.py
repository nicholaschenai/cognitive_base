import yaml
import re

from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader


def parse_md_with_frontmatter(markdown_content):
    """
    Parses a markdown string to separate YAML frontmatter and the markdown content.

    Args:
        markdown_content (str): The markdown content as a string.

    Returns:
        tuple: A tuple containing a dictionary of metadata and the markdown content without frontmatter.
    """
    metadata = {}

    if markdown_content.startswith('---'):
        # Split the content by '---',  ensures that it only splits at the first two occurrences
        parts = markdown_content.split('---', 2)

        if len(parts) >= 3:
            # Extract the frontmatter and the markdown content
            frontmatter = parts[1].strip()
            markdown_content = parts[2].strip()

            # Parse the frontmatter as YAML
            metadata = yaml.safe_load(frontmatter)

    # If no frontmatter is detected, return empty metadata and the original content
    return metadata, markdown_content


def extract_md_folder(folder_path, **kwargs):
    loader = DirectoryLoader(
        folder_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()

    # split it into chunks
    text_splitter = MarkdownTextSplitter(**kwargs)
    docs = text_splitter.split_documents(documents)

    return docs


def extract_blocks(text, identifier='', concat=True):
    """
    Extracts code blocks from the given text using the specified identifier.

    Parameters:
        text (str): The text from which to extract code blocks.
        identifier (str): An optional identifier to specify the type of code blocks to extract.

    Returns:
        str: A string containing all extracted code blocks, concatenated and separated by newlines.
    """
    # Escape the identifier to avoid regex injection issues
    # (note: regex injection is only if identifier is supplied by external users)
    # identifier = re.escape(identifier)

    # Create the regex pattern dynamically
    if identifier:
        pattern = re.compile(rf"```(?:{identifier})(.*?)```", re.DOTALL)
    else:
        pattern = re.compile(r"```(.*?)```", re.DOTALL)

    pattern_list = pattern.findall(text)
    if concat:
        return "\n".join(pattern_list)
    return pattern_list

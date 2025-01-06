

import textwrap

tag_indent_template = """
[{tag}]:
{data}
[/{tag}]
"""


def tag_indent_format(tag, data_list, label=False, idx=1, label_name='Document'):
    """
    Formats a list of data items by wrapping each in XML-style tags and indenting the content.

    This function takes each item in the data list and:
    1. Wraps it in opening/closing tags using the provided tag name
    2. Optionally adds a label number to the tag (e.g. "Tag (Document 1)")
    3. Indents the content by 4 spaces
    4. Joins all formatted items together

    Args:
        tag (str): The tag name to wrap around each data item
        data_list (list): List of data items to format
        label (bool, optional): Whether to add numbered labels to tags. Defaults to False.
        idx (int, optional): Starting index for labels. Defaults to 1.
        label_name (str, optional): Label prefix to use. Defaults to 'Document'.

    Returns:
        str: The formatted string containing all data items wrapped in tags and indented

    Example:
        >>> tag_indent_format('Entry', ['a', 'b'], label=True)
        [Entry (Document 1)]:
            a
        [/Entry (Document 1)]
        [Entry (Document 2)]:
            b
        [/Entry (Document 2)]
    """
    final_str = ''
    for data in data_list:
        final_str += tag_indent_template.format(
            tag=tag + (f' ({label_name} {idx})' if label else ''),
            data=textwrap.indent(str(data), '    ')
        )
        idx += 1
    return final_str


def truncate_str(s, max_length=300):
    """
    Truncate a string to a maximum length, appending '...' if truncated.

    Parameters:
    s (str): The string to be truncated.
    max_length (int, optional): The maximum length of the truncated string including the ellipsis. Defaults to 300.

    Returns:
    str: The truncated string with '...' appended if it exceeds the maximum length.
    """
    if len(s) > max_length:
        return s[:max_length - 3] + '...'
    return s


def dict_indent_format(main_string, data):
    """
    Formats a template string by indenting dictionary values and substituting them into placeholders.

    This function takes a template string containing placeholders and a dictionary of values.
    It indents each value in the dictionary by 4 spaces and then formats the template string
    by substituting the indented values into their corresponding placeholders.

    Args:
        main_string (str): A template string containing placeholders in the format {key}.
        data (dict): A dictionary mapping placeholder keys to values that will be indented.

    Returns:
        str: The formatted string with indented values substituted into the placeholders.
    """
    indented_data = {k: textwrap.indent(str(v), '    ') for k, v in data.items()}
    return main_string.format(**indented_data)

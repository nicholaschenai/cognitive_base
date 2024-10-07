"""
Some implementations adapted from
- "Voyager: An Open-Ended Embodied Agent with Large Language Models" Wang et. al. (2023) https://github.com/MineDojo/Voyager
- ACR: https://github.com/nus-apr/auto-code-rover/
"""
import importlib
import random
import asyncio
import collections
import os
import time
import logging
import pdb
import traceback
import json

from pprint import pp
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

logger = logging.getLogger("logger")


def terminal_width():
    t_width = 120
    try:
        t_width = os.get_terminal_size().columns
    except:
        pass
    return t_width


WIDTH = min(120, terminal_width() - 10)

console = Console()


def print_panel(msg: str, title="", use_md=False) -> None:
    if use_md:
        msg = Markdown(msg)
    border_style, _ = random.choice(color_mapping)
    panel = Panel(msg, title=title, title_align="left", border_style=border_style, width=WIDTH)
    console.print(panel, markup=False)


def str_from_msg(msg, return_type):
    """
    Converts an AI message to a string based on the specified return type.

    Parameters:
    - msg (AIMessage or dict): The message to convert.
    - return_type (str): The type of return value. Can be 'ai_message' for direct content or 'pydantic_object' for a JSON representation.

    Returns:
    - str: The converted message as a string.
    """
    if return_type == 'ai_message':
        return msg.content
    elif return_type == 'pydantic_object':
        raw_ai_message = msg['raw']
        tool_calls = raw_ai_message.tool_calls
        dict_str = ''
        for tool_call in tool_calls:
            dict_str += json.dumps(tool_call['args'], indent=4)

        return dict_str


def get_ai_message_pretty_repr(ai_out, return_type):
    """
    Generates a pretty representation of an AI message based on the specified return type.

    Parameters:
    - ai_out (AIMessage or dict): The AI message to represent.
    - return_type (str): The type of return value. Determines the format of the output.

    Returns:
    - str: A pretty representation of the AI message.
    """
    if return_type == 'ai_message':
        ai_message = ai_out
    elif return_type == 'pydantic_object':
        ai_message = ai_out['raw']
    return ai_message.pretty_repr()


def print_messages(messages, name):
    for message in messages:
        # print(message.content)
        # message.pretty_print()
        # print_panel(message.pretty_repr(), title=name)
        title = name
        if hasattr(message, 'pretty_repr'):
            formatted_message = message.pretty_repr()
        else:
            formatted_message = message['content']
            title = f'{name} {message["role"]}'
        print_panel(formatted_message, title=title)


# TODO: deprecate this after checking sync wrapper works for async ver
def parse_retry_loop(
        messages,
        parse_fn,
        lm,
        parse_tries=3,
        fallback=None,
        debug_mode=False,
        parser=None,
        return_messages=False,
        verbose=False,
        return_type='ai_message',
        name='base_reasoning',
):
    """
    LM call, then parse messages. with retries upon failure.

    Parameters:
    - messages (list): List of messages to parse.
    - parse_fn (function): The parsing function to use.
    - lm (langchain chatmodel): The language model to use for parsing.
    - parse_tries (int): Number of attempts to try parsing.
    - fallback: The fallback value if parsing fails.
    - debug_mode (bool): If True, enters debug mode.
    - parser: The parser to use, if any.
    - return_messages (bool): If True, returns the messages along with the parsed result.
    - verbose (bool): If True, prints verbose output.
    - return_type (str): The type of return value for message conversion.

    Returns:
    - dict or tuple: The parsed result, optionally with messages.
    """
    parsed_result = fallback if (fallback is not None) else {}
    parse_success = False

    for i in range(parse_tries):
        if verbose:
            print_messages(messages, f'{name} prompt')

        logger.info(f'LM call n parse attempt {i + 1} / {parse_tries}\n')
        try:
            ai_message = lm.invoke(messages)

            msg = ai_message if return_type == 'ai_message' else AIMessage(
                content=str_from_msg(ai_message, return_type))
            messages.append(msg)
        except Exception as e:
            error_msg = f"Error during LM call! Retrying. Error msg:\n{str(e)}, {type(e).__name__}\n"
            logger.warning(error_msg)
            logger.warning(traceback.format_exc())
            time.sleep(5)
            continue
        if verbose or debug_mode:
            # print full thing incase error msg
            pp(ai_message)
            # print(get_ai_message_pretty_repr(ai_message, return_type))
            # print(str_from_msg(ai_message, return_type))
            print_panel(str_from_msg(ai_message, return_type), title=f'{name} response')
        if debug_mode:
            custom_breakpoint()
        try:
            if parser:
                parsed_result = parse_fn(ai_message, parser=parser)
            else:
                parsed_result = parse_fn(ai_message)
            parse_success = True
            break
        except Exception as e:
            error_msg = f"Error during parsing! {str(e)}, {type(e).__name__}\n"
            logger.warning(error_msg)
            messages.append(HumanMessage(content=error_msg))
    if not parse_success:
        logger.error(f'All parse attempts failed')
    if return_messages:
        return {'parsed_result': parsed_result, 'messages': messages}
    return parsed_result


def custom_breakpoint(use_input=True):
    if use_input:
        confirmed = False
        cont = None
        while not confirmed:
            cont = input("continue? (y/n)")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        if cont == 'n':
            exit()
    else:
        pdb.set_trace()


def f_mkdir(*fpaths):
    """
    Recursively creates directories for the given file paths. Does nothing if they already exist.

    Parameters:
    - *fpaths: Variable length argument list of file paths to create directories for.

    Returns:
    - str: The final path created.
    """
    fpath = f_join(*fpaths)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def f_join(*fpaths):
    """
    Joins given file paths and expands special symbols like `~` for the home directory.

    Parameters:
    - *fpaths: Variable length argument list of file paths to join.

    Returns:
    - str: The joined and expanded file path.
    """
    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    Parameters:
    - args (tuple): The arguments to pack.

    Returns:
    - list: The packed arguments as a list.

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def f_expand(fpath):
    """
    Expands environment variables and user symbols in a file path.

    Parameters:
    - fpath (str): The file path to expand.

    Returns:
    - str: The expanded file path.
    """
    return os.path.expandvars(os.path.expanduser(fpath))


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


async def async_parse_retry_loop(
        messages,
        parse_fn,
        lm,
        parse_tries=3,
        fallback=None,
        debug_mode=False,
        parser=None,
        return_messages=False,
        verbose=False,
        return_type='ai_message',
        name='base_reasoning',
):
    """
    async version of parse_retry_loop
    """
    parsed_result = fallback if (fallback is not None) else {}
    parse_success = False

    for i in range(parse_tries):
        logger.info(f'LM call n parse attempt {i + 1} / {parse_tries}\n')
        try:
            ai_message = await lm.ainvoke(messages)

            msg = ai_message if return_type == 'ai_message' else AIMessage(
                content=str_from_msg(ai_message, return_type))
            messages.append(msg)
        except Exception as e:
            error_msg = f"Error during LM call! Retrying. Error msg:\n{str(e)}, {type(e).__name__}\n"
            logger.warning(error_msg)
            logger.warning(traceback.format_exc())
            await asyncio.sleep(5)
            continue
        if verbose or debug_mode:
            # print full thing incase error msg
            pp(ai_message)
            # print(get_ai_message_pretty_repr(ai_message, return_type))
            print(str_from_msg(ai_message, return_type))
        if debug_mode:
            await asyncio.to_thread(custom_breakpoint)
        try:
            if parser:
                parsed_result = parse_fn(ai_message, parser=parser)
            else:
                parsed_result = parse_fn(ai_message)
            parse_success = True
            break
        except Exception as e:
            error_msg = f"Error during parsing! {str(e)}, {type(e).__name__}\n"
            logger.warning(error_msg)
            messages.append(HumanMessage(content=error_msg))
    if not parse_success:
        logger.error(f'All parse attempts failed')
    if return_messages:
        return {'parsed_result': parsed_result, 'messages': messages}
    return parsed_result


def parse_retry_loop_sync(
        messages,
        parse_fn,
        lm,
        parse_tries=3,
        fallback=None,
        debug_mode=False,
        parser=None,
):
    """
    Synchronous wrapper for the asynchronous parse retry loop.

    :param messages:
    :param parse_fn:
    :param lm:
    :param parse_tries:
    :param fallback:
    :param debug_mode:
    :param parser:
    :return:
    """
    return asyncio.run(async_parse_retry_loop(messages, parse_fn, lm, parse_tries, fallback, debug_mode, parser))


def generic_parse_fn(message, parser):
    """
    Generic parsing function that invokes a parser on a message.

    Parameters:
    - message: The message to parse.
    - parser: The parser to use.

    Returns:
    - dict: The parsed message as a dictionary.
    """
    try:
        res = parser.invoke(message)
        return res.dict(exclude_unset=True, exclude_none=True)
    except Exception as e:
        error_msg = (f"Error when parsing output! Please check the formatting and try again. "
                     f"The error message is:\n  {str(e)}, {type(e).__name__}\n")
        raise Exception(error_msg)


def pydantic_parse_fn(message, parser):
    """
    If Pydantic model was used, llm.with_structured_output was called and we extract the parsed response.

    Parameters:
    - message: The with_structured_output returned by the language model.
    - parser: langchain parser. only for format instructions

    Returns:
    - dict: The parsed message as a dictionary.
    """
    try:
        parsing_error = message['parsing_error']
        if parsing_error is None:
            return message['parsed'].dict(exclude_unset=True, exclude_none=True)
        raise parsing_error
    except Exception as e:
        error_msg = (
            f"Error when parsing output! Please check the formatting and try again. "
            f"The error message is:\n  {str(e)}, {type(e).__name__}\n"
            f"The expected format is: \n {parser.get_format_instructions()}"
        )
        raise Exception(error_msg)


def dump_json(data, *file_path, **kwargs):
    """
    Dumps data to a JSON file at the specified path.

    Parameters:
    - data: The data to dump.
    - *file_path: The file path to dump the data to.
    - **kwargs: Additional keyword arguments for json.dump.
    """
    file_path = f_join(file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def load_json(*file_path, **kwargs):
    """
    Loads data from a JSON file at the specified path.

    Parameters:
    - *file_path: The file path to load the data from.
    - **kwargs: Additional keyword arguments for json.load.

    Returns:
    - The loaded data.
    """
    file_path = f_join(file_path)
    if not os.path.exists(file_path):
        print(f'{file_path} does  not exist, returning blank dict')
        return {}
    with open(file_path, "r") as fp:
        return json.load(fp, **kwargs)


def truncate_str(s, max_length=300):
    if len(s) > max_length:
        return s[:max_length - 3] + '...'
    return s


color_mapping = [
    ('magenta', '\033[95m'),
    ('cyan', '\033[96m'),
    ('green', '\033[32m'),
    ('yellow', '\033[33m'),
    ('blue', '\033[34m'),
    ('purple', '\033[35m'),
    ('red', '\033[31m')
]


def dump_text(s, *fpaths):
    """
    Write a string to a file specified by a series of file paths joined together.

    Args:
        s (str): The string to write to the file.
        *fpaths: Variable length argument list representing parts of the file path to be joined.
    """
    with open(f_join(*fpaths), "w", encoding='utf-8') as fp:
        fp.write(s)


def lm_cache_init(folder, filename="lm_cache.db"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=os.path.join(folder, filename)))


def get_cls(component):
    """
    Dynamically loads a class from a given module.

    This function takes a dictionary containing the 'module' and 'class' keys,
    dynamically imports the specified module, and returns the class specified
    in the 'class' key.

    Parameters:
    - component (dict): A dictionary with 'module' and 'class' keys, where
      'module' is the name of the module to import and 'class' is the name of
      the class to retrieve.

    Returns:
    - cls: The class object specified by the 'class' key in the component.
    """
    module_name = component['module']
    class_name = component['class']
    module = importlib.import_module(module_name)
    print(f'getting and building cls {class_name} from {module_name}')
    cls = getattr(module, class_name)
    return cls

import logging
import asyncio
import traceback
import time

from pprint import pp
from langchain.schema import HumanMessage, AIMessage

from . import custom_breakpoint, print_messages, print_panel, str_from_msg

logger = logging.getLogger("logger")

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
            break
        except Exception as e:
            error_msg = f"Error during parsing! {str(e)}, {type(e).__name__}\n"
            logger.warning(error_msg)
            messages.append(HumanMessage(content=error_msg))
    else:
        logger.error(f'All parse attempts failed')

    if return_messages:
        return {'parsed_result': parsed_result, 'messages': messages}
    return parsed_result

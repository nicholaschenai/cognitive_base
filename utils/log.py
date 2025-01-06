import shutil
import logging
import time
import random
import os
import logging
import traceback

from pathlib import Path

from . import dump_json, lm_cache_init

logger = logging.getLogger("logger")

def setup_extra_file_handler(logger, extra_log_file_name):
    """
    Adds an extra file handler to the provided logger.
    This is for per-task logging

    This function creates a new file handler that writes log messages to the specified
    file with UTF-8 encoding. The log level for this handler is set to INFO, and the
    log messages are formatted to include the log level and the message.

    Args:
        logger (logging.Logger): The logger to which the file handler will be added.
        extra_log_file_name (str): The name of the file where log messages will be written.

    Returns:
        None
    """
    extra_file_handler = logging.FileHandler(extra_log_file_name, encoding='utf-8')
    extra_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    extra_file_handler.setFormatter(formatter)
    logger.addHandler(extra_file_handler)


def setup_logging_n_base_dirs(args):
    log_folder = args.log_folder
    if not Path(args.result_dir).exists():
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    if not Path(log_folder).exists():
        Path(log_folder).mkdir(parents=True, exist_ok=True)

    random_int = random.randint(0, 10000)
    log_file_name = f"{log_folder}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random_int}.log"

    # so that we can save per task
    extra_log_file_name = f"{args.result_dir}/extra_log.log"
    # if this file exists, delete it via os n warn
    if os.path.isfile(extra_log_file_name):
        print("remove extra_log as it exists")
        os.remove(extra_log_file_name)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    setup_extra_file_handler(logger, extra_log_file_name)

    logger.info(f'[result dir] {args.result_dir}')
    with open(os.path.join(args.result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{log_file_name}\n")

    # LM caching
    lm_cache_init(args.lm_cache_dir)


def train_ckpt(agent):
    data = {attr: getattr(agent, attr) for attr in getattr(agent, 'attr_to_save')}
    args = getattr(agent, 'args')
    dump_json(data, f"{args['result_dir']}/train_ckpt_info.json", indent=4)
    train_iter = getattr(agent, 'train_iter')
    if not train_iter % args['save_every']:
        shutil.copytree(args['ckpt_dir'], f"{args['result_dir']}/saved_train_ckpt/{train_iter}", dirs_exist_ok=True)


def move_log_file(new_log_file_name, extra_log_folder):
    """
    Moves the existing log file to a new location and updates the logger to use the new log file. for the extra log which is for per-task logging
    Args:
        new_log_file_name (str): The new file name for the log file.
        extra_log_folder (str): The folder where the extra log file is currently located.
    Raises:
        FileNotFoundError: If the extra log file does not exist.
        PermissionError: If the file cannot be moved due to permission issues.
    """
    extra_log_file_name = f"{extra_log_folder}/extra_log.log"
    abs_name = os.path.abspath(extra_log_file_name)

    # Get the logger
    logger = logging.getLogger("logger")

    # Find and remove the existing extra file handler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == abs_name:
            handler.close()
            logger.removeHandler(handler)
            break

    os.rename(extra_log_file_name, new_log_file_name)

    # Add the new extra file handler
    setup_extra_file_handler(logger, extra_log_file_name)


def construct_task_folder(result_dir, mode, task_id):
    """
    Constructs a task folder path and creates the directory if it does not exist.

    Args:
        result_dir (str): The base directory where the task folder will be created.
        mode (str): The mode or type of task (e.g., 'train', 'test') which will be part of the folder name.
        task_id (str): The unique identifier for the task. Any '/' characters in the task_id will be replaced with '_'.

    Returns:
        str: The path to the created task folder.
    """
    task_folder = f"{result_dir}/{mode}_outputs/{str(task_id).replace('/', '_')}"
    if not Path(task_folder).exists():
        Path(task_folder).mkdir(parents=True, exist_ok=True)
    return task_folder


# old

def setup_logging(log_folder="log_files"):
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    random_int = random.randint(0, 10000)
    log_file_name = f"{log_folder}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random_int}.log"
    # so that we can save per task
    extra_log_file_name = f"{log_folder}/extra_log.log"
    # if this file exists, delete it via os n warn
    if os.path.isfile(extra_log_file_name):
        print("remove extra_log as it exists")
        os.remove(extra_log_file_name)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    extra_file_handler = logging.FileHandler(extra_log_file_name, encoding='utf-8')
    extra_file_handler.setLevel(logging.INFO)
    extra_file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(extra_file_handler)
    logger.addHandler(console_handler)
    return log_file_name


def setup_base_dirs(args, log_file_name):
    logger = logging.getLogger("logger")
    if not Path(args.result_dir).exists():
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'[result dir] {args.result_dir}')
    with open(os.path.join(args.result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{log_file_name}\n")
    # LM caching
    lm_cache_init(args.lm_cache_dir)


def setup_logging_n_base_dirs_old(args):
    log_file_name = setup_logging(args.log_folder)
    setup_base_dirs(args, log_file_name)


def handle_rollout_error(e, task_id, result_dir):
    """
    Handles errors that occur during a rollout process by logging the error and writing it to a file.

    Args:
        e (Exception): The exception that was raised.
        task_id (str): The identifier of the task during which the error occurred.
        result_dir (str or Path): The directory where the error log file should be saved.

    Logs:
        Logs the error message and traceback using the logger.

    Writes:
        Writes the error message and traceback to a file named "error.txt" in the specified result directory.
    """
    unhandled_error_str = f"[task_id]: {task_id} [Unhandled Error] {repr(e)}\n"
    error_trace = traceback.format_exc()
    logger.error(f"error in rollout.\n{unhandled_error_str}\n{error_trace}")
    with open(Path(result_dir) / "error.txt", "a") as f:
        f.write(unhandled_error_str)
        f.write(error_trace)

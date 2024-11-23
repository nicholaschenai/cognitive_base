import argparse
import time

default_result_dir = f"results/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"


def get_base_parser():
    parser = argparse.ArgumentParser(description="Run experiment")

    # dataset
    parser.add_argument("--batch_size", type=int, default=1)

    # saving / checkpointing
    parser.add_argument("--result_dir", type=str, default=default_result_dir, help='Directory to store results')
    parser.add_argument("--lm_cache_dir", type=str, default="lm_cache", help="directory to store LM cache")
    parser.add_argument("--log_folder", type=str, default="log_files", help="directory to store log files")
    
    # LM params
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0613")

    # LM misc
    parser.add_argument("--request_timeout", type=int, default=300)

    # debug
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")

    return parser
import shutil
from typing import Dict

from ....utils import dump_json
from ....utils.log import construct_task_folder


class TaskLogger:
    def __init__(self, result_dir: str, train: bool, task_id: str):
        self.task_folder = construct_task_folder(result_dir, 'train' if train else 'test', task_id)
        self.last_iteration_file = None

    def log_iteration(self, iteration_data: Dict, iteration: int):
        parsed_result = iteration_data.pop('parsed_result', {})
        iteration_data.update(parsed_result)
        self.last_iteration_file = f"{self.task_folder}/output_{iteration}.json"
        dump_json(iteration_data, self.last_iteration_file, indent=4)

    def log_rollout(self, rollout_data: Dict):
        dump_json(rollout_data['messages'], f"{self.task_folder}/messages.json", indent=4)
        # Copy last iteration file to output.json if it exists for backwards compatibility
        if self.last_iteration_file:
            shutil.copy2(self.last_iteration_file, f"{self.task_folder}/output.json")

    def log_train(self, training_data: Dict):
        messages = training_data.pop('messages', [])
        dump_json(messages, f"{self.task_folder}/messages.json", indent=4)
        dump_json(training_data, f"{self.task_folder}/training_data.json", indent=4)

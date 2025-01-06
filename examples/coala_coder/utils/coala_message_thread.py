import copy
import json

from typing import Any, Dict, List, Optional, Tuple

from ....utils.msg_tools import MessageThread
from ....utils.formatting import tag_indent_format


class CoalaMessageThread(MessageThread):
    def __init__(self, messages: List[Dict[str, Any]] = None):
        super().__init__(messages)

        self._temp_messages = []
        
        # Track rollout data
        self.rollout_data = {
            'messages': [],  # Store all messages in order
        }
        # Track training data
        self.training_data = {
            'summary': None,
            'reflection': None,
            'desc': None,
        }
        # Transition container
        self._current_transition = {
            'critique': None,
            'raw_msg': None,
            'obs': None,
            'reward': None
        }
        # Iteration data container
        self._current_iteration = {
            'env_feedback': None,
            'state': None,
            'code': None,
            'full_code': None,
            'parsed_result': None,
            'reward': None,
        }

    def add(self, role: str, message: str, is_canonical: bool = True, is_temp: bool = False):
        """Add a message to the thread.
        
        Args:
            role (str): The role of the message sender
            message (str): The content of the message
            is_canonical (bool): Whether this message should be part of the canonical thread
            is_temp (bool): Whether this is a temporary message for the current LM call
        """
        assert role in ["system", "user", "assistant"]
        msg = {"role": role, "content": message}
        
        self.rollout_data['messages'].append(msg)

        if is_canonical and not is_temp:
            self.messages.append(msg)
        if is_temp:
            self._temp_messages.append(msg)

    def to_msg(self) -> List[Dict]:
        """Get canonical messages"""
        return copy.deepcopy(self.messages)

    def to_msg_with_temp(self) -> List[Dict]:
        """Get all messages including temporary ones for current LM call"""
        msgs = self.to_msg()
        msgs.extend(copy.deepcopy(self._temp_messages))
        self._temp_messages = []  # Clear temp messages after use
        return msgs

    """
    Transition recording methods
    """
    def record_critique(self, critique: str):
        """Record critique"""
        self._current_transition['critique'] = critique
        self.add("assistant", critique)

    def record_raw_msg(self, parsed_result: Dict = None):
        """Record raw message and optionally parsed result"""
        if parsed_result:
            self._current_iteration['parsed_result'] = parsed_result

            raw_msg = parsed_result.get("raw_msg", "")
            self._current_transition['raw_msg'] = raw_msg
            self.add("assistant", raw_msg)

            self._current_iteration['code'] = parsed_result.get("program_code", "")

    def update_full_code(self, full_code: str = None):
        """Update code information for current iteration"""
        self._current_iteration['full_code'] = full_code

    def record_env_out(self, obs: str, reward: bool, info: Dict):
        """Record environment output"""
        # Record in transition
        self._current_transition.update({
            'obs': obs,
            'reward': reward
        })
        
        # Record in iteration data
        self._current_iteration.update({
            'env_feedback': obs,
            'state': info.get('individial_results', None),
            'reward': reward
        })
        
        # Add message
        self.add("user", tag_indent_format("Environment Feedback", [obs]))

    """
    transition and iteration data retrieval methods
    """
    def get_latest_transition(self) -> Optional[Dict]:
        """Get and clear current transition. Partial transitions are allowed."""
        if not any(v is not None for v in self._current_transition.values()):
            return None
        
        transition = {k: v for k, v in self._current_transition.items() if v is not None}
        self._current_transition = {k: None for k in self._current_transition}
        return transition

    def get_iteration_data(self) -> Optional[Dict]:
        """Get and clear current iteration data"""
        if not any(v is not None for v in self._current_iteration.values()):
            return None
            
        iteration = {k: v for k, v in self._current_iteration.items() if v is not None}
        self._current_iteration = {k: None for k in self._current_iteration}
        return iteration

    def get_latest_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get and clear both transition and iteration data"""
        transition = self.get_latest_transition()
        iteration = self.get_iteration_data()
        return transition, iteration
    
    def get_rollout_data(self) -> Dict:
        """Get complete rollout data (messages and full code history)"""
        return copy.deepcopy(self.rollout_data)

    """
    Training data recording methods
    """
    def record_summary(self, summary: str):
        """Record summary"""
        self.add("assistant", summary, is_canonical=False)
        self.training_data['summary'] = summary

    def record_reflection(self, reflection: str):
        """Record reflection"""
        self.add("assistant", reflection, is_canonical=False)
        self.training_data['reflection'] = reflection

    def record_desc(self, desc: str):
        """Record description"""
        self.add("assistant", desc, is_canonical=False)
        self.training_data['desc'] = desc

    """
    Training data retrieval methods
    """
    def get_training_data(self) -> Dict:
        """Get complete training data"""
        training_data = copy.deepcopy(self.training_data)
        training_data['messages'] = copy.deepcopy(self.rollout_data['messages'])
        return training_data

    """
    File saving and loading methods
    """
    def save_to_file(self, file_path: str):
        """Save thread state including rollout and training data"""
        data = {
            'messages': self.messages,
            'rollout_data': self.rollout_data,
            'training_data': self.training_data
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, file_path: str):
        """Load thread state"""
        with open(file_path) as f:
            data = json.load(f)
        thread = cls(data['messages'])
        thread.rollout_data = data['rollout_data']
        thread.training_data = data['training_data']
        return thread

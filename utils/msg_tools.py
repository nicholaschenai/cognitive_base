import json
import copy
from typing import Dict, List, Optional


class MessageThread:
    """
    Adapted from AutoCodeRover
    Represents a thread of conversation with the model.
    Abstrated into a class so that we can dump this to a file at any point.
    """

    def __init__(self, messages=None):
        self.messages: list[dict] = messages or []

    def add(self, role: str, message: str):
        """
        Add a new message to the thread.
        Args:
            message (str): The content of the new message.
            role (str): The role of the new message.
        """
        assert role in ["system", "user", "assistant"]
        self.messages.append({"role": role, "content": message})

    def to_msg(self) -> List[Dict]:
        """
        Convert to the format to be consumed by the model.
        Returns:
            List[Dict]: The message thread.
        """
        return copy.deepcopy(self.messages)

    def save_to_file(self, file_path: str):
        """
        Save the current state of the message thread to a file.
        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w") as f:
            json.dump(self.messages, f, indent=4)

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load the message thread from a file.
        Args:
            file_path (str): The path to the file.
        Returns:
            MessageThread: The message thread.
        """
        with open(file_path) as f:
            messages = json.load(f)
        return cls(messages)

    def subset(
        self, 
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        include_system: bool = True,
    ) -> 'MessageThread':
        """Create a new thread with a subset of messages.
        
        Args:
            start_idx: Starting index for subset (inclusive)
            end_idx: Ending index for subset (exclusive)
            include_system: Always include system messages regardless of other filters
        """
        messages = self.messages[start_idx:end_idx] if start_idx is not None or end_idx is not None else self.messages[:]
        
        if include_system and messages[0]['role'] != 'system':
            messages.insert(0, self.messages[0])
            
        return MessageThread(messages=messages)

    def last_n(self, n: int, include_system: bool = True) -> 'MessageThread':
        """Get the last n messages in the thread."""
        return self.subset(start_idx=-n, include_system=include_system)
    
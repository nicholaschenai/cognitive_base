"""
standardize some methods esp count
"""


class BaseDB:
    def count(self):
        raise NotImplementedError("Subclasses should implement this method")

"""
interface to standardize some methods esp count
"""


class DatabaseWrapper:
    def __init__(self, db, count_method):
        self._db = db
        self._count_method = count_method

    def __getattr__(self, name):
        # Delegate attribute access to the underlying database object
        return getattr(self._db, name)

    def count(self):
        return self._count_method(self._db)

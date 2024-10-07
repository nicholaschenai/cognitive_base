class BaseUpdate:
    def __init__(self, db, transform=None):
        self.db = db
        self.transform = transform

    def update(self, query, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

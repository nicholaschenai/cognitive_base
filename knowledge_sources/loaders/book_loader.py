import os

from tqdm import tqdm

from cognitive_base.utils import load_json

class BookSource:
    """Represents a source of book-like content"""
    def __init__(self, filepath: str, transform_fn, base_folder: str, **kwargs):
        self.filepath = filepath
        self.transform_fn = transform_fn
        self.base_folder = base_folder
        self.kwargs = kwargs

    def load_entries(self):
        return load_json(os.path.join(self.base_folder, self.filepath))
    
    def transform_content(self, book_index: int, entry):
        return self.transform_fn(book_index, entry)

class BookLoader:
    """Handles loading book-format content into agent memory"""
    def __init__(self, sources: list[BookSource], debug_mode=False, debug_subset=None):
        self.sources = sources
        self.debug_mode = debug_mode
        self.debug_subset = debug_subset

    def load_into_agent(self, agent):
        for source in self.sources:
            entries = source.load_entries()
            
            for book_index, entry in tqdm(enumerate(entries), desc="Loading book entries"):
                texts, metadatas = source.transform_content(book_index, entry)
                
                for text, metadata in zip(texts, metadatas):
                    agent.semantic_mem.update(text, metadata=metadata)
                
                if self.debug_mode and book_index == self.debug_subset:
                    break
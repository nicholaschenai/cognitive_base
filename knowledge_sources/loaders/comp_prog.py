import os
from pathlib import Path
from ..transforms import transform_handbook_content, transform_bookv2_content
from .book_loader import BookSource, BookLoader

def create_comp_prog_loader(debug_mode=False, debug_subset=None, **kwargs):
    """
    Creates a loader for competitive programming manuals.
    Curated by https://github.com/princeton-nlp/USACO, 
    which original source is from https://cp-algorithms.com/ and https://github.com/pllk/cphb
    
    Args:
        base_folder: Optional override for data location. If None, uses default data path
        debug_mode: Whether to run in debug mode
        debug_subset: Subset of data to use in debug mode
        
    Returns:
        BookLoader instance configured for competitive programming content
    """

    # Get the path to the data directory relative to this file
    current_dir = Path(__file__).parent.resolve()
    # Navigate to cognitive_base/examples/data
    base_folder = os.path.join(current_dir.parent.parent, 'examples', 'data')
    print(f'base_folder: {base_folder}')
    
    sources = [
        BookSource(
            filepath='cp_handbook.json',
            transform_fn=transform_handbook_content,
            base_folder=base_folder
        ),
        BookSource(
            filepath='cpbook_v2.json',
            transform_fn=transform_bookv2_content,
            base_folder=base_folder
        )
    ]
    
    return BookLoader(sources, debug_mode, debug_subset)

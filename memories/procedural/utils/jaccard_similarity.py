from .value_normalizer import normalize_value


def jaccard_similarity_dict(dict1, dict2):
    """
    Calculate the Jaccard similarity between two dictionaries by comparing their values.
    """
    set1 = set((k, frozenset(normalize_value(v))) for k, v in dict1.items())
    set2 = set((k, frozenset(normalize_value(v))) for k, v in dict2.items())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

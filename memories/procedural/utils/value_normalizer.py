def normalize_value(value):
    """
    Normalize the given value by converting it to a set. to be used with jaccard scoring strategy for rules in dict format.

    Args:
        value: The value to be normalized.

    Returns:
        A set containing the normalized value.

    Raises:
        ValueError: If the value type is not supported.
    """
    if isinstance(value, str):
        return {value}
    elif isinstance(value, list):
        return set(value)
    else:
        raise ValueError("Unsupported value type")
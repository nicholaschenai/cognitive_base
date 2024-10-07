def match_rigid_conditions(cue_conditions, rule_conditions):
    """
    Check if the cue conditions match the rule conditions.

    Args:
        cue_conditions (dict): The cue conditions to be checked.
        rule_conditions (dict): The rule conditions to match against.

    Returns:
        bool: True if the cue conditions match the rule conditions, False otherwise.
    """
    for key, value in rule_conditions.items():
        if key in cue_conditions:
            if isinstance(value, str) and value != cue_conditions[key]:
                return False
            elif isinstance(value, list) and not any(v in cue_conditions[key] for v in value):
                return False
        else:
            return False
    return True

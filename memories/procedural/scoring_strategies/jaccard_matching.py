from ..utils.jaccard_similarity import jaccard_similarity_dict


def jaccard_match_score(rule, cue):
    """
    Calculates the Jaccard similarity between a rule and a cue.

    Parameters:
    rule (dict): The rule to compare.
    cue (dict): The cue to compare.

    Returns:
    float: The Jaccard match score between the rule and cue.
    """
    rigid_conditions_rule = rule.get('rigid_conditions', {})
    rigid_conditions_cue = cue.get('rigid_conditions', {})
    return jaccard_similarity_dict(rigid_conditions_rule, rigid_conditions_cue)

# Contextual Weighting
def weighted_match_score(rule, problem_conditions):
    score = 0
    for cond, weight in zip(rule.conditions, rule.weights):
        if cond in problem_conditions:
            score += weight
    return score / sum(rule.weights)

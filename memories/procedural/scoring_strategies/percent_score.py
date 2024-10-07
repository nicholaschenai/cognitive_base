def percent_match_score(rule, problem_conditions):
    return sum(cond in problem_conditions for cond in rule.conditions) / len(rule.conditions)

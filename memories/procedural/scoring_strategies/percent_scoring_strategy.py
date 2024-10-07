from .scoring_strategy import ScoringStrategy
from .percent_score import percent_match_score


class PercentScoringStrategy(ScoringStrategy):
    def calculate_score(self, rule, problem_conditions):
        return percent_match_score(rule, problem_conditions)

from .scoring_strategy import ScoringStrategy
from .weighted_score import weighted_match_score


class FuzzyScoringStrategy(ScoringStrategy):
    def calculate_score(self, rule, problem_conditions):
        return weighted_match_score(rule, problem_conditions)

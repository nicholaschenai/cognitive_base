from .scoring_strategy import ScoringStrategy
from .hybrid_score import hybrid_match_score


class HybridScoringStrategy(ScoringStrategy):
    def __init__(self, weight_fuzzy=0.5, weight_embedding=0.5):
        self.weight_fuzzy = weight_fuzzy
        self.weight_embedding = weight_embedding

    def calculate_score(self, rule, problem_conditions):
        return hybrid_match_score(rule.conditions, problem_conditions, self.weight_fuzzy, self.weight_embedding)

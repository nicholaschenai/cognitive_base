from .scoring_strategy import ScoringStrategy
from .embedding_score import embedding_match_score


class EmbeddingScoringStrategy(ScoringStrategy):
    def calculate_score(self, rule, problem_conditions):
        return embedding_match_score(rule, problem_conditions)

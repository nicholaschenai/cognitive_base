from .scoring_strategy import ScoringStrategy
from .jaccard_matching import jaccard_match_score


class JaccardScoringStrategy(ScoringStrategy):
    """
    Implements a scoring strategy based on the Jaccard similarity coefficient.

    This strategy calculates the similarity score between a given rule and cue by
    leveraging the Jaccard match score, which measures the similarity between the
    sets of conditions in the rule and the cue. The score is a float value between
    0 and 1, where 1 indicates perfect similarity (i.e., the rule and cue are identical),
    and 0 indicates no similarity.

    Attributes:
        None

    Methods:
        calculate_score(rule, cue): Calculates the Jaccard similarity score between
                                    the given rule and cue.
    """
    def calculate_score(self, rule, cue):
        return jaccard_match_score(rule, cue)

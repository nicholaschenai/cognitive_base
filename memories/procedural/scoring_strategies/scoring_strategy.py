from abc import ABC, abstractmethod


class ScoringStrategy(ABC):
    @abstractmethod
    def calculate_score(self, rule, cue):
        pass

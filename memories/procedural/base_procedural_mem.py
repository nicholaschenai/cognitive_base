"""
base scaffolding for rules and rule matching
"""

from ..base_mem import BaseMem
from .scoring_strategies.jaccard_scoring_strategy import JaccardScoringStrategy

from ...utils import load_json, dump_json


class BaseProceduralMem(BaseMem):
    """
    Base class for procedural memory. It initializes the procedural memory with a scoring strategy, 
    retrieval top k, checkpoint directory, vectordb name, and a flag to resume from the last checkpoint. 
    It also loads rules from a JSON file if the resume flag is set.
    """
    def __init__(
        self,
        scoring_strategy=JaccardScoringStrategy(),
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        vectordb_name="procedural",
        resume=True,
        **kwargs,
    ):
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            resume=resume,
            **kwargs,
        )

        self.rules = []
        self.rules_path = f"{self.ckpt_dir}/{self.vectordb_name}/rules.json"
        if resume:
            self.rules = load_json(self.rules_path)
            if not self.rules:
                self.rules = []

        # other options: weighted score, hybrid score, percent score, embedding score
        # TODO: currently only jaccard works. refactor others to use generic dict rules rather than ProceduralRule
        self.scoring_strategy = scoring_strategy

    """
    helper fns
    """

    # TODO: refactor to use generic dict rules rather than ProceduralRule
    @staticmethod
    def exact_match(rule, problem_conditions):
        """
        Static method to check if the given problem conditions exactly match the rule's conditions.

        :param rule: The rule to match against.
        :param problem_conditions: The conditions of the problem to match.
        :return: True if all conditions match exactly, False otherwise.
        """
        # Check if the problem conditions match the rule's conditions
        return all(cond in problem_conditions for cond in rule.conditions)

    """
    Retrieval Actions (to working mem / decision procedure)
    """

    # TODO: refactor to use generic dict rules rather than ProceduralRule
    # Hierarchical Rule Representation
    def retrieve_by_priority(self, problem_conditions):
        """
        Retrieves a rule based on priority from the list of rules if it exactly matches the given problem conditions.

        :param problem_conditions: The conditions of the problem to match.
        :return: The first rule that matches the problem conditions based on priority, or None if no match is found.
        """
        # Sort rules by priority
        self.rules.sort(key=lambda rule: rule.priority, reverse=True)
        for rule in self.rules:
            if self.exact_match(rule, problem_conditions):
                return rule
        return None

    def retrieve_by_score(self, cue, threshold=0.5):
        """
        Retrieves the best matching rule based on a scoring strategy for a given cue, considering a threshold.

        :param cue: The cue to match against the rules.
        :param threshold: The minimum score threshold for a rule to be considered a match.
        :return: The best matching rule if its score is above the threshold, None otherwise.
        """
        best_match = None
        best_score = 0
        for rule in self.rules:
            score = self.scoring_strategy.calculate_score(rule, cue)
            if score > best_score and score >= threshold:
                best_match = rule
                best_score = score
        return best_match

    """
    Learning Actions (from working mem)
    """
    def add_rule(self, rule):
        self.rules.append(rule)
        dump_json(self.rules, self.rules_path)

    # TODO: weights, priority
    # def add_rule(self, conditions, action, weights=None, priority=1):
    #     self.rules.append(ProceduralRule(conditions, action, weights, priority))

    # sketch code to break ties by num conditions
    # def retrieve_by_score(self, cue, threshold=0.5, break_ties=False):
    #     best_match = None
    #     best_score = 0
    #     best_num_cond = 0
    #     for rule in self.rules:
    #         score = self.scoring_strategy.calculate_score(rule, cue)
    #         num_cond = len(rule['rigid_conditions'])
    #         tie_criteria = (score == best_score and num_cond > best_num_cond) if break_ties else False
    #         if (score > best_score and score >= threshold) or tie_criteria:
    #             best_match = rule
    #             best_score = score
    #             best_num_cond = num_cond
    #     return best_match

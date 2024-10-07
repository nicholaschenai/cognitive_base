from .jaccard_matching import jaccard_match_score
from .embedding_score import embedding_match_score


def hybrid_match_score(rule, problem_conditions, weight_fuzzy=0.5, weight_embedding=0.5):
    fuzzy_score = jaccard_match_score(rule, problem_conditions)
    embedding_score = embedding_match_score(rule, problem_conditions)
    return weight_fuzzy * fuzzy_score + weight_embedding * embedding_score

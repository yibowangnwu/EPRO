"""Portable boxed-answer math reward using verl's bundled evaluator.

The original study's custom evaluator is not bundled in this release. This
adapter is for the runnable example; pass another reward path to use your own.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    from verl.utils.reward_score.math_reward import compute_score as boxed_math_score

    return boxed_math_score(solution_str, str(ground_truth))

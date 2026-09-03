"""SHAP adapter for explaining a scalar quality function over endpoint prompts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def explain_text_score(
    texts: Sequence[str],
    score_function: Callable[[Sequence[str]], Sequence[float]],
    tokenizer,
    max_evals: int = 100,
):
    """Explain token contributions to a scalar score returned by an endpoint.

    For generative models SHAP needs a scalar target. Callers should use a task
    metric such as answer token-F1, groundedness, or safety risk, and document
    that target alongside every explanation.
    """
    import shap

    masker = shap.maskers.Text(tokenizer)

    def model_function(masked_texts: Sequence[str]) -> np.ndarray:
        scores = np.asarray(score_function(masked_texts), dtype=float)
        return scores.reshape(-1, 1)

    explainer = shap.Explainer(
        model_function,
        masker,
        algorithm="partition",
        output_names=["quality_score"],
    )
    return explainer(list(texts), max_evals=max_evals)

"""DeepEval RAG metrics using the inference job's Azure-authenticated judge."""

import asyncio
import json
import logging
import os
from typing import Callable

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

log = logging.getLogger(__name__)


class AzureDeepEvalJudge(DeepEvalBaseLLM):
    def __init__(self, completion: Callable[[str], str], deployment: str):
        self.completion = completion
        self.deployment = deployment
        super().__init__(model_name=deployment)

    def load_model(self):
        return self.completion

    def get_model_name(self) -> str:
        return self.deployment

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            prompt += "\nReturn JSON matching this schema:\n" + json.dumps(schema.model_json_schema())
        text = self.completion(prompt)
        return schema.model_validate_json(text) if schema is not None else text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        return await asyncio.to_thread(self.generate, prompt, schema)


def build_deepeval_judge(completion: Callable[[str], str], deployment: str):
    model = AzureDeepEvalJudge(completion, deployment)

    def judge(*, question: str, context: str, response: str, ground_truth: str) -> dict:
        test_case = LLMTestCase(
            input=question,
            actual_output=response,
            expected_output=ground_truth,
            retrieval_context=[context],
        )
        metrics = {
            "faithfulness": FaithfulnessMetric(model=model, include_reason=True, async_mode=False),
            "relevancy": AnswerRelevancyMetric(model=model, include_reason=True, async_mode=False),
            "correctness": GEval(
                name="Correctness",
                criteria=(
                    "Assess the factual accuracy and completeness of actual_output against "
                    "expected_output when answering input. The expected_output is cot_answer "
                    "and may contain reasoning plus a final answer inside <ANSWER> tags. "
                    "Compare final answers semantically; do not penalize different wording "
                    "or reasoning style. Penalize contradictions and missing required facts."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=model,
                async_mode=False,
            ),
        }
        results = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                results[name] = {"score": metric.score, "reason": metric.reason}
            except Exception as exc:
                log.exception("DeepEval %s measurement failed", name)
                error = f"{type(exc).__name__}: {exc}"
                results[name] = {"score": None, "reason": error, "error": error}
        return results

    return judge

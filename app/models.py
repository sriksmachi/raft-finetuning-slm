"""Validated action inputs for isolated demo and production APIs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool = Field(default=False, strict=True)


class Preparation(Action):
    source_folder: str = Field(min_length=1, max_length=500)
    recipe: Literal["context-grounded", "context-assisted"]
    scenario: Literal["passing", "blocked", "failure"] = "passing"
    num_samples: int = Field(default=240, ge=3, le=100000, strict=True)


class ProductionPreparation(Action):
    source_folder: str = Field(min_length=1, max_length=500)
    recipe: Literal["context-grounded"] = "context-grounded"
    num_samples: int = Field(default=240, ge=3, le=100000, strict=True)
    num_questions: int = Field(default=5, ge=1, le=100, strict=True)
    oracle_probability: float = Field(default=0.8, ge=0, le=1)
    data_asset_name: str = Field(default="slm-studio-data", pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,99}$")


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: Literal[
        "demo-phi-3-mini",
        "demo-phi-4-mini",
        "demo-llama-3.2-1b",
        "demo-llama-3.2-3b",
        "demo-qwen-2.5-1.5b",
        "demo-qwen-2.5-3b",
        "demo-ministral-3b",
        "demo-mistral-7b-instruct",
    ] = "demo-phi-3-mini"
    max_seq_length: int = Field(default=2048, ge=128, le=8192)
    batch_size: int = Field(default=2, ge=1, le=32)
    gradient_accumulation: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    epochs: int = Field(default=3, ge=1, le=20)
    lora_rank: int = Field(default=16, ge=1, le=128)


class Training(Parameters, Action):
    pass


class Registration(Action):
    override: bool = False
    confirmation: str = ""
    reason: str = Field(default="", max_length=2000)
    actor: str = Field(default="local-demo-user", min_length=1, max_length=100)


class Deployment(Action):
    actor: str = Field(default="local-demo-user", min_length=1, max_length=100)
    traffic_percent: int = Field(default=10, ge=1, le=100)

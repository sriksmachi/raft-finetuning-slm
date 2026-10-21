"""Validated action inputs for isolated demo and production APIs."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool = Field(default=False, strict=True)


class Preparation(Action):
    source_folder: str = Field(default="data/", min_length=1, max_length=500)
    recipe: Literal["context-grounded", "context-assisted"]
    scenario: Literal["passing", "blocked", "failure"] = "passing"
    num_samples: int = Field(default=240, ge=3, le=100000, strict=True)


class ProductionPreparation(Action):
    source_folder: str = Field(default="data/", min_length=1, max_length=500)
    recipe: Literal["context-grounded"] = "context-grounded"
    num_samples: int = Field(default=240, ge=3, le=100000, strict=True)
    num_questions: int = Field(default=5, ge=1, le=100, strict=True)
    oracle_probability: float = Field(default=0.8, ge=0, le=1)
    data_asset_name: str = Field(default="slm-studio-data", pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,99}$")
    data_asset_version: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("data-%d%m%y%H%M%S"),
        min_length=17, max_length=30, pattern=r"^data-[0-9]{12}$", validate_default=True,
    )

    @field_validator("data_asset_version")
    @classmethod
    def valid_asset_timestamp(cls, value: str) -> str:
        datetime.strptime(value.removeprefix("data-"), "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
        return value


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


class TrainerParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    base_model: str = Field(default="Llama-3.2-3B-Instruct", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
    max_seq_length: int = Field(default=2048, ge=128, le=32768, strict=True)
    learning_rate: float = Field(default=2e-5, gt=0, le=0.01)
    epochs: float = Field(default=1, gt=0, le=100)
    batch_size: int = Field(default=2, ge=1, le=64, strict=True)
    gradient_accumulation: int = Field(default=16, ge=1, le=256, strict=True)
    lora_rank: int = Field(default=16, ge=1, le=256, strict=True)
    lora_adapter_name: str = Field(default="finetuned_adapter", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    chat_template: str = Field(default="llama-3.2", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    load_in_4bit: bool = Field(default=False, strict=True)
    seed: int = Field(default=42, ge=0, le=4294967295, strict=True)


class ProductionTraining(TrainerParameters, Action):
    dataset_run_id: str | None = Field(default=None, pattern=r"^prod-[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    dataset_id: str | None = Field(
        default=None, pattern=r"^azureml:[A-Za-z][A-Za-z0-9_-]{0,254}:[A-Za-z0-9_.-]{1,30}$",
    )
    dataset_recipe: Literal["context-grounded", "context-assisted"] = "context-grounded"

    @model_validator(mode="after")
    def supported_dataset(self):
        if bool(self.dataset_run_id) == bool(self.dataset_id):
            raise ValueError("Select a published dataset run or enter an Azure dataset ID, not both.")
        if self.dataset_recipe != "context-grounded":
            raise ValueError("RAG training is not defined. Select a RAFT dataset to train.")
        return self


class Registration(Action):
    override: bool = False
    confirmation: str = ""
    reason: str = Field(default="", max_length=2000)
    actor: str = Field(default="local-demo-user", min_length=1, max_length=100)


class Deployment(Action):
    actor: str = Field(default="local-demo-user", min_length=1, max_length=100)
    traffic_percent: int = Field(default=10, ge=1, le=100)

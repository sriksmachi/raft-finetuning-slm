"""Validated inputs for production operations."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool = Field(default=False, strict=True)


class ProductionPreparation(Action):
    source_folder: str = Field(default="data/", min_length=1, max_length=500)
    recipe: Literal["context-grounded"] = "context-grounded"
    num_samples: int = Field(default=300, ge=3, le=100000, strict=True)
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


class ProductionEvaluation(Action):
    job_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class ReleasePlan(Action):
    evaluation_run_id: str = Field(pattern=r"^prod-[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    action: Literal["register", "register_and_deploy"] = "register"
    model_name: str = Field(default="slm-studio-candidate", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,99}$")
    endpoint_name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{2,31}$")
    production_data_uri: str | None = Field(default=None, min_length=1, max_length=1000)
    monitor_compute: str = Field(default="cpu-cluster", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

    @model_validator(mode="after")
    def deployment_prerequisites(self):
        if self.action == "register_and_deploy" and not (self.endpoint_name and self.production_data_uri):
            raise ValueError("Deployment needs a new endpoint name and governed normalized production request URI for drift monitoring.")
        return self


class ReleaseApproval(Action):
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2000)
    accept_gpu_cost: bool = Field(default=False, strict=True)

    @field_validator("actor", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Approval actor and reason must not be blank.")
        return value.strip()

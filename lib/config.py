"""Environment-driven configuration for Azure ML notebooks and scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureMLConfig:
    subscription_id: str
    resource_group: str
    workspace_name: str

    @classmethod
    def from_env(cls) -> "AzureMLConfig":
        values = {
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
            "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
            "workspace_name": os.getenv("AZUREML_WORKSPACE_NAME"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Missing Azure ML configuration: {names}")
        return cls(**values)  # type: ignore[arg-type]

    def create_ml_client(self):
        """Create an MLClient using the standard local/managed identity chain."""
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        return MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )

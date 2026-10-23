"""GPT-assisted sizing constrained by available GPUs and Azure retail prices."""

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

GPU_SKUS = {
    "Standard_NC4as_T4_v3": {"gpu_memory_gb": 16, "gpu_count": 1},
    "Standard_NC24ads_A100_v4": {"gpu_memory_gb": 80, "gpu_count": 1},
    "Standard_NC48ads_A100_v4": {"gpu_memory_gb": 160, "gpu_count": 2},
}
PRICE_API = "https://prices.azure.com/api/retail/prices"


def retail_price(sku: str, region: str) -> dict:
    if sku not in GPU_SKUS or not re.fullmatch(r"[a-z0-9]+", region):
        raise ValueError("Unsupported GPU SKU or Azure region.")
    query = urlencode({
        "currencyCode": "'USD'",
        "$filter": f"serviceName eq 'Virtual Machines' and armSkuName eq '{sku}' and armRegionName eq '{region}' and priceType eq 'Consumption'",
    })
    url = f"{PRICE_API}?{query}"
    source = url
    matches = []
    seen = set()
    while url:
        if url in seen or len(seen) >= 20 or urlsplit(url).hostname != "prices.azure.com":
            raise ValueError("Unexpected Azure retail pricing pagination.")
        seen.add(url)
        with urlopen(url, timeout=30) as response:
            page = json.load(response)
        for item in page["Items"]:
            label = f"{item.get('skuName', '')} {item.get('productName', '')}".lower()
            if (item.get("armSkuName") == sku and item.get("armRegionName") == region
                    and item.get("type") == "Consumption" and item.get("unitOfMeasure") == "1 Hour"
                    and item.get("currencyCode") == "USD"
                    and not any(word in label for word in ("windows", "spot", "low priority"))):
                price = item.get("retailPrice")
                if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
                    raise ValueError("Azure returned an invalid hourly retail price.")
                matches.append(item)
        url = page.get("NextPageLink")
    if not matches:
        raise ValueError(f"No Linux PAYG hourly retail price found for {sku} in {region}.")
    latest = max(item["effectiveStartDate"] for item in matches)
    current = [item for item in matches if item["effectiveStartDate"] == latest]
    if len({item["retailPrice"] for item in current}) != 1:
        raise ValueError("Azure returned ambiguous GPU prices; deployment approval is unavailable.")
    return {
        "hourly_usd": current[0]["retailPrice"], "source": source,
        "effective_start": latest, "meter_id": current[0]["meterId"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "basis": "One Linux PAYG VM estimate; excludes storage, network, monitoring, taxes and contractual discounts. GPU billing continues while allocated.",
    }


def recommend_gpu(model_name: str, training: dict, available: set[str], region: str, credential, max_input_tokens: int = 2048) -> dict:
    from azure.identity import get_bearer_token_provider
    from openai import AzureOpenAI

    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)[bB](?:[-_/]|$)", model_name)
    if not match:
        raise ValueError("Model name must record its parameter size (for example 3B); automatic GPU sizing cannot safely infer it.")
    parameters = float(match[1])
    # Scoring loads full fp16 base weights even when the adapter was trained in 4-bit.
    required = math.ceil(parameters * 2 * 1.4 + 4 * max(1, max_input_tokens / 2048))
    candidates = [
        {"sku": sku, **specification, "price": retail_price(sku, region)}
        for sku, specification in GPU_SKUS.items()
        if sku in available and specification["gpu_memory_gb"] >= required
    ]
    if not candidates:
        raise ValueError(f"No supported regional GPU SKU fits the estimated {required} GB fp16 inference budget.")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    scope = os.getenv("AZURE_OPENAI_TOKEN_SCOPE") or (
        "https://ai.azure.com/.default" if (urlsplit(endpoint).hostname or "").endswith(".services.ai.azure.com")
        else "https://cognitiveservices.azure.com/.default"
    )
    deployment = os.environ["AZURE_OPENAI_GPT_DEPLOYMENT"]
    client = AzureOpenAI(
        azure_endpoint=endpoint, api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_ad_token_provider=get_bearer_token_provider(credential, scope),
    )
    response = client.chat.completions.create(
        model=deployment, max_completion_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Select the lowest-cost suitable GPU from the supplied eligible list for single-replica fp16 LoRA serving. Never invent SKUs or prices. Return JSON with sku and rationale strings."},
            {"role": "user", "content": json.dumps({
                "model": model_name, "estimated_parameters_billion": parameters,
                "training": training, "serving": "fp16 full base plus LoRA, not training-time 4-bit",
                "estimated_required_gpu_memory_gb": required, "eligible": candidates,
            })},
        ],
    )
    recommendation = json.loads(response.choices[0].message.content)
    selected = next((entry for entry in candidates if entry["sku"] == recommendation.get("sku")), None)
    if selected is None or not isinstance(recommendation.get("rationale"), str) or not recommendation["rationale"].strip():
        raise ValueError("GPT did not return an eligible SKU with a rationale.")
    if selected["price"]["hourly_usd"] > min(entry["price"]["hourly_usd"] for entry in candidates):
        raise ValueError("GPT recommendation was not the lowest priced eligible SKU; no GPU was allocated.")
    return {
        **selected, "rationale": recommendation["rationale"], "region": region,
        "estimated_parameters_billion": parameters, "estimated_required_gpu_memory_gb": required,
        "serving_dtype": "float16", "instance_count": 1, "recommender": deployment,
        "max_input_tokens": max_input_tokens, "max_output_tokens": 512,
        "quota_note": "Regional SKU support checked; Azure quota and capacity are revalidated during provisioning.",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }

import os
import json
import time
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types


# ----------------------------
# JSON parsing / validation
# ----------------------------
def parse_json_strict(s: str) -> list:
    """Parse JSON and return an array of error objects."""
    s = (s or "").strip()
    
    # Try parsing as-is
    try:
        obj = json.loads(s)
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # If it's a dict, try to convert to array format
            # This handles cases where model returns dict instead of array
            result = []
            for category, report in obj.items():
                result.append({"category": category, "report": report})
            return result
    except json.JSONDecodeError:
        pass

    # Recovery: try to find array brackets
    i = s.find("[")
    j = s.rfind("]")
    if i >= 0 and j > i:
        try:
            arr = json.loads(s[i : j + 1])
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    # Recovery: try to find object brackets (convert dict to array)
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            if isinstance(obj, dict):
                result = []
                for category, report in obj.items():
                    result.append({"category": category, "report": report})
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError("Model output was not valid JSON.")


def validate_errors_array(errors: Any) -> None:
    """Validate the errors array structure."""
    if not isinstance(errors, list):
        raise ValueError(f"errors must be a JSON array, got {type(errors).__name__}")
    
    if len(errors) < 3:
        categories = [item.get("category", "unknown") if isinstance(item, dict) else "invalid" for item in errors]
        raise ValueError(f"Expected at least 3 errors, got {len(errors)}. Categories returned: {categories}")
    
    for i, item in enumerate(errors):
        if not isinstance(item, dict):
            raise ValueError(f"Error item {i} must be an object/dict, got {type(item).__name__}")
        
        if "category" not in item:
            raise ValueError(f"Error item {i} missing 'category' field. Available keys: {list(item.keys())}")
        if "report" not in item:
            raise ValueError(f"Error item {i} missing 'report' field. Available keys: {list(item.keys())}")
        
        if not isinstance(item["category"], str):
            raise ValueError(f"Error item {i} 'category' must be a string, got {type(item['category']).__name__}")
        if not isinstance(item["report"], str):
            raise ValueError(f"Error item {i} 'report' must be a string, got {type(item['report']).__name__}")
        if not item["report"].strip():
            raise ValueError(f"Error item {i} 'report' cannot be empty")


# ----------------------------
# Prompt helpers
# ----------------------------
def read_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove the placeholders (they will be replaced by actual content in user prompt)
    text = text.replace("<<<QUERY_TEXT>>>", "").strip()
    text = text.replace("<<<REPORT_TEXT>>>", "").strip()

    return text


def build_user_prompt(findings: str, impression: str) -> str:
    # Keep this short; you want the cached prefix to contain the big stuff.
    # The prompt already contains instructions, we just need to provide the findings and impression.
    return f"FINDINGS:\n{findings.strip()}\n\nIMPRESSION:\n{impression.strip()}"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# ----------------------------
# Usage metadata helpers (robust across SDK field casing)
# ----------------------------
def _get_field(obj: Any, names: Tuple[str, ...], default=0) -> int:
    if obj is None:
        return default
    # pydantic model / object
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is None:
                continue
            return int(v)
    # dict
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return int(obj[n])
    return default


def estimate_cost_usd(
    usage_metadata: Any,
    input_per_m: float,
    cached_input_per_m: float,
    output_per_m: float,
) -> float:
    """
    Estimate cost from token usage metadata.
    Notes:
      - promptTokenCount includes cached tokens when cachedContent is used.
      - cachedContentTokenCount is the cached portion.
      - output pricing includes reasoning; if thoughtsTokenCount is provided, include it too.
    """
    prompt_tokens = _get_field(usage_metadata, ("prompt_token_count", "promptTokenCount"))
    cached_tokens = _get_field(usage_metadata, ("cached_content_token_count", "cachedContentTokenCount"))
    cand_tokens = _get_field(usage_metadata, ("candidates_token_count", "candidatesTokenCount"))
    thoughts_tokens = _get_field(usage_metadata, ("thoughts_token_count", "thoughtsTokenCount"))

    non_cached_tokens = max(prompt_tokens - cached_tokens, 0)
    billed_output_tokens = cand_tokens + thoughts_tokens

    cost = (
        non_cached_tokens * (input_per_m / 1_000_000.0)
        + cached_tokens * (cached_input_per_m / 1_000_000.0)
        + billed_output_tokens * (output_per_m / 1_000_000.0)
    )
    return float(cost)


# ----------------------------
# Context cache
# ----------------------------
def get_or_create_cache(
    client: genai.Client,
    model: str,
    system_prompt: str,
    cache_file: str,
    ttl_seconds: int,
    display_name: str,
) -> str:
    """
    Creates an explicit context cache and stores its full resource name locally.

    IMPORTANT (Vertex AI):
      cached content name format must be full resource name like:
      projects/PROJECT_NUMBER/locations/LOCATION/cachedContents/CACHE_ID
    """
    if os.path.exists(cache_file):
        name = open(cache_file, "r", encoding="utf-8").read().strip()
        if name:
            try:
                client.caches.get(name=name)
                return name
            except Exception:
                pass  # expired or not found -> recreate

    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=system_prompt,
            # minimal placeholder content is fine; system_instruction is what we want cached
            contents=[types.Content(role="user", parts=[types.Part(text=" ")])],
            ttl=f"{ttl_seconds}s",
        ),
    )

    name = cache.name
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(name)
    return name


# ----------------------------
# Model call
# ----------------------------
def call_error_generation(
    client: genai.Client,
    model: str,
    cache_name: str,
    findings: str,
    impression: str,
    max_output_tokens: int,
    thinking_level: str,
    retries: int = 5,
) -> Tuple[list, Any]:
    """Generate error variants of the IMPRESSION that contradict the FINDINGS."""
    user_prompt = build_user_prompt(findings, impression)

    # Response schema: array of objects with "category" and "report" fields
    # Example: [{"category": "Change Severity", "report": "..."}, ...]
    # Require at least 3 items as specified in the prompt
    response_schema = {
        "type": "array",
        "minItems": 3,
        "items": {
            "type": "object",
            "required": ["category", "report"],
            "properties": {
                "category": {"type": "string"},
                "report": {"type": "string"},
            },
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    max_output_tokens=max_output_tokens,
                    candidate_count=1,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                ),
            )

            errors_array = parse_json_strict(resp.text or "")

            # Validate the errors array
            validate_errors_array(errors_array)

            return errors_array, resp.usage_metadata

        except Exception as e:
            last_err = e
            print(f"[retry {attempt+1}/{retries}] {type(e).__name__}: {e}")
            time.sleep(min(30, 2 ** attempt))

    raise last_err if last_err else RuntimeError("Unknown failure.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--system_prompt_file", required=True)

    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")  # or "global"

    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--thinking_level", default="low", choices=["minimal", "low", "medium", "high"])

    ap.add_argument("--cache_file", default=".vertex_cache_name.txt")
    ap.add_argument("--cache_ttl_seconds", type=int, default=24 * 3600)
    ap.add_argument("--checkpoint_every", type=int, default=1000)
    ap.add_argument("--max_output_tokens", type=int, default=2048, help="Max tokens for error generation (default 2048 for IMPRESSION variants)")
    ap.add_argument("--max_usd", type=float, default=0.0, help="Hard stop when estimated spend reaches this USD amount (0 disables).")
    ap.add_argument("--max_rows", type=int, default=0, help="Process at most N rows this run (0 disables).")

    args = ap.parse_args()

    # Vertex AI client (uses ADC or GOOGLE_APPLICATION_CREDENTIALS)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    system_prompt = read_prompt(args.system_prompt_file)

    cache_name = get_or_create_cache(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        cache_file=args.cache_file,
        ttl_seconds=args.cache_ttl_seconds,
        display_name=f"error-generation-{prompt_hash(system_prompt)}",
    )
    print(f"Using context cache: {cache_name}")

    # Pricing for Gemini 3 Flash preview on Vertex AI (<=200k input):
    # Input $0.50 / 1M, cached input $0.05 / 1M, output $3.00 / 1M
    # (Keep in sync with the Vertex AI pricing page.)
    INPUT_PER_M = 0.50
    CACHED_INPUT_PER_M = 0.05
    OUTPUT_PER_M = 3.00

    df = pd.read_csv(args.input_csv)
    
    # Ensure required columns exist
    if "query" not in df.columns:
        raise ValueError("Input CSV must have 'query' column containing the FINDINGS section")
    if "answer" not in df.columns:
        raise ValueError("Input CSV must have 'answer' column containing the IMPRESSION section")
    
    if "output" not in df.columns:
        df["output"] = None

    # Optional tracking columns (useful for auditing spend)
    for col in ["est_cost_usd"]:
        if col not in df.columns:
            df[col] = None

    total_est = 0.0
    processed = 0

    for i in tqdm(range(len(df))):
        if args.max_rows and processed >= args.max_rows:
            print(f"Reached --max_rows={args.max_rows}. Stopping.")
            break

        existing = df.at[i, "output"]
        if isinstance(existing, str) and existing.strip() and existing.strip().lower() != "n/a":
            continue

        findings = str(df.at[i, "query"])
        impression = str(df.at[i, "answer"])
        
        if pd.isna(df.at[i, "query"]) or not findings.strip():
            print(f"[row {i}] Skipping empty findings")
            df.at[i, "output"] = "n/a"
            continue
        
        if pd.isna(df.at[i, "answer"]) or not impression.strip():
            print(f"[row {i}] Skipping empty impression")
            df.at[i, "output"] = "n/a"
            continue

        try:
            errors_array, usage = call_error_generation(
                client=client,
                model=args.model,
                cache_name=cache_name,
                findings=findings,
                impression=impression,
                max_output_tokens=args.max_output_tokens,
                thinking_level=args.thinking_level,
            )

            # Save the errors array as JSON string
            df.at[i, "output"] = json.dumps(errors_array, ensure_ascii=False)

            est = estimate_cost_usd(usage, INPUT_PER_M, CACHED_INPUT_PER_M, OUTPUT_PER_M)
            df.at[i, "est_cost_usd"] = est
            total_est += est
            processed += 1

            if args.max_usd and total_est >= args.max_usd:
                print(f"Reached --max_usd=${args.max_usd:.2f} (estimated). Stopping.")
                break

        except Exception as e:
            print(f"[row {i}] error: {e}")
            df.at[i, "output"] = "n/a"
            df.at[i, "est_cost_usd"] = None

        if (i + 1) % args.checkpoint_every == 0:
            df.to_csv(args.output_csv, index=False)
            print(f"Checkpoint saved at row {i + 1}. Estimated spend so far: ${total_est:.4f}")

    df.to_csv(args.output_csv, index=False)
    print(f"Done. Estimated spend this run: ${total_est:.4f}")


if __name__ == "__main__":
    main()

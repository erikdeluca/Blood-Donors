"""AI agents integrated in the project.

Usage:
    python ai_agents.py reviewer app/logic.py
    python ai_agents.py reviewer "notebooks/Bayesian HMM GLM.ipynb"
    python ai_agents.py documenter python/hmm_glm_model.py --model gemini-3.1-flash-lite
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

DEFAULT_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
RETRYABLE_CODES = {429, 500, 503, 504}
MAX_CHARS = 80_000

ROLE_PROMPTS: dict[str, str] = {
    "reviewer": (
        "Act as a Senior Software Engineer and Data Scientist. "
        "Review the code below focusing on logic errors, bugs, security risks, "
        "and major maintainability issues. Be concise and return a Markdown "
        "bulleted list. If everything looks good, reply with 'OK'."
    ),
    "documenter": (
        "Act as a Technical Writer. "
        "Generate concise Google-style docstrings and a short README summary "
        "for the code below."
    ),
    "tester": (
        "Act as a QA Engineer. "
        "Write pytest unit tests for the code below. Return ONLY Python code."
    ),
}


def ipynb_to_qmd(path: Path) -> str:
    """Convert a Jupyter notebook into clean Quarto (.qmd) text without outputs."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    lang = (
        nb.get("metadata", {})
        .get("kernelspec", {})
        .get("language", "python")
        .lower()
    )

    chunks: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue

        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            chunks.append(source)
        elif cell_type == "code":
            chunks.append(f"```{{{lang}}}\n{source}\n```")

    return "\n\n".join(chunks)


def load_file_content(path: Path, max_chars: int = MAX_CHARS) -> str:
    """Read source code, stripping outputs when the input is a Jupyter notebook."""
    if path.suffix.lower() == ".ipynb":
        content = ipynb_to_qmd(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")

    if len(content) > max_chars:
        content = (
            content[:max_chars]
            + f"\n\n... [truncated after {max_chars:,} characters] ..."
        )
    return content

def resolve_models(selected: str | None) -> tuple[str, ...]:
    if selected:
        return (selected, *(m for m in DEFAULT_MODELS if m != selected))
    return DEFAULT_MODELS


def run_agent(
    role: str,
    file_path: Path,
    content: str,
    *,
    api_key: str,
    models: tuple[str, ...],
    max_retries: int = 3,
) -> str:
    """Send file content to Gemini with retry on transient errors and model fallback."""
    client = genai.Client(api_key=api_key)
    user_message = f"File: {file_path.as_posix()}\n\n{content}"
    config = types.GenerateContentConfig(
        system_instruction=ROLE_PROMPTS[role],
        temperature=0.2,
    )

    last_error = "❌ Unknown error"
    for model in models:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=config,
                )
                return (response.text or "⚠️ Empty response from model.").strip()
            except errors.APIError as e:
                last_error = f"❌ Agent API Error ({e.code}, {model}): {e.message}"
                if e.code not in RETRYABLE_CODES:
                    return last_error
                if attempt < max_retries:
                    wait = 2**attempt
                    print(
                        f"⚠️ {model} returned {e.code} (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
        print(f"↪️ Switching fallback model after {model} failed...", file=sys.stderr)

    return last_error

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight Gemini AI agents on project files."
    )
    parser.add_argument(
        "role",
        choices=sorted(ROLE_PROMPTS),
        help="Agent role to execute.",
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the source file or Jupyter notebook (.ipynb).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL"),
        help=f"Primary Gemini model (default fallback chain: {', '.join(DEFAULT_MODELS)}).",
    )
    return parser.parse_args(argv)


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.", file=sys.stderr)
        return 1

    if not args.file_path.is_file():
        print(f"❌ File not found: {args.file_path}", file=sys.stderr)
        return 1

    try:
        content = load_file_content(args.file_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"❌ Failed to read {args.file_path}: {e}", file=sys.stderr)
        return 1

    if not content.strip():
        print("OK (empty file)")
        return 0

    models = resolve_models(args.model)
    print(
        f"🤖 Agent {args.role.upper()} ({models[0]}) analyzing {args.file_path}...",
        file=sys.stderr,
    )
    print(
        run_agent(
            args.role,
            args.file_path,
            content,
            api_key=api_key,
            models=models,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

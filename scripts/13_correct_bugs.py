import argparse
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv

# Official library imports
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types

# 1. LOAD ENVIRONMENT VARIABLES
# Looks for .env next to this script and one level up (the project root), instead of relying
# on whatever the current working directory happens to be when you run "python 13_...py".
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_CANDIDATES = [
    os.path.join(_HERE, '.env'),
    os.path.join(_HERE, '..', '.env'),
    'C:/Dissertacao/api_keys',          # in case this is the keys file itself (no extension)
    'C:/Dissertacao/api_keys/.env',     # in case this is a folder containing a .env file
    'C:/Dissertacao/api_keys.env',
]
_ENV_LOADED_FROM = None
for _candidate in _ENV_CANDIDATES:
    if os.path.isfile(_candidate):
        load_dotenv(dotenv_path=_candidate)
        _ENV_LOADED_FROM = _candidate
        break
else:
    load_dotenv()  # last resort: default python-dotenv search

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_missing = [name for name, value in [
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ("GEMINI_API_KEY", GEMINI_API_KEY),
] if not value]
if _missing:
    print(f"Looked for .env at: {_ENV_CANDIDATES} (found: {_ENV_LOADED_FROM})")
    raise SystemExit(
        f"ERROR: missing environment variable(s) {_missing}. "
        f"Create a .env file (next to this script or one folder above) with lines like "
        f"OPENAI_API_KEY=sk-..., no quotes, no spaces around '='."
    )

# 2. CLIENT INITIALIZATION
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================================
# CONFIGURATION
# ==========================================================
# NOTE: this input is the NEW holdout (script 8, case-level split). It has ~305 bugs, not 300.
DEFAULT_INPUT = 'C:/Dissertacao/data_bases/04_final/holdout_305_bugs_llm.csv'
# NOTE: deliberately a NEW file name, different from any previous run, so this never "resumes"
# from results that were computed on the OLD (leaked) holdout.
DEFAULT_OUTPUT = 'C:/Dissertacao/data_bases/05_results/tournament_results_holdout_v2_pt.csv'

# Double-check these against each provider's current docs before running: exact API model
# identifiers change over time and an outdated one fails immediately for every call.
GPT_MODEL = "gpt-5.4"
CLAUDE_MODEL = "claude-opus-4-6"
GEMINI_MODEL = "gemini-3.1-pro-preview"

PROMPT_TEMPLATES = {
    'en': """You are a software engineer specialized in Java.
Fix the Resource Leak in the code below.
RULE: Return ONLY the corrected code.
No explanations, no markdown (```java), no greetings.

Code:
{code}
""",
    'pt': """Você é um engenheiro de software especialista em Java.
Corrija o Resource Leak no código abaixo.
REGRA: Retorne APENAS o código corrigido.
Sem explicações, sem markdown (```java), sem saudações.

Código:
{code}
""",
}

CODE_FENCE_RE = re.compile(r'^\s*```(?:java)?\s*\n?|\n?\s*```\s*$', re.MULTILINE)


def strip_code_fences(text):
    """Defensive cleanup: some models wrap the answer in ```java ... ``` despite instructions."""
    if not isinstance(text, str):
        return text
    return CODE_FENCE_RE.sub('', text).strip()


def is_ok(value):
    """True only for a real, non-empty, non-error response."""
    return (pd.notna(value) and str(value).strip() != ""
            and not str(value).strip().startswith("ERROR"))


# ==========================================
# 3. CALL FUNCTIONS FOR EACH MODEL
# (all error paths return a string starting with "ERROR" so is_ok() can detect them)
# ==========================================

def request_gpt_fix(prompt):
    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return strip_code_fences(response.choices[0].message.content)
    except Exception as e:
        return f"ERROR (GPT): {e}"


def request_claude_fix(prompt):
    try:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return strip_code_fences(response.content[0].text)
    except Exception as e:
        return f"ERROR (Claude): {e}"


def request_gemini_fix(prompt):
    try:
        # thinking_level="low": Gemini 3.1 Pro cannot fully disable its internal reasoning
        # ("thinking tokens"), which are billed as output tokens and can dwarf the visible
        # answer for a simple task like this one. "low" is the cheapest setting available and
        # is enough for a mechanical resource-leak fix (no complex reasoning needed).
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
                thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return strip_code_fences(response.text)
    except Exception as e:
        return f"ERROR (Gemini): {e}"


# ==========================================
# 4. MAIN PIPELINE (The Tournament)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--lang', choices=['en', 'pt'], default='pt',
                        help='language of the prompt sent to the LLMs')
    args = parser.parse_args()

    print("Starting the LLM Tournament...")

    if not os.path.exists(args.input):
        print(f"ERROR: The file {args.input} was not found!")
        return

    # ========================================================
    # RESUME LOGIC (CHECKPOINT)
    # ========================================================
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        print(f"Partial file found at {args.output}. Resuming what is missing...")
    else:
        df = pd.read_csv(args.input)
        print("Starting processing from scratch...")
        for col in ['fix_gpt', 'fix_claude', 'fix_gemini']:
            if col not in df.columns:
                df[col] = ""

    print(f"   -> {len(df)} bugs to process (the full holdout, no row is dropped).")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    template = PROMPT_TEMPLATES[args.lang]

    skipped = 0
    for index, row in df.iterrows():
        gpt_ok = is_ok(row.get('fix_gpt'))
        claude_ok = is_ok(row.get('fix_claude'))
        gemini_ok = is_ok(row.get('fix_gemini'))

        if gpt_ok and claude_ok and gemini_ok:
            skipped += 1
            continue

        print(f"Processing bug {index + 1}/{len(df)}...")

        buggy_code = row['code_with_bug']
        full_prompt = template.format(code=buggy_code)

        if not gpt_ok:
            df.at[index, 'fix_gpt'] = request_gpt_fix(full_prompt)
            time.sleep(1)

        if not claude_ok:
            df.at[index, 'fix_claude'] = request_claude_fix(full_prompt)
            time.sleep(2)

        if not gemini_ok:
            df.at[index, 'fix_gemini'] = request_gemini_fix(full_prompt)
            time.sleep(3)

        # Saves partial progress after each completed row
        df.to_csv(args.output, index=False)

    print(f"\nTournament finished! {skipped} bugs were already complete and were skipped.")
    print(f"Results saved to '{args.output}'.")


if __name__ == "__main__":
    main()
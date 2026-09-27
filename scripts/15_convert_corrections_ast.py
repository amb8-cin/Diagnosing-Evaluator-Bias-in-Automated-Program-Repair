import argparse
import os
import re

import javalang
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_INPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/tournament_results_FILTERED_AST_{lang}.csv'  # script 14
DEFAULT_OUTPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/llm_responses_ast_{lang}.csv'

LLM_COLUMNS = {
    'fix_claude': 'AST_Claude',
    'fix_gpt': 'AST_GPT',
    'fix_gemini': 'AST_Gemini',
}


def pick(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


def try_parse(cleaned_code):
    """
    Attempts to wrap the code at different levels (Class, Method, or Statements)
    until javalang can build the Abstract Syntax Tree (AST).
    """
    try:
        return javalang.parse.parse(cleaned_code)
    except Exception:
        pass

    lines = cleaned_code.split('\n')
    imports = []
    body = []
    for line in lines:
        if line.strip().startswith("import ") or line.strip().startswith("package "):
            imports.append(line)
        else:
            body.append(line)

    imports_str = "\n".join(imports)
    body_str = "\n".join(body)

    class_code = f"{imports_str}\npublic class DummyClass {{\n{body_str}\n}}"
    try:
        return javalang.parse.parse(class_code)
    except Exception:
        pass

    method_code = f"{imports_str}\npublic class DummyClass {{\n public void dummyMethod() throws Exception {{\n{body_str}\n}}\n}}"
    try:
        return javalang.parse.parse(method_code)
    except Exception:
        pass

    code_with_extra_braces = method_code + "\n}\n}\n}"
    try:
        return javalang.parse.parse(code_with_extra_braces)
    except Exception:
        return None


_CODE_START_RE = re.compile(
    r'^\s*(package\s|import\s|@\w+|public\s|private\s|protected\s|static\s|final\s|abstract\s|'
    r'class\s|void\s|try\s*[({]|for\s*\(|while\s*\(|if\s*\(|switch\s*\(|return\b|'
    r'int\s|long\s|double\s|float\s|boolean\s|char\s|String\s|\w+\s*\()'
)


def strip_leading_prose(code_str):
    """Drops any explanatory text before the first line that actually looks like Java code."""
    lines = code_str.split('\n')
    for i, line in enumerate(lines):
        if _CODE_START_RE.match(line):
            return '\n'.join(lines[i:])
    return code_str  # nothing recognizable as code; leave unchanged (will likely stay a SYNTAX_ERROR)


def extract_ast_paths(java_code):
    """Extracts the syntactic signatures from the code, cleaning up LLM artifacts."""
    code_str = str(java_code)

    if code_str.strip().startswith("ERROR"):
        return "SYNTAX_ERROR"  # an API error is, by definition, not a valid fix

    cleaned_code = code_str.replace("\u3010", "").replace("\u3011", "")
    cleaned_code = re.sub(r"^```java|```$", "", cleaned_code, flags=re.MULTILINE).strip()

    tree = try_parse(cleaned_code)

    if tree is None:
        # Some models add a prose explanation before the code despite the "no explanations"
        # instruction. That prose alone is not valid Java, so the whole response fails to parse
        # even when the actual code afterwards is fine. Retry by skipping straight to the first
        # line that looks like real Java.
        trimmed = strip_leading_prose(cleaned_code)
        if trimmed != cleaned_code:
            tree = try_parse(trimmed)

    if tree is None:
        return "SYNTAX_ERROR"

    extracted_features = []
    for path, node in tree:
        if isinstance(node, javalang.tree.TryStatement):
            extracted_features.append("TryStatement")
            if node.resources and len(node.resources) > 0:
                extracted_features.append("HasResources_ImplicitClose")
            if getattr(node, 'finally_block', None):
                extracted_features.append("HasFinallyBlock")

        elif isinstance(node, javalang.tree.MethodInvocation):
            if node.member == "close":
                extracted_features.append("MethodCall_close")
            elif node.member == "disconnect":
                extracted_features.append("MethodCall_disconnect")
            elif node.member == "release":
                extracted_features.append("MethodCall_release")
            elif node.member == "recycle":
                extracted_features.append("MethodCall_recycle")
            elif node.member == "free":
                extracted_features.append("MethodCall_free")
            elif node.member == "destroy":
                extracted_features.append("MethodCall_destroy")
            else:
                extracted_features.append("MethodCall_Other")

        elif isinstance(node, javalang.tree.CatchClause):
            extracted_features.append("CatchClause")

    if not extracted_features:
        return "NO_RELEVANT_FEATURES"

    return " ".join(extracted_features)


def convert_one_language(lang, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"\n[{lang.upper()}] SKIPPED: file not found at {input_path}")
        return None

    print(f"\n[{lang.upper()}] Reading tournament responses from: {input_path}")
    df = pd.read_csv(input_path)
    id_col = pick(df, 'Case_ID', 'ID_Caso')
    print(f"   -> Total bugs analyzed by AIs: {len(df)}")

    for col_code, col_ast in LLM_COLUMNS.items():
        if col_code in df.columns:
            print(f"   -> Extracting structures for model: {col_code}...")
            df[col_ast] = df[col_code].apply(extract_ast_paths)
        else:
            print(f"   WARNING: column {col_code} not found in CSV!")

    final_columns = [id_col] + [c for c in LLM_COLUMNS.values() if c in df.columns]
    df_ast = df[final_columns].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ast.to_csv(output_path, index=False)

    # Quick transparency check: how many responses did not even compile as valid Java
    for col_ast in LLM_COLUMNS.values():
        if col_ast in df_ast.columns:
            n_syntax_error = int((df_ast[col_ast] == "SYNTAX_ERROR").sum())
            print(f"   {col_ast}: {n_syntax_error}/{len(df_ast)} responses failed to parse (SYNTAX_ERROR)")

    print(f"   Feature file saved at: {output_path}")
    return len(df_ast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['pt', 'en', 'both'], default='both')
    parser.add_argument('--input', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--output', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    args = parser.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]

    print("Starting code conversion to Abstract Syntax Trees (AST)...")
    for lang in langs:
        input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(lang=lang)
        output_path = args.output or DEFAULT_OUTPUT_TEMPLATE.format(lang=lang)
        convert_one_language(lang, input_path, output_path)

    print("\nDone. Next step: script 17 (score these AST features with the trained oracle).")


if __name__ == "__main__":
    main()
import javalang
import pandas as pd
import re

# =====================================================================
# VALIDATION FUNCTIONS (The same ones we used before)
# =====================================================================
def try_parse(cleaned_code):
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
            
    str_imports = "\n".join(imports)
    str_body = "\n".join(body)

    class_code = f"{str_imports}\npublic class DummyClass {{\n{str_body}\n}}"
    try:
        return javalang.parse.parse(class_code)
    except Exception:
        pass

    method_code = f"{str_imports}\npublic class DummyClass {{\n public void dummyMethod() throws Exception {{\n{str_body}\n}}\n}}"
    try:
        return javalang.parse.parse(method_code)
    except Exception:
        pass

    curly_braces_code = method_code + "\n}\n}\n}"
    try:
        return javalang.parse.parse(curly_braces_code)
    except Exception:
        return None

def has_syntax_error(java_code):
    code_str = str(java_code)
    cleaned_code = code_str.replace("【", "").replace("】", "")
    cleaned_code = re.sub(r"^```java|```$", "", cleaned_code, flags=re.MULTILINE).strip()
    
    tree = try_parse(cleaned_code)
    return tree is None

# =====================================================================
# INSPECTION LOGIC
# =====================================================================
def inspect_failures(input_csv_path, output_txt_path, code_column):
    print(f"🔎 Reading CSV to hunt for errors: {input_csv_path}")
    
    try:
        df = pd.read_csv(input_csv_path)
    except pd.errors.ParserError:
        df = pd.read_csv(input_csv_path, sep=';')

    print("🕵️‍♂️ Identifying problematic snippets...\n")
    
    error_snippets = []
    
    # Iterates through the dataframe rows
    for index, row in df.iterrows():
        code = row[code_column]
        if has_syntax_error(code):
            error_snippets.append((index, code))

    total_errors = len(error_snippets)
    print(f"⚠️ {total_errors} snippets with syntax errors were found.\n")
    
    if total_errors == 0:
        print("All perfect! No errors found.")
        return

    # Prints the first 3 to the terminal for quick curiosity satisfaction
    print("-" * 50)
    print("👀 PREVIEW OF THE FIRST 3 ERRORS:")
    print("-" * 50)
    for i in range(min(3, total_errors)):
        idx, cod = error_snippets[i]
        print(f"[ROW {idx + 2} OF THE CSV]") # +2 because pandas counts from 0 and the CSV has a header
        print(cod)
        print("\n" + "="*50 + "\n")

    # Saves all to a TXT file for analysis
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"INSPECTION REPORT: {total_errors} REJECTED SNIPPETS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, cod in error_snippets:
            f.write(f"--- ERROR IN ROW {idx + 2} OF THE original CSV ---\n")
            f.write(str(cod) + "\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
    print(f"📁 A complete report with all {total_errors} broken codes was saved to:")
    print(f"👉 {output_txt_path}")

# =====================================================================
# CONFIGURATION
# =====================================================================
if __name__ == "__main__":
    INPUT_FILE = "C:/Dissertacao/data_bases/04_final/dataset_synthetic_chatgpt.csv" 
    OUTPUT_TXT_FILE = "C:/Dissertacao/artifacts/inspection_errors.txt"
    CODE_COLUMN = 'Codigo_Snippet' 
    
    inspect_failures(INPUT_FILE, OUTPUT_TXT_FILE, CODE_COLUMN)
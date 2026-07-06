import javalang
import pandas as pd
import re

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

def extract_ast_paths(java_code):
    """
    Extracts the syntactic signatures from the code, cleaning up LLM artifacts.
    """
    code_str = str(java_code)
    
    cleaned_code = code_str.replace("【", "").replace("】", "")
    cleaned_code = re.sub(r"^```java|```$", "", cleaned_code, flags=re.MULTILINE).strip()
    
    tree = try_parse(cleaned_code)
    
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

def process_tournament_responses(input_csv_path, output_csv_path):
    print(f"📄 Reading tournament responses from: {input_csv_path}")
    
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"📊 Total bugs analyzed by AIs: {len(df)}")
    print("🌲 Starting code conversion to Abstract Syntax Trees (AST)...\n")

    # Mapping: Original Code Column -> New AST Features Column
    llm_columns = {
        'fix_claude': 'AST_Claude',
        'fix_gpt': 'AST_GPT',
        'fix_gemini': 'AST_Gemini'
    }

    for col_code, col_ast in llm_columns.items():
        if col_code in df.columns:
            print(f"   -> Extracting structures for model: {col_code}...")
            # Extracts features but DOES NOT delete rows with errors.
            # Syntax error means the AI failed the repair!
            df[col_ast] = df[col_code].apply(extract_ast_paths)
        else:
            print(f"   ⚠️ Column {col_code} not found in CSV!")

    # We select only the columns that matter for the oracle
    final_columns = ['ID_Caso'] + list(llm_columns.values())
    df_ast = df[final_columns].copy()

    df_ast.to_csv(output_csv_path, index=False)
    
    print("\n✅ Success! All responses have been converted.")
    print(f"💾 Feature file generated at: {output_csv_path}")
    print("🚀 You can now run the script 'llm_tournament_ast.py'!")

# =====================================================================
if __name__ == "__main__":
    # Original file with pure Java code
    INPUT_FILE = "C:/Dissertation/data_bases/05_results/llm_tournament_results_FILTERED_AST.csv" 
    
    # New file to be consumed by the oracle validator
    OUTPUT_FILE = "C:/Dissertation/data_bases/05_results/llm_responses_ast.csv"
    
    process_tournament_responses(INPUT_FILE, OUTPUT_FILE)
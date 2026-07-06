import pandas as pd

def extract_versions_from_diff(diff_code):
    """
    Parses the diff and reconstructs the original buggy code (1) 
    and the fixed code (0), cleaning the diff signals.
    """
    if not isinstance(diff_code, str):
        return "", ""
        
    buggy_lines = []
    fix_lines = []
    
    # Ignore the first header lines of the diff (--- and +++)
    lines = diff_code.split('\n')
    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
            
        # If the line starts with '-', it belongs ONLY to the buggy version
        if line.startswith('-'):
            buggy_lines.append(line[1:]) # Removes the '-'
            
        # If the line starts with '+', it belongs ONLY to the fixed version
        elif line.startswith('+'):
            fix_lines.append(line[1:]) # Removes the '+'
            
        # If it starts with a space, it belongs to BOTH versions (it's the context)
        elif line.startswith(' '):
            buggy_lines.append(line[1:])
            fix_lines.append(line[1:])
            
        # If it has no prefix (sometimes happens due to formatting), assume context
        else:
            buggy_lines.append(line)
            fix_lines.append(line)
            
    return '\n'.join(buggy_lines).strip(), '\n'.join(fix_lines).strip()

def generate_balanced_dataset(input_csv_path, output_csv_path):
    print("Starting extraction of '0s' and balancing of the Dataset...\n")
    
    df = pd.read_csv(input_csv_path, sep=None, engine='python', encoding='utf-8-sig')
    new_data = []
    
    cases_processed = 0
    
    for index, row in df.iterrows():
        diff = row['Codigo_Diff']
        buggy_code, fix_code = extract_versions_from_diff(diff)
        
        # Only add if extraction worked
        if buggy_code and fix_code:
            # 1. Add the BUGGY version (Label = 1)
            new_data.append({
                "Case_ID": f"{row['ID_Caso']}_BUG",
                "Application": row['Aplicacao'],
                "Resource_Class": row['Classe_Recurso'],
                "Code_Snippet": buggy_code,
                "Has_Resource_Leak": 1  # 🔴 SICK
            })
            
            # 2. Add the FIXED version (Label = 0)
            new_data.append({
                "Case_ID": f"{row['ID_Caso']}_FIX",
                "Application": row['Aplicacao'],
                "Resource_Class": row['Classe_Recurso'],
                "Code_Snippet": fix_code,
                "Has_Resource_Leak": 0  # 🟢 HEALTHY
            })
            
            cases_processed += 1
            
    # Create the new DataFrame
    balanced_df = pd.DataFrame(new_data)
    
    # Save the new file (using sep=';' to open well in your Excel)
    balanced_df.to_csv(output_csv_path, index=False, sep=';', encoding='utf-8-sig')
    
    print("✅ Process completed successfully!")
    print(f"📊 Processed {cases_processed} original Diffs.")
    print(f"⚖️ The new dataset now has {len(balanced_df)} cases (50% buggy, 50% healthy)!")
    print(f"📁 File saved to: {output_csv_path}")

# ==========================================
# HOW TO RUN
# ==========================================
if __name__ == "__main__":
    # We will use the main CSV we created earlier
    ORIGINAL_CSV = 'C:/Dissertacao/data_bases/dataset_treino_droidleaks.csv'
    
    # The new file with double the size and "0s" included
    BALANCED_CSV = 'C:/Dissertacao/data_bases/data_bases/03_processed/balanced_dataset.csv'
    
    generate_balanced_dataset(ORIGINAL_CSV, BALANCED_CSV)
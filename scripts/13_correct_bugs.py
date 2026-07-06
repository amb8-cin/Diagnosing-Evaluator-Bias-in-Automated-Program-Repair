import pandas as pd
import time
import os
from dotenv import load_dotenv

# Official library imports
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. CLIENT INITIALIZATION
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 3. CALL FUNCTIONS FOR EACH MODEL
# ==========================================

def request_gpt_fix(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.4", # Update to the exact model you will use
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT ERROR: {e}"

def request_claude_fix(prompt):
    try:
        response = anthropic_client.messages.create(
            model="claude-opus-4-6", # Exact and stable model name
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"CLAUDE ERROR: {e}"

def request_gemini_fix(prompt):
    try:
        # Added '-latest' suffix, recognized by the API as default
        model = genai.GenerativeModel('gemini-3.1-pro-preview') 
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        return response.text
    except Exception as e:
        return f"GEMINI ERROR: {e}"

# ==========================================
# 4. MAIN PIPELINE (The Tournament)
# ==========================================
def main():
    print("🚀 Starting the LLM Tournament...")
    
    input_file = './data_sets/holdout_300_bugs_llm.csv'
    output_file = './data_sets/05_results/llm_tournament_results.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: The file {input_file} was not found!")
        return

    # ========================================================
    # RESUME LOGIC (CHECKPOINT)
    # ========================================================
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print("📁 Partial file found. Evaluating what has already been processed...")
    else:
        df = pd.read_csv(input_file)
        print("🚀 Starting processing from scratch...")
        # Creates columns if they don't exist
        for col in ['fix_gpt', 'fix_claude', 'fix_gemini']:
            if col not in df.columns:
                df[col] = ""

    # Gets the 300 cases
    df = df.head(300) 

    skipped_cases = 0

    # Iterate over each bug
    for index, row in df.iterrows():
        
        # Security check: verifies if all 3 responses exist and are not errors
        gpt_ok = pd.notna(row.get('fix_gpt')) and str(row.get('fix_gpt')).strip() != "" and not str(row.get('fix_gpt')).startswith("ERROR")
        claude_ok = pd.notna(row.get('fix_claude')) and str(row.get('fix_claude')).strip() != "" and not str(row.get('fix_claude')).startswith("ERROR")
        gemini_ok = pd.notna(row.get('fix_gemini')) and str(row.get('fix_gemini')).strip() != "" and not str(row.get('fix_gemini')).startswith("ERROR")

        if gpt_ok and claude_ok and gemini_ok:
            skipped_cases += 1
            continue # Skips to the next iteration silently
            
        # If reached here, it means there's missing processing (e.g., will start from 46)
        if skipped_cases > 0 and skipped_cases == index:
            print(f"⏩ Skipping {skipped_cases} successfully processed cases...")
            
        print(f"🔄 Processing bug {index + 1}/{len(df)}...")
        
        buggy_code = row['code_with_bug'] # Confirm if this is indeed the column name
        
        full_prompt = f"""
        Você é um engenheiro de software especialista em Java. 
        Corrija o Resource Leak no código abaixo.
        REGRA: Retorne APENAS o código corrigido. 
        Sem explicações, sem markdown (```java), sem saudações.
        
        Código:
        {buggy_code}
        """
        
        # Executing calls (Checking if empty or if an ERROR occurred before calling)
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
        df.to_csv(output_file, index=False)

    print(f"\n✅ Tournament finished! Results successfully saved to '{output_file}'.")

if __name__ == "__main__":
    main()
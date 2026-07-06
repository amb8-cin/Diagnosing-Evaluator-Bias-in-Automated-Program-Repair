import pandas as pd
import numpy as np

# 1. Load your dataset
df = pd.read_csv('./data_bases/04_final/synthetic_chatgpt_dataset.csv', sep=';')

# 2. Create the Unique Pair ID (Ex: Case_003_AnkiDroid_SYNTHETIC_1)
# Let's just remove the final 4 or 3 letter suffix (_BUG or _FIX)
df['Pair_ID'] = df['Case_ID'].apply(lambda x: x.rsplit('_', 1)[0])

# 3. Get the list of all unique pairs (should be around 1007 pairs)
all_pairs = df['Pair_ID'].unique()
print(f"Total pairs (bug/fix) found: {len(all_pairs)}")

# 4. Randomly select 300 pair IDs for the LLMs tournament
np.random.seed(42) # Ensures the selection is always the same when running the script
tournament_pairs = np.random.choice(all_pairs, size=300, replace=False)

# 5. Create the dataset for the Orchestrator (LLMs only receive the BUG from these 300)
# We filter where Pair_ID is in the selection AND where the resource indicates an error (1)
df_llm = df[(df['Pair_ID'].isin(tournament_pairs)) & (df['Has_Resource_Leak'] == 1)].copy()
df_llm = df_llm.rename(columns={'Code_Snippet': 'code_with_bug'})

# 6. Create the dataset for the Validator Training (Everything NOT selected for the tournament)
# This removes both the BUG and the FIX from the 300 LLM cases
df_validator_training = df[~df['Pair_ID'].isin(tournament_pairs)].copy()

# Remove the auxiliary column before saving
df_llm = df_llm.drop(columns=['Pair_ID'])
df_validator_training = df_validator_training.drop(columns=['Pair_ID'])

# 7. Save the files
df_llm.to_csv('./data_bases/04_final/holdout_300_bugs_llm.csv', index=False)
df_validator_training.to_csv('./data_bases/04_final/validator_training_final.csv', index=False, sep=';')

print(f"--- Division Report ---")
print(f"✅ File for LLMs: {len(df_llm)} lines (only bug codes)")
print(f"✅ File for Validator: {len(df_validator_training)} lines (bug/fix pairs for training)")
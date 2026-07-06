import os
import difflib

def generate_dataset_diffs(base_directory):
    print("Starting Diff generation for DroidLeaks...\n")
    
    # Lists all folders within the dataset
    folders = [p for p in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, p))]
    
    processed_cases = 0
    errors = 0

    for folder in folders:
        folder_path = os.path.join(base_directory, folder)
        buggy_file = os.path.join(folder_path, "BuggyCode.java")
        fix_file = os.path.join(folder_path, "GroundTruth_Fix.java")
        output_diff = os.path.join(folder_path, "code_difference.diff")

        # Only process if the folder contains both Java files
        if os.path.exists(buggy_file) and os.path.exists(fix_file):
            try:
                # Read the files (using errors='ignore' to avoid issues with old characters)
                with open(buggy_file, 'r', encoding='utf-8', errors='ignore') as f_buggy:
                    buggy_lines = f_buggy.readlines()
                    
                with open(fix_file, 'r', encoding='utf-8', errors='ignore') as f_fix:
                    fix_lines = f_fix.readlines()

                # Generate the Diff (with 3 lines of context above and below the error)
                diff = difflib.unified_diff(
                    buggy_lines, 
                    fix_lines, 
                    fromfile='BuggyCode.java (With Error)', 
                    tofile='GroundTruth_Fix.java (Fixed)',
                    n=3  # Change this number if you want more or fewer context lines
                )
                
                diff_lines = list(diff)
                
                # Save the result to the new .diff file
                with open(output_diff, 'w', encoding='utf-8') as f_out:
                    if len(diff_lines) == 0:
                        f_out.write("No differences found. The files are identical.")
                    else:
                        f_out.writelines(diff_lines)
                
                processed_cases += 1
                
            except Exception as e:
                print(f"⚠️ Error processing case {folder}: {e}")
                errors += 1

    print(f"\n✅ Process complete!")
    print(f"📊 Diffs generated successfully: {processed_cases} cases.")
    if errors > 0:
        print(f"❌ Errors found: {errors}")

# ==========================================
# HOW TO RUN
# ==========================================
if __name__ == "__main__":
    # Exact path to your folder with extracted cases
    DATASET_PATH = 'C:\Dissertacao\data_bases\02_extracted'
    
    generate_dataset_diffs(DATASET_PATH)
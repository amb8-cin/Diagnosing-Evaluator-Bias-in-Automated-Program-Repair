import os
import pandas as pd

def consolidate_diffs_to_csv(base_directory, output_file_path):
    print(f"Starting Dataset consolidation for AI...\n")
    
    data_for_csv = []
    
    # Lists only directories (ignoring loose files in the root)
    directories = [p for p in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, p))]
    
    processed_cases = 0

    for directory in directories:
        directory_path = os.path.join(base_directory, directory)
        diff_file_path = os.path.join(directory_path, "code_difference.diff")
        meta_file_path = os.path.join(directory_path, "metadata.txt")

        # Only processes if the diff file exists in this directory
        if os.path.exists(diff_file_path):
            with open(diff_file_path, 'r', encoding='utf-8') as f:
                diff_content = f.read().strip()
            
            # Ignores files where no differences were found or that are empty
            if not diff_content or "No difference found" in diff_content:
                continue
                
            # Initializes metadata variables with default values
            app_name = directory
            buggy_file_path = "Unknown"
            resource_class = "Unknown"
            
            # Extracts context from the metadata.txt file
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'r', encoding='utf-8') as m:
                    meta_lines = m.readlines()
                    for line in meta_lines:
                        if line.startswith("App:"):
                            app_name = line.split("App:")[1].strip()
                        elif line.startswith("Buggy File:"):
                            buggy_file_path = line.split("Buggy File:")[1].strip()
                        elif line.startswith("Resource:"):
                            resource_class = line.split("Resource:")[1].strip()

            # Adds the row (case) to our structured list
            data_for_csv.append({
                "Case_ID": directory,
                "Application": app_name,
                "Resource_Class": resource_class,
                "File_Path": buggy_file_path,
                "Diff_Code": diff_content,
                "Has_Resource_Leak": 1  # 1 indicates the presence of the bug (useful for AI)
            })
            processed_cases += 1

    # Converts to Pandas DataFrame and saves to CSV
    df = pd.DataFrame(data_for_csv)
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig', sep=';') # utf-8-sig helps Excel read special characters correctly
    
    print(f"✅ Process completed successfully!")
    print(f"📊 Final dataset generated with {processed_cases} clean cases.")
    print(f"📁 File saved to: {output_file_path}")

# ==========================================
# HOW TO RUN
# ==========================================
if __name__ == "__main__":
    # The directory where your extracted cases are located
    DATASET_DIRECTORY = 'C:/Dissertacao/data_bases/02_extracted'
    
    # The final file that will be generated (will be outside the repositories folder to avoid mixing)
    FINAL_CSV_FILE_PATH = 'C:/Dissertacao/data_bases/01_raw/extracted_teste.csv'
    
    consolidate_diffs_to_csv(DATASET_DIRECTORY, FINAL_CSV_FILE_PATH)
import pandas as pd
import git
import os
import shutil

# Dictionary with GitHub links for applications
GITHUB_URL_MAP = {
    "AnkiDroid": "https://github.com/ankidroid/Anki-Android.git",
    "AnySoftKeyboard": "https://github.com/AnySoftKeyboard/AnySoftKeyboard.git",
    "Osmand": "https://github.com/osmandapp/OsmAnd.git",
    "SMSDroid": "https://github.com/felixb/smsdroid.git",
    "Google Authenticator": "https://github.com/google/google-authenticator-android.git",
    "Owncloud": "https://github.com/owncloud/android.git",
    "Bankdroid": "https://github.com/liato/android-bankdroid.git",
    "OSMTracker": "https://github.com/labexp/osmtracker-android.git",
    "IRCCloud": "https://github.com/irccloud/android.git",
    "Wordpress": "https://github.com/wordpress-mobile/WordPress-Android.git",
    "ChatSecure": "https://github.com/guardianproject/ChatSecureAndroid.git",
    "Transdroid": "https://github.com/erickok/transdroid.git",
    "CSipSimple": "https://github.com/r3gis3r/CSipSimple.git",
    "APG": "https://github.com/thialfihar/apg.git",
    "FBReaderJ": "https://github.com/geometer/FBReaderJ.git",
    "ConnectBot": "https://github.com/connectbot/connectbot.git",
    "SipDroid": "https://github.com/i-p-tel/sipdroid.git",
    "Ushahidi": "https://github.com/ushahidi/Ushahidi_Android.git",
    "OsmDroid": "https://github.com/osmdroid/osmdroid.git",
    "Surespot": "https://github.com/surespot/android.git",
    "Zxing": "https://github.com/zxing/zxing.git",
    "Cgeo": "https://github.com/cgeo/cgeo.git",
    "K-9 Mail": "https://github.com/k9mail/k-9.git",
    "CallMeter": "https://github.com/felixb/callmeter.git",
    "Open-GPSTracker": "https://github.com/nlbhub/Open-GPSTracker.git",
    "VLC": "https://code.videolan.org/videolan/vlc-android.git",
    "Xabber": "https://github.com/redsolution/xabber-android.git",
    "Quran for Android": "https://github.com/quran/quran_android.git",
    "CycleStreets": "https://github.com/cyclestreets/android.git",
    "Bitcoin-wallet": "https://github.com/schildbach/bitcoin-wallet.git",
    "Terminal": "https://github.com/jackpal/Android-Terminal-Emulator.git",
    "Hacker News": "https://github.com/manmal/hn-android.git"
}

def find_file_in_subfolders(root_folder, file_name):
    """Searches all subfolders until the desired file is found"""
    target_name = os.path.basename(file_name)
    
    for root, dirs, files in os.walk(root_folder):
        if target_name in files:
            return os.path.join(root, target_name)
            
    return None

def extract_experiment_cases(csv_path, repos_dir, output_dir, log_path):
    print(f"Starting extraction... The error report will be saved to: {log_path}")
    
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the log file to write errors
    with open(log_path, 'w', encoding='utf-8') as log_file:
        
        def log_failure(message):
            # Writes the failure to the log file
            log_file.write(message + '\n')
        
        try:
            df = pd.read_excel(csv_path)
        except Exception as e:
            msg = f"Fatal error reading Excel file: {e}"
            print(msg)
            log_failure(msg)
            return

        valid_cases = df[df['Use in experiments?'] == 'yes']

        for index, row in valid_cases.iterrows():
            app_name = str(row['App name']).strip()
            bug_hash = str(row['Buggy revision']).strip()
            fix_hash = str(row['Fix revision']).strip()
            raw_file_path = str(row['Buggy file']).strip()
            resource_class = str(row['Concerned Class']).strip()
            
            # Cleaning up file name
            if '.java' in raw_file_path:
                file_path = raw_file_path.split('.java')[0] + '.java'
            else:
                file_path = raw_file_path
            
            if pd.isna(app_name) or pd.isna(bug_hash) or pd.isna(raw_file_path) or app_name == 'nan':
                continue

            # Shows only progress in the terminal
            print(f"Processing Case {index}: {app_name}...")
            
            error_prefix = f"[Case {index} - {app_name}]"
            
            if app_name not in GITHUB_URL_MAP:
                log_failure(f"{error_prefix} ⚠️ Warning: GitHub link for application not found. Skipping.")
                continue
                
            repo_url = GITHUB_URL_MAP[app_name]
            repo_path = os.path.join(repos_dir, app_name)
            case_output = os.path.join(output_dir, f"Case_{index:03d}_{app_name}")
            
            # 1. CLONE THE REPOSITORY
            if not os.path.exists(repo_path):
                try:
                    git.Repo.clone_from(repo_url, repo_path)
                except Exception as e:
                    log_failure(f"{error_prefix} ❌ Error cloning repository: {e}")
                    continue

            # 2. EXTRACT FILES
            os.makedirs(case_output, exist_ok=True)
            with open(os.path.join(case_output, "metadata.txt"), "w") as meta:
                meta.write(f"App: {app_name}\nResource: {resource_class}\nBuggy File: {file_path}\n")

            try:
                repo = git.Repo(repo_path)
                
                # Extract Buggy
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                repo.git.checkout(bug_hash)
                
                actual_buggy_path = find_file_in_subfolders(repo_path, file_path)
                if actual_buggy_path:
                    shutil.copy2(actual_buggy_path, os.path.join(case_output, "BuggyCode.java"))
                else:
                    log_failure(f"{error_prefix} ⚠️ Buggy file not found in any subfolder: {file_path}")
                
                # Extract Fix
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                repo.git.checkout(fix_hash)
                
                actual_fix_path = find_file_in_subfolders(repo_path, file_path)
                if actual_fix_path:
                    shutil.copy2(actual_fix_path, os.path.join(case_output, "GroundTruth_Fix.java"))
                else:
                    log_failure(f"{error_prefix} ⚠️ Fix file not found in any subfolder: {file_path}")
                        
            except git.exc.GitCommandError as e:
                log_failure(f"{error_prefix} ❌ Git error (possibly incorrect hash): {e}")
            except Exception as e:
                log_failure(f"{error_prefix} ❌ Unexpected error while processing: {e}")

# ==========================================
# HOW TO RUN - DROIDLEAKS
# ==========================================
if __name__ == "__main__":
    CSV_FILE = 'C:/Dissertacao/data_bases/01_raw/droidleaks.xlsx'
    REPOSITORIES_FOLDER = 'C:/Dissertacao/data_bases/01_raw/repositorios_droidleaks/'
    EXPERIMENT_FOLDER = 'C:/Dissertacao/data_bases/02_extracted'
    
    # Path to the new LOG file
    LOG_FILE = 'C:/Users/andrezza.bonfim/Documents/docs meus/Mestrado/dissertação/repositorios_droidleaks/relatorio_falhas.txt'
    
    extract_experiment_cases(CSV_FILE, REPOSITORIES_FOLDER, EXPERIMENT_FOLDER, LOG_FILE)
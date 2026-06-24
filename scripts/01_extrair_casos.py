import pandas as pd
import git
import os
import shutil

# Dicionário com os links do GitHub das aplicações
MAPA_URLS_GITHUB = {
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

def procurar_ficheiro_nas_subpastas(pasta_raiz, nome_ficheiro):
    """Vasculha todas as subpastas até encontrar o ficheiro desejado"""
    nome_alvo = os.path.basename(nome_ficheiro)
    
    for root, dirs, files in os.walk(pasta_raiz):
        if nome_alvo in files:
            return os.path.join(root, nome_alvo)
            
    return None

def extrair_casos_para_experimento(csv_path, repos_dir, output_dir, log_path):
    print(f"A iniciar a extração... O relatório de erros será guardado em: {log_path}")
    
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Abrimos o arquivo de log para escrever os erros
    with open(log_path, 'w', encoding='utf-8') as log_file:
        
        def registrar_falha(mensagem):
            # Escreve a falha no arquivo de log
            log_file.write(mensagem + '\n')
        
        try:
            df = pd.read_excel(csv_path)
        except Exception as e:
            msg = f"Erro fatal ao ler o ficheiro Excel: {e}"
            print(msg)
            registrar_falha(msg)
            return

        casos_validos = df[df['Use in experiments?'] == 'yes']

        for index, row in casos_validos.iterrows():
            app_name = str(row['App name']).strip()
            bug_hash = str(row['Buggy revision']).strip()
            fix_hash = str(row['Fix revision']).strip()
            file_path_bruto = str(row['Buggy file']).strip()
            resource_class = str(row['Concerned Class']).strip()
            
            # Limpeza do nome do ficheiro
            if '.java' in file_path_bruto:
                file_path = file_path_bruto.split('.java')[0] + '.java'
            else:
                file_path = file_path_bruto
            
            if pd.isna(app_name) or pd.isna(bug_hash) or pd.isna(file_path_bruto) or app_name == 'nan':
                continue

            # Mostra no terminal apenas o progresso
            print(f"Processando Caso {index}: {app_name}...")
            
            prefixo_erro = f"[Caso {index} - {app_name}]"
            
            if app_name not in MAPA_URLS_GITHUB:
                registrar_falha(f"{prefixo_erro} ⚠️ Aviso: Não tenho o link do GitHub para a aplicação. A saltar.")
                continue
                
            repo_url = MAPA_URLS_GITHUB[app_name]
            repo_path = os.path.join(repos_dir, app_name)
            case_output = os.path.join(output_dir, f"Case_{index:03d}_{app_name}")
            
            # 1. CLONAR O REPOSITÓRIO
            if not os.path.exists(repo_path):
                try:
                    git.Repo.clone_from(repo_url, repo_path)
                except Exception as e:
                    registrar_falha(f"{prefixo_erro} ❌ Erro ao clonar o repositório: {e}")
                    continue

            # 2. EXTRAIR OS FICHEIROS
            os.makedirs(case_output, exist_ok=True)
            with open(os.path.join(case_output, "metadata.txt"), "w") as meta:
                meta.write(f"App: {app_name}\nResource: {resource_class}\nBuggy File: {file_path}\n")

            try:
                repo = git.Repo(repo_path)
                
                # Extrair Buggy
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                repo.git.checkout(bug_hash)
                
                caminho_real_buggy = procurar_ficheiro_nas_subpastas(repo_path, file_path)
                if caminho_real_buggy:
                    shutil.copy2(caminho_real_buggy, os.path.join(case_output, "BuggyCode.java"))
                else:
                    registrar_falha(f"{prefixo_erro} ⚠️ Ficheiro Buggy não encontrado em nenhuma subpasta: {file_path}")
                
                # Extrair Fix
                repo.git.reset('--hard')
                repo.git.clean('-fd')
                repo.git.checkout(fix_hash)
                
                caminho_real_fix = procurar_ficheiro_nas_subpastas(repo_path, file_path)
                if caminho_real_fix:
                    shutil.copy2(caminho_real_fix, os.path.join(case_output, "GroundTruth_Fix.java"))
                else:
                    registrar_falha(f"{prefixo_erro} ⚠️ Ficheiro Fix não encontrado em nenhuma subpasta: {file_path}")
                        
            except git.exc.GitCommandError as e:
                registrar_falha(f"{prefixo_erro} ❌ Erro do Git (hash possivelmente incorreto): {e}")
            except Exception as e:
                registrar_falha(f"{prefixo_erro} ❌ Erro inesperado ao processar: {e}")

# ==========================================
# COMO EXECUTAR - DROIDLEAKS
# ==========================================
if __name__ == "__main__":
    ARQUIVO_CSV = 'C:/Dissertacao/data_bases/01_raw/droidleaks.xlsx'
    PASTA_REPOSITORIOS = 'C:/Dissertacao/data_bases/01_raw/repositorios_droidleaks'
    PASTA_EXPERIMENTO = 'C:/Dissertacao/data_bases/02_extracted'
    
    # Caminho do novo arquivo de LOG
    ARQUIVO_LOG = 'C:/Users/andrezza.bonfim/Documents/docs meus/Mestrado/dissertação/repositorios_droidleaks/relatorio_falhas.txt'
    
    extrair_casos_para_experimento(ARQUIVO_CSV, PASTA_REPOSITORIOS, PASTA_EXPERIMENTO, ARQUIVO_LOG)
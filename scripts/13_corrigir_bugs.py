import pandas as pd
import time
import os
from dotenv import load_dotenv

# Importação das bibliotecas oficiais
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# 1. CARREGAR VARIÁVEIS DE AMBIENTE
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. INICIALIZAÇÃO DOS CLIENTES
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
cliente_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 3. FUNÇÕES DE CHAMADA PARA CADA MODELO
# ==========================================

def pedir_correcao_gpt(prompt):
    try:
        resposta = cliente_openai.chat.completions.create(
            model="gpt-5.4", # Atualize para o modelo exato que vai usar
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"ERRO GPT: {e}"

def pedir_correcao_claude(prompt):
    try:
        resposta = cliente_anthropic.messages.create(
            model="claude-opus-4-6", # Nome exato e estável do modelo
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return resposta.content[0].text
    except Exception as e:
        return f"ERRO CLAUDE: {e}"

def pedir_correcao_gemini(prompt):
    try:
        # Adicionado o sufixo '-latest' que é o padrão reconhecido pela API
        modelo = genai.GenerativeModel('gemini-3.1-pro-preview') 
        resposta = modelo.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        return resposta.text
    except Exception as e:
        return f"ERRO GEMINI: {e}"

# ==========================================
# 4. PIPELINE PRINCIPAL (O Torneio)
# ==========================================
def main():
    print("🚀 Iniciando o Torneio das LLMs...")
    
    arquivo_input = './data_bases/holdout_300_bugs_llm.csv'
    arquivo_output = './data_bases/resultados_torneio_llms.csv'
    
    if not os.path.exists(arquivo_input):
        print(f"❌ Erro: O arquivo {arquivo_input} não foi encontrado!")
        return

    # ========================================================
    # LÓGICA DE RETOMADA (CHECKPOINT)
    # ========================================================
    if os.path.exists(arquivo_output):
        df = pd.read_csv(arquivo_output)
        print("📁 Arquivo parcial encontrado. Avaliando o que já foi processado...")
    else:
        df = pd.read_csv(arquivo_input)
        print("🚀 Iniciando processamento do zero...")
        # Cria as colunas caso não existam
        for col in ['fix_gpt', 'fix_claude', 'fix_gemini']:
            if col not in df.columns:
                df[col] = ""

    # Pega os 300 casos
    df = df.head(300) 

    casos_ignorados = 0

    # Iterar sobre cada bug
    for index, row in df.iterrows():
        
        # Verificação de segurança: checa se as 3 respostas existem e não são erro
        gpt_ok = pd.notna(row.get('fix_gpt')) and str(row.get('fix_gpt')).strip() != "" and not str(row.get('fix_gpt')).startswith("ERRO")
        claude_ok = pd.notna(row.get('fix_claude')) and str(row.get('fix_claude')).strip() != "" and not str(row.get('fix_claude')).startswith("ERRO")
        gemini_ok = pd.notna(row.get('fix_gemini')) and str(row.get('fix_gemini')).strip() != "" and not str(row.get('fix_gemini')).startswith("ERRO")

        if gpt_ok and claude_ok and gemini_ok:
            casos_ignorados += 1
            continue # Pula para a próxima iteração silenciosamente
            
        # Se chegou aqui, é porque falta processar (vai começar do 46, por exemplo)
        if casos_ignorados > 0 and casos_ignorados == index:
            print(f"⏩ Pulando {casos_ignorados} casos já processados com sucesso...")
            
        print(f"🔄 Processando bug {index + 1}/{len(df)}...")
        
        codigo_bug = row['codigo_com_bug'] # Confirme se o nome da coluna é esse mesmo
        
        prompt_completo = f"""
        Você é um engenheiro de software especialista em Java. 
        Corrija o Resource Leak no código abaixo.
        REGRA: Retorne APENAS o código corrigido. 
        Sem explicações, sem markdown (```java), sem saudações.
        
        Código:
        {codigo_bug}
        """
        
        # Execução das chamadas (Verificando se está vazio ou se deu ERRO antes de chamar)
        if not gpt_ok:
            df.at[index, 'fix_gpt'] = pedir_correcao_gpt(prompt_completo)
            time.sleep(1) 
        
        if not claude_ok:
            df.at[index, 'fix_claude'] = pedir_correcao_claude(prompt_completo)
            time.sleep(2)
        
        if not gemini_ok:
            df.at[index, 'fix_gemini'] = pedir_correcao_gemini(prompt_completo)
            time.sleep(3)
        
        # Salva o progresso parcial a cada linha concluída
        df.to_csv(arquivo_output, index=False)

    print(f"\n✅ Torneio finalizado! Resultados salvos com sucesso em '{arquivo_output}'.")

if __name__ == "__main__":
    main()
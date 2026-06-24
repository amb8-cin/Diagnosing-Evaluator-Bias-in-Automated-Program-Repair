import pandas as pd
import time
import os
from dotenv import load_dotenv

# Importação das bibliotecas oficiais
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# ==========================================
# 1. CARREGAR VARIÁVEIS DE AMBIENTE
# ==========================================
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# 2. INICIALIZAÇÃO DOS CLIENTES
# ==========================================
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
    print("🚀 Retomando o Torneio das LLMs para os casos pendentes...")
    
    # Lendo e salvando no mesmo arquivo unificado
    arquivo_csv = 'C:/Dissertacao/data_bases/ast/resultados_torneio_llms_FILTRADO_AST.csv'
    arquivo_final = 'C:/Dissertacao/data_bases/ast/resultados_torneio_llms_ast.csv'
    
    if not os.path.exists(arquivo_csv):
        print(f"❌ Erro: O arquivo {arquivo_csv} não foi encontrado!")
        return

    df = pd.read_csv(arquivo_csv)
    print(f"📊 Arquivo carregado com {len(df)} instâncias.")

    casos_ignorados = 0

    # Iterar sobre cada bug
    for index, row in df.iterrows():
        
        # LÓGICA ATUALIZADA: Precisa rodar se estiver PENDENTE DE EXECUÇÃO ou se for um ERRO de API anterior
        precisa_gpt = str(row.get('fix_gpt')).strip() == "PENDENTE DE EXECUÇÃO" or str(row.get('fix_gpt')).startswith("ERRO")
        precisa_claude = str(row.get('fix_claude')).strip() == "PENDENTE DE EXECUÇÃO" or str(row.get('fix_claude')).startswith("ERRO")
        precisa_gemini = str(row.get('fix_gemini')).strip() == "PENDENTE DE EXECUÇÃO" or str(row.get('fix_gemini')).startswith("ERRO")

        # Se nenhum deles precisa rodar, pula o caso
        if not precisa_gpt and not precisa_claude and not precisa_gemini:
            casos_ignorados += 1
            continue 
            
        if casos_ignorados > 0 and casos_ignorados == index:
            print(f"⏩ Pulando {casos_ignorados} casos já resolvidos...")
            
        print(f"🔄 Processando bug ID: {row['ID_Caso']} (Linha {index + 1}/{len(df)})...")
        
        codigo_bug = row['codigo_com_bug'] 
        
        prompt_completo = f"""
        Você é um engenheiro de software especialista em Java. 
        Corrija o Resource Leak no código abaixo.
        REGRA: Retorne APENAS o código corrigido. 
        Sem explicações, sem markdown (```java), sem saudações.
        
        Código:
        {codigo_bug}
        """
        
        # Execução das chamadas (Apenas para quem realmente precisa)
        if precisa_gpt:
            print("   -> Solicitando GPT...")
            df.at[index, 'fix_gpt'] = pedir_correcao_gpt(prompt_completo)
            time.sleep(1) 
        
        if precisa_claude:
            print("   -> Solicitando Claude...")
            df.at[index, 'fix_claude'] = pedir_correcao_claude(prompt_completo)
            time.sleep(2)
        
        if precisa_gemini:
            print("   -> Solicitando Gemini...")
            df.at[index, 'fix_gemini'] = pedir_correcao_gemini(prompt_completo)
            time.sleep(3)
            
        # Atualiza o status para você saber que este não está mais pendente
        if df.at[index, 'Status_Execucao'] == "FALTA RODAR - Enviar para as LLMs":
             df.at[index, 'Status_Execucao'] = "RESOLVIDO NESTA EXECUÇÃO"
        
        # Salva o progresso parcial a cada linha concluída (Segurança contra quedas)
        df.to_csv(arquivo_csv, index=False, encoding='utf-8')

    print(f"\n✅ Execução finalizada! Resultados salvos e atualizados em '{arquivo_csv}'.")

if __name__ == "__main__":
    main()
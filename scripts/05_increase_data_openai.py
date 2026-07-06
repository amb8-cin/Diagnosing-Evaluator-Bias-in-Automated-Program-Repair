import pandas as pd
from openai import OpenAI
import json
import time
import os

# 1. Configurar o Cliente da OpenAI
CHAVE_API = ""
client = OpenAI(api_key=CHAVE_API)

def gerar_sinteticos_com_chatgpt(codigo_bug, codigo_fix, classe_recurso):
    """Envia o prompt para o ChatGPT (gpt-4o-mini) com sistema de Retry"""
    
    prompt = f"""
    Atue como um Engenheiro de Software Android Especialista. Estou a construir um dataset para treinar um modelo de Machine Learning focado em detetar falhas de 'Resource Leaks' no ecossistema Android, especificamente envolvendo a classe: {classe_recurso}.
    
    Aqui está um exemplo real de um código com fuga de recurso (Buggy) e a sua respetiva correção (Fix):
    
    BUG:
    {codigo_bug}
    
    FIX:
    {codigo_fix}
    
    A sua tarefa:
    Gere 10 novos exemplos sintéticos baseados neste exato mesmo tipo de falha, aplicados em contextos de aplicações Android diferentes (ex: app de música, scanner, fitness, e outros que você considerem que possam ser aplicados).
    
    ATENÇÃO: Retorne a resposta ESTRITAMENTE num formato JSON válido. O JSON DEVE ser um objeto contendo uma única chave chamada "exemplos", que será uma lista de objetos.
    
    Formato exigido:
    {{
      "exemplos": [
        {{
          "contexto": "App de Musica",
          "buggy_code": "codigo java do bug aqui",
          "fix_code": "codigo java corrigido aqui"
        }}
      ]
    }}
    """
    
    # SISTEMA DE AUTO-RETRY (Tenta 3 vezes se houver falha de rede)
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            # Pedido à API do ChatGPT
            # Pedido à API do ChatGPT
            response = client.chat.completions.create(
                model="gpt-5.4-mini", # <--- ALTERE AQUI!
                response_format={ "type": "json_object" }, 
                messages=[
                    {"role": "system", "content": "És um assistente especializado em código que gera saídas estritamente em formato JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extrai o texto da resposta
            resposta_texto = response.choices[0].message.content
            dados_json = json.loads(resposta_texto)
            
            # Devolve apenas a lista de exemplos gerada
            return dados_json.get("exemplos", [])
            
        except Exception as e:
            print(f"    ⚠️ Falha na tentativa {tentativa + 1} de {max_tentativas}: {e}")
            if tentativa < max_tentativas - 1:
                print("    ⏳ A aguardar 5 segundos antes de tentar novamente...")
                time.sleep(5)
            else:
                print("    ❌ Erro definitivo neste caso. Vamos saltar.")
                return []

def main():
    print("🚀 A iniciar o Aumento de Dados com o ChatGPT (MODO BLINDADO)...\n")
    
    ARQUIVO_SAIDA = './data_bases/04_final/dataset_synthetic_chatgpt.csv'
    
    # 1. Carregar os dados originais
    df = pd.read_csv('.data_bases/03_processed/balanced_dataset.csv', sep=';')
    
    casos_ja_processados = set()
    novos_dados_sinteticos = []
    
    # 2. AUTO-RESUME: Verificar se já temos progresso guardado
    if os.path.exists(ARQUIVO_SAIDA):
        print("📂 Ficheiro anterior encontrado! A retomar de onde paramos...")
        df_existente = pd.read_csv(ARQUIVO_SAIDA, sep=';')
        novos_dados_sinteticos = df_existente.to_dict('records')
        
        for idx, row in df_existente.iterrows():
            id_base = str(row['ID_Caso']).split('_SINTETICO')[0]
            casos_ja_processados.add(id_base)
            
        print(f"✅ Já temos {len(casos_ja_processados)} casos processados e guardados.\n")

    # 3. Agrupar os pares de BUG e FIX
    casos_reais = {}
    for index, row in df.iterrows():
        id_base = str(row['ID_Caso']).replace('_BUG', '').replace('_FIX', '')
        
        if id_base not in casos_reais:
            casos_reais[id_base] = {'recurso': row['Classe_Recurso']}
            
        if '_BUG' in str(row['ID_Caso']):
            casos_reais[id_base]['bug'] = row['Codigo_Snippet']
        elif '_FIX' in str(row['ID_Caso']):
            casos_reais[id_base]['fix'] = row['Codigo_Snippet']

    # 4. Processar todos os casos que faltam
    for id_base, dados in casos_reais.items():
        # Se já foi processado antes, salta logo!
        if id_base in casos_ja_processados:
            continue
            
        if 'bug' in dados and 'fix' in dados:
            print(f"A gerar sintéticos para o caso: {id_base}...")
            
            sinteticos = gerar_sinteticos_com_chatgpt(
                codigo_bug=dados['bug'], 
                codigo_fix=dados['fix'],
                classe_recurso=dados['recurso']
            )
            
            if not sinteticos:
                continue # Se falhou, salta este caso
                
            for i, sint in enumerate(sinteticos):
                id_sintetico_base = f"{id_base}_SINTETICO_{i+1}"
                
                novos_dados_sinteticos.append({
                    "ID_Caso": f"{id_sintetico_base}_BUG",
                    "Aplicacao": sint.get('contexto', 'Sintetico'),
                    "Classe_Recurso": dados['recurso'],
                    "Codigo_Snippet": sint.get('buggy_code', ''),
                    "Tem_Fuga_de_Recurso": 1
                })
                
                novos_dados_sinteticos.append({
                    "ID_Caso": f"{id_sintetico_base}_FIX",
                    "Aplicacao": sint.get('contexto', 'Sintetico'),
                    "Classe_Recurso": dados['recurso'],
                    "Codigo_Snippet": sint.get('fix_code', ''),
                    "Tem_Fuga_de_Recurso": 0
                })
                
            # AUTO-SAVE: Grava o ficheiro imediatamente a cada caso terminado!
            df_sintetico = pd.DataFrame(novos_dados_sinteticos)
            df_sintetico.to_csv(ARQUIVO_SAIDA, index=False, sep=';', encoding='utf-8-sig')
            
            # Como a API do ChatGPT é muito mais tolerante com os limites (se tiver saldo),
            # podemos usar uma pausa pequena de apenas 1 a 2 segundos!
            time.sleep(2) 

    print("\n🎉 Processo totalmente concluído com o ChatGPT!")
    print(f"📁 Dataset final guardado em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

# 1. Carregar o seu dataset
df = pd.read_csv('./data_bases/04_final/dataset_sintetico_chatgpt.csv', sep=';')

# 2. Criar o ID Único do Par (Ex: Case_003_AnkiDroid_SINTETICO_1)
# Vamos apenas remover o sufixo final de 4 ou 3 letras (_BUG ou _FIX)
df['ID_Par'] = df['ID_Caso'].apply(lambda x: x.rsplit('_', 1)[0])

# 3. Pegar a lista de todos os pares únicos (deve dar em torno de 1007 pares)
todos_os_pares = df['ID_Par'].unique()
print(f"Total de pares (bug/fix) encontrados: {len(todos_os_pares)}")

# 4. Sortear 300 IDs de pares para o torneio das LLMs
np.random.seed(42) # Garante que o sorteio seja sempre o mesmo ao rodar o script
pares_torneio = np.random.choice(todos_os_pares, size=300, replace=False)

# 5. Criar o dataset para o Orquestrador (As LLMs recebem apenas o BUG desses 300)
# Filtramos onde o ID_Par está no sorteio E onde o recurso indica erro (1)
df_llm = df[(df['ID_Par'].isin(pares_torneio)) & (df['Tem_Fuga_de_Recurso'] == 1)].copy()
df_llm = df_llm.rename(columns={'Codigo_Snippet': 'codigo_com_bug'})

# 6. Criar o dataset para o Treino do Validador (Tudo que NÃO foi sorteado para o torneio)
# Isso remove tanto o BUG quanto o FIX dos 300 casos das LLMs
df_treino_validador = df[~df['ID_Par'].isin(pares_torneio)].copy()

# Remover a coluna auxiliar antes de salvar
df_llm = df_llm.drop(columns=['ID_Par'])
df_treino_validador = df_treino_validador.drop(columns=['ID_Par'])

# 7. Salvar os arquivos
df_llm.to_csv('./data_bases/04_final/holdout_300_bugs_llm.csv', index=False)
df_treino_validador.to_csv('./data_bases/04_final/treino_validador_final.csv', index=False, sep=';')

print(f"--- Relatório de Divisão ---")
print(f"✅ Arquivo para LLMs: {len(df_llm)} linhas (apenas os códigos com bug)")
print(f"✅ Arquivo para Validador: {len(df_treino_validador)} linhas (pares bug/fix para treino)")
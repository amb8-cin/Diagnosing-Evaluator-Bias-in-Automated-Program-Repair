import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. Movemos a função para fora para o Python sempre encontrá-la
def extrair_embedding(codigo, tokenizer, model, device):
    inputs = tokenizer(codigo, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Pegamos o CLS token (representação global do código)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()

def extrair_salvar_embeddings():
    print("🚀 Iniciando a Extração de Embeddings (GraphCodeBERT)...")

    nome_modelo = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModel.from_pretrained(nome_modelo)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. IMPORTANTE: Usando o arquivo de treino filtrado
    caminho_dados = 'C:/Dissertacao/data_bases/treino_validador_final.csv' 
    
    try:
        df = pd.read_csv(caminho_dados, sep=';')
    except:
        df = pd.read_csv(caminho_dados)

    # Criando o ID_Par para evitar o Data Leakage no treinador depois
    if 'ID_Caso' in df.columns:
        df['ID_Par'] = df['ID_Caso'].apply(lambda x: x.rsplit('_', 1)[0])

    # Garantindo que o nome da coluna de código esteja correto
    coluna_codigo = 'Codigo_Snippet' if 'Codigo_Snippet' in df.columns else 'codigo_com_bug'
    
    df = df.dropna(subset=[coluna_codigo, 'Tem_Fuga_de_Recurso'])
    codigos = df[coluna_codigo].astype(str).tolist()
    labels = df['Tem_Fuga_de_Recurso'].astype(int).tolist()
    ids_par = df['ID_Par'].tolist()

    print(f"📊 Processando {len(codigos)} códigos...")

    X_embeddings = []
    for i, codigo in enumerate(codigos):
        if i % 100 == 0 and i > 0:
            print(f"   -> Extraídos {i}/{len(codigos)}...")
        
        # Chamando a função que agora está lá em cima
        embedding = extrair_embedding(codigo, tokenizer, model, device)
        X_embeddings.append(embedding)
        
    print("💾 Salvando as características neurais e IDs em CSV...")
    
    df_embeddings = pd.DataFrame(X_embeddings)
    df_embeddings.columns = [f"dim_{i}" for i in range(768)]
    df_embeddings['Tem_Fuga_de_Recurso'] = labels
    df_embeddings['ID_Par'] = ids_par # Salva o ID para o GroupShuffleSplit
    
    caminho_saida = 'C:/Dissertacao/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    df_embeddings.to_csv(caminho_saida, index=False)
    print(f"✅ Arquivo salvo com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    extrair_salvar_embeddings()
import pandas as pd
import joblib
import re

# Função de limpeza (tem de ser idêntica à usada no treino)
def limpar_codigo_java(codigo):
    if not isinstance(codigo, str): return ""
    codigo = re.sub(r'//.*', '', codigo)
    codigo = re.sub(r'/\*.*?\*/', '', codigo, flags=re.DOTALL)
    codigo = re.sub(r'\s+', ' ', codigo)
    return codigo.strip().lower()

def main():
    print("⚖️ A carregar o Juiz (Modelo Random Forest)...")
    modelo = joblib.load('./trained_models/validador_resource_leak.pkl')
    vectorizer = joblib.load('./trained_models/vectorizer.pkl')

    # Carregar os resultados do torneio
    df = pd.read_csv('./data_bases/05_results/resultados_torneio_llms.csv') # ou o nome da sua planilha
    
    modelos_llm = ['fix_gpt', 'fix_claude', 'fix_gemini']
    vitorias = {model: 0 for model in modelos_llm}

    print("\n🧐 Analisando as correções...")

    for llm in modelos_llm:
        # 1. Limpar o código gerado pela LLM
        codigos_limpos = df[llm].apply(limpar_codigo_java)
        
        # 2. Vetorizar
        X_vetorizado = vectorizer.transform(codigos_limpos)
        
        # 3. Pedir o veredito ao Juiz (0 = Seguro, 1 = Com Bug)
        previsoes = modelo.predict(X_vetorizado)
        
        # Guardar o resultado na planilha para conferência manual depois
        df[f'veredito_{llm}'] = previsoes
        
        # Contar quantos códigos o Juiz considerou "Seguros" (Classe 0)
        casos_seguros = (previsoes == 0).sum()
        vitorias[llm] = casos_seguros

    # Exibir o Placar Final
    print("\n" + "="*30)
    print("🏆 PLACAR FINAL DO TORNEIO")
    print("="*30)
    total = len(df)
    for llm, ganhos in vitorias.items():
        taxa = (ganhos / total) * 100
        print(f"{llm.upper()}: {ganhos}/{total} bugs corrigidos ({taxa:.2f}%)")
    print("="*30)

    # Salvar planilha final com os vereditos
    df.to_csv('./data_bases/05_results/relatorio_final_dissertacao.csv', index=False, sep=';')
    print("\n✅ Relatório detalhado salvo em 'relatorio_final_dissertacao.csv'")

if __name__ == "__main__":
    main()
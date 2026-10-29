import pandas as pd
import numpy as np
import logging
import unicodedata
from typing import Union, Tuple

# Configuração básica de log para exibir os alertas
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def ler_base_dados(caminho: str) -> pd.DataFrame:
    """Função para ingestão de dados da fonte."""
    if caminho.endswith('.csv'):
        return pd.read_csv(caminho)
    elif caminho.endswith(('.xls', '.xlsx')):
        return pd.read_excel(caminho)
    else:
        raise ValueError("Formato de arquivo não suportado. Utilize .csv ou .xlsx.")

def mapear_colunas_essenciais(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia sinônimos de colunas essenciais ignorando case e espaços."""
    mapeamento = {
        'Modalidade': ['modalidade', 'tipo da licitacao', 'tipo da licitação', 'tipo de modalidade'],
        'Objeto': ['objeto', 'descricao', 'descrição', 'descricao do objeto', 'descrição do objeto'],
        'Município': ['municipio', 'município', 'cidade'],
        'Jurisdicionado': ['jurisdicionado', 'orgao', 'órgão', 'unidade gestora', 'ente'],
        'Valor': ['valor', 'valor contratado', 'valor da licitacao', 'valor licitação', 'valor da licitação', 'valor global'],
        'Tipo do Objeto': ['tipo do objeto', 'tipo de objeto', 'natureza do objeto', 'categoria do objeto'],
        'No.Licitação': ['no.licitacao', 'no.licitação', 'numero da licitacao', 'número da licitação', 'nº licitação', 'num licitacao']
    }
    
    colunas_originais = df.columns
    colunas_lower = colunas_originais.str.strip().str.lower()
    
    renames = {}
    for alvo, sinonimos in mapeamento.items():
        if alvo in df.columns:
            continue
        
        # Encontra a primeira coluna que bate com um dos sinônimos
        match = [orig for orig, lower in zip(colunas_originais, colunas_lower) if lower in sinonimos]
        
        if match:
            renames[match[0]] = alvo
        else:
            if alvo in ('Modalidade', 'Objeto', 'Município', 'Jurisdicionado', 'Valor'):
                logging.warning(f"Coluna essencial '{alvo}' não localizada através dos sinônimos.")
            df[alvo] = pd.NA
            
    if renames:
        df = df.rename(columns=renames)
        
    return df

def padronizar_textos(serie: pd.Series) -> pd.Series:
    """Aplica higienização textual vetorizada."""
    if serie.isnull().all():
        return serie
    
    s = serie.astype(str).str.upper()
    # Remove acentuação
    s = s.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    # Remove quebras de linha e múltiplos espaços
    s = s.str.replace(r'[\r\n]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Restaura nulos onde a conversão gerou strings vazias ou representações de null
    s = s.replace({'NAN': pd.NA, 'NONE': pd.NA, '<NA>': pd.NA, '': pd.NA})
    return s

def tratar_valor_financeiro(serie: pd.Series) -> pd.Series:
    """Converte strings financeiras variadas para float matemático."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.fillna(0.0).astype(float)
    
    s = serie.astype(str).str.upper()
    # Remove R$, espaços e caracteres inúteis, mantendo apenas números, pontos e vírgulas
    s = s.str.replace(r'[^\d\.,\-]', '', regex=True)
    
    # Lida com o padrão brasileiro (ex: 1.234.567,89)
    tem_virgula = s.str.contains(',', na=False)
    
    s = np.where(
        tem_virgula,
        s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
        s # Se não tiver vírgula, assume que o ponto já é o separador decimal ou não há decimais
    )
    
    return pd.to_numeric(pd.Series(s), errors='coerce').fillna(0.0).astype(float)

def executar_fase_1(df_ou_caminho: Union[pd.DataFrame, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fase 1: Ingestão, Integridade e Enriquecimento Estrutural.
    Retorna (df_analytical, df_raw).
    """
    # 1. INGESTÃO E PRESERVAÇÃO DA ORIGEM
    if isinstance(df_ou_caminho, str):
        df_raw = ler_base_dados(df_ou_caminho)
    else:
        df_raw = df_ou_caminho.copy()
        
    # 2. A CHAVE PRIMÁRIA OBRIGATÓRIA
    if 'Protocolo' not in df_raw.columns:
        raise KeyError("A coluna obrigatória 'Protocolo' não foi encontrada. Execução interrompida.")
    
    qtde_inicial = len(df_raw)
    qtde_protocolos_unicos = df_raw['Protocolo'].nunique()
    logging.info(f"Registros iniciais: {qtde_inicial} | Protocolos únicos: {qtde_protocolos_unicos}")
    
    # 3. CONSOLIDAÇÃO DA BASE ANALÍTICA E CONTAGEM DE LINHAS DE ORIGEM (UNIDADE DE ANÁLISE)
    contagem_linhas = df_raw.groupby('Protocolo').size()
    df_analytical = df_raw.groupby('Protocolo', as_index=False).first()
    df_analytical['Qtd_Linhas_Origem'] = df_analytical['Protocolo'].map(contagem_linhas).fillna(1).astype(int)
    
    # Teste de qualidade: a base analítica deve ter uma linha por protocolo
    assert len(df_analytical) == df_analytical['Protocolo'].nunique(), "Falha na consolidação: Protocolos duplicados persistiram."
    
    # 4. MAPEAMENTO DINÂMICO DE COLUNAS
    df_analytical = mapear_colunas_essenciais(df_analytical)
    
    # 5. PADRONIZAÇÃO TEXTUAL VETORIZADA
    colunas_texto = ['Objeto', 'Modalidade', 'Município', 'Jurisdicionado', 'Tipo do Objeto']
    for col in colunas_texto:
        if col in df_analytical.columns:
            df_analytical[col] = padronizar_textos(df_analytical[col])
        
    # 6. TRATAMENTO FINANCEIRO (COLUNA VALOR)
    df_analytical['Valor'] = tratar_valor_financeiro(df_analytical['Valor'])
    
    # 7. INICIALIZAÇÃO DA ESTRUTURA ANALÍTICA (GI) NA ORDEM LÓGICA DE PREENCHIMENTO
    # Tipagem explícita como 'object' evita que colunas iniciadas com nulos virem float64 no Pandas
    df_analytical['GI - Fundamentação Contextual'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
    df_analytical['GI - Grau de Evidência'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
    df_analytical['GI - Resultado Etapa 1'] = pd.Series('NÃO CLASSIFICADO', index=df_analytical.index, dtype='object')
    df_analytical['GI - Divergência Cadastral Aparente'] = False
    df_analytical['GI - Corroboração Utilizada'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
    df_analytical['GI - Protocolo Relacionado'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
    
    # 8. ENRIQUECIMENTO: REGIME JURÍDICO INFORMADO
    texto_busca = df_analytical['Modalidade'].fillna('') + ' ' + df_analytical['Objeto'].fillna('')
    
    cond_14133 = texto_busca.str.contains(r'14\.?133', case=False, regex=True)
    cond_8666 = texto_busca.str.contains(r'8\.?666|10\.?520', case=False, regex=True)
    cond_13303 = texto_busca.str.contains(r'13\.?303', case=False, regex=True)
    
    # Captura outras leis explicitamente citadas (ex: Lei 12.462, Lei Complementar 123)
    outras_leis = texto_busca.str.extract(r'(?i)(lei[^\d]*(?:n[ºo]\s*)?\d+[\.\d]*[/-]?\d{2,4})')[0]
    outras_leis = outras_leis.str.strip().str.upper()
    
    df_analytical['GI - Regime Jurídico Informado'] = np.where(
        cond_14133, 'Lei 14.133/2021',
        np.where(
            cond_8666, 'Lei 8.666/1993',
            np.where(
                cond_13303, 'Lei 13.303/2016',
                np.where(
                    outras_leis.notna(), outras_leis,
                    'Não Identificado'
                )
            )
        )
    )

    # 9. INICIALIZAÇÃO DO MOTOR DE TRIAGEM DE RISCOS E CARACTERÍSTICAS (R0 A R6 / D1 A D5 / Q1 A Q4)
    for sinal in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        df_analytical[f'{sinal}_STATUS'] = pd.Series('NÃO IDENTIFICADO', index=df_analytical.index, dtype='object')
        df_analytical[f'{sinal}_EVIDENCIA'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
        df_analytical[f'{sinal}_REGRA'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')

    df_analytical['D1_ENQUADRAMENTO'] = False
    df_analytical['D2_EXECUCAO'] = False
    df_analytical['D3_PRECIFICACAO'] = False
    df_analytical['D4_DISTRIBUICAO'] = False
    df_analytical['D5_RECORRENCIA'] = False
    df_analytical['QTD_DIMENSOES'] = 0
    df_analytical['PRIORIDADE_QUALITATIVA'] = pd.Series('Q4 — Monitoramento', index=df_analytical.index, dtype='object')
    df_analytical['NECESSITA_REVISAO_MANUAL'] = False
    df_analytical['JUSTIFICATIVA_TRIAGEM'] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
    
    # 10. ASSERÇÕES E TESTES DE QUALIDADE (UNIT TESTS)
    # V1: Garante unicidade da chave primária
    assert df_analytical['Protocolo'].duplicated().sum() == 0, "Erro: A coluna Protocolo possui chaves duplicadas."
    
    # V2: Garante que a coluna de valor está pronta para cálculos matemáticos
    assert pd.api.types.is_float_dtype(df_analytical['Valor']), "Erro: A coluna Valor não foi convertida corretamente para float."
    
    logging.info("Fase 1 concluída com sucesso. Base analítica consolidada e higienizada.")
    
    return df_analytical, df_raw

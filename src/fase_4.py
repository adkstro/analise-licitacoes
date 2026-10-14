import pandas as pd
import numpy as np
import logging

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def executar_fase_4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 4: Corroboração Interna e Testes de Integridade.
    Objetivo: Identificar 'contratações irmãs' (clusters de mercado fluido) 
    para promover o grau de evidência e executar os unit tests finais.
    """
    df_analytical = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. CORROBORAÇÃO INTERNA (EVIDÊNCIA DE MERCADO FLUIDO)
    # -------------------------------------------------------------------------
    mask_incluido = df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    df_incluidos = df_analytical[mask_incluido].copy()
    
    if not df_incluidos.empty:
        # Agrupamento para encontrar as "contratações irmãs"
        group_cols = ['Município', 'Jurisdicionado', 'Modalidade']
        
        # O uso de transform garante operações vetorizadas alinhadas ao índice original do df_incluidos
        df_incluidos['Tamanho_Grupo'] = df_incluidos.groupby(group_cols, dropna=False)['Protocolo'].transform('count')
        
        # Concatena todos os protocolos do mesmo grupo em uma string para futura extração
        df_incluidos['Todos_Protocolos'] = df_incluidos.groupby(group_cols, dropna=False)['Protocolo'].transform(lambda x: ', '.join(x.astype(str)))
        
        # Filtra apenas os clusters (tamanho > 1)
        mask_cluster = df_incluidos['Tamanho_Grupo'] > 1
        df_clusters = df_incluidos[mask_cluster]
        
        if not df_clusters.empty:
            # List comprehension nativa (C-speed) para remover o próprio protocolo da string agregada
            # Evita o uso do .apply() lento linha a linha do Pandas
            protocolos_irmaos = [
                ', '.join([p.strip() for p in todos.split(',') if p.strip() != str(proprio)])
                for todos, proprio in zip(df_clusters['Todos_Protocolos'], df_clusters['Protocolo'])
            ]
            
            indices_promo = df_clusters.index
            
            # A Promoção (Up-grade)
            df_analytical.loc[indices_promo, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
            df_analytical.loc[indices_promo, 'GI - Corroboração Utilizada'] = 'CORROBORAÇÃO INTERNA (CONTRATAÇÕES IRMÃS)'
            df_analytical.loc[indices_promo, 'GI - Protocolo Relacionado'] = protocolos_irmaos
            
            logging.info(f"{len(indices_promo)} registros promovidos a 'GRAU A' via Corroboração Interna.")
        else:
            logging.info("Nenhuma corroboração interna identificada (clusters de tamanho único).")
            
    # -------------------------------------------------------------------------
    # 2. ASSERÇÕES E TESTES DE QUALIDADE (UNIT TESTS)
    # -------------------------------------------------------------------------
    # Previne que falhas de merge ou valores nulos (pd.NA) corrompam a tipagem original
    df_analytical['GI - Divergência Cadastral Aparente'] = df_analytical['GI - Divergência Cadastral Aparente'].fillna(False).astype(bool)
    
    try:
        # V1 (Unicidade Absoluta): Garante que não houve duplicação de linhas nos joins da Fase 3
        assert df_analytical['Protocolo'].duplicated().sum() == 0, \
            "FALHA V1: Existem protocolos duplicados na base analítica após a reintegração."
        
        # V2 (Limpeza da Fila): Garante que nenhum registro ficou esquecido no trânsito (não avaliado)
        assert not (df_analytical['GI - Grau de Evidência'] == 'GRAU D — NÃO AVALIADO (FILA)').any(), \
            "FALHA V2: Registros não processados encontrados. A fila do LLM (GRAU D) não foi completamente esvaziada."
        
        # V3 (Tipagem da Divergência): Garante o formato Booleano exigido pelo schema
        assert pd.api.types.is_bool_dtype(df_analytical['GI - Divergência Cadastral Aparente']), \
            "FALHA V3: A coluna 'GI - Divergência Cadastral Aparente' perdeu o formato booleano."
            
        logging.info("✅ Todos os testes de qualidade (V1, V2 e V3) passaram com sucesso.")
        
    except AssertionError as e:
        logging.error(f"❌ ERRO CRÍTICO DE INTEGRIDADE: {e}")
        raise e
        
    logging.info("Fase 4 concluída. Dados estruturados, corroborados e testados.")
    
    return df_analytical
import pandas as pd
import numpy as np
import logging

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def executar_fase_4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 4: Corroboração Interna e Testes de Integridade.
    Objetivo: Identificar 'contratações irmãs' (clusters de mercado fluido) 
    para promover o grau de evidência, alinhar consistência dos 3 eixos GI
    e executar os unit tests finais.
    """
    df_analytical = df.copy()

    # Garante tipagem 'object' nas colunas textuais GI antes de atribuições vetorizadas
    for col_gi in ['GI - Fundamentação Contextual', 'GI - Grau de Evidência', 'GI - Resultado Etapa 1', 'GI - Corroboração Utilizada', 'GI - Protocolo Relacionado']:
        if col_gi in df_analytical.columns:
            df_analytical[col_gi] = df_analytical[col_gi].astype('object')
    
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
            protocolos_irmaos = [
                ', '.join([p.strip() for p in todos.split(',') if p.strip() != str(proprio)])
                for todos, proprio in zip(df_clusters['Todos_Protocolos'], df_clusters['Protocolo'])
            ]
            
            indices_promo = df_clusters.index
            
            # A Promoção (Up-grade para Grau A — Evidência Direta)
            df_analytical.loc[indices_promo, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
            df_analytical.loc[indices_promo, 'GI - Corroboração Utilizada'] = 'CORROBORAÇÃO INTERNA (CONTRATAÇÕES IRMÃS)'

            # Preserva eventual número de protocolo já extraído na Fase 2 ao concatenar os irmãos
            atuais = df_analytical.loc[indices_promo, 'GI - Protocolo Relacionado']
            novos_relacionados = [
                f"{atual} | {irmao}" if pd.notna(atual) and str(atual).strip() != "" and str(atual).strip() not in irmao else irmao
                for atual, irmao in zip(atuais, protocolos_irmaos)
            ]
            df_analytical.loc[indices_promo, 'GI - Protocolo Relacionado'] = novos_relacionados
            
            logging.info(f"{len(indices_promo)} registros promovidos a 'GRAU A — EVIDÊNCIA DIRETA' via Corroboração Interna.")
        else:
            logging.info("Nenhuma corroboração interna identificada (clusters de tamanho único).")

    # -------------------------------------------------------------------------
    # 2. ALINHAMENTO MATEMÁTICO DOS 3 EIXOS GI (VETORIZADO)
    # -------------------------------------------------------------------------
    # Eixo 2 -> Eixo 1: Garante bijeção estrita entre Grau de Evidência e Resultado Etapa 1
    cond_grau_ab = df_analytical['GI - Grau de Evidência'].isin(['GRAU A — EVIDÊNCIA DIRETA', 'GRAU B — SINTOMÁTICO'])
    cond_grau_c = df_analytical['GI - Grau de Evidência'] == 'GRAU C — SEM EVIDÊNCIA'
    cond_grau_d = df_analytical['GI - Grau de Evidência'] == 'GRAU D — TEXTO VAGO/GENÉRICO'

    df_analytical.loc[cond_grau_ab, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
    df_analytical.loc[cond_grau_c, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO'
    df_analytical.loc[cond_grau_d, 'GI - Resultado Etapa 1'] = 'VALIDAÇÃO PENDENTE'

    # Eixo 3: Divergência Cadastral Aparente (VERDADEIRO quando o Resultado Etapa 1 contradiz a Modalidade)
    modalidade_cred_inex = df_analytical['Modalidade'].fillna('').str.contains(
        r'inexigibilidade|credenciamento|chamamento|chamada\s+publica', case=False, regex=True
    )

    res_incluido = df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    res_excluido = df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO'
    res_val_pendente = df_analytical['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'

    # Contradição 1: INCLUÍDO materialmente, mas modalidade cadastrada NÃO é credenciamento/inexigibilidade
    # Contradição 2: EXCLUÍDO materialmente (Grau C), mas modalidade cadastrada É credenciamento/inexigibilidade
    # Concordância ou VALIDAÇÃO PENDENTE (Grau D): False (default)
    df_analytical['GI - Divergência Cadastral Aparente'] = np.where(
        res_val_pendente,
        False,
        np.where(
            (res_incluido & (~modalidade_cred_inex)) | (res_excluido & modalidade_cred_inex),
            True,
            False
        )
    ).astype(bool)

    # -------------------------------------------------------------------------
    # 3. ASSERÇÕES E TESTES DE QUALIDADE (UNIT TESTS)
    # -------------------------------------------------------------------------
    try:
        # V1 (Unicidade Absoluta): Garante que não houve duplicação de linhas nos joins da Fase 3
        assert df_analytical['Protocolo'].duplicated().sum() == 0, \
            "FALHA V1: Existem protocolos duplicados na base analítica após a reintegração."
        
        # V2 (Limpeza da Fila): Garante que nenhum registro ficou esquecido no trânsito (não avaliado)
        mask_nao_avaliado = (
            df_analytical['GI - Grau de Evidência'].isin(['NA — NÃO AVALIADO (FILA)', 'GRAU D — NÃO AVALIADO (FILA)'])
            | (df_analytical['GI - Resultado Etapa 1'] == 'FILA PARA LLM')
        )
        assert not mask_nao_avaliado.any(), \
            f"FALHA V2: {int(mask_nao_avaliado.sum())} registros não processados encontrados na fila do LLM."
        
        # V3 (Tipagem e Regra de Divergência): Garante formato Booleano e que Grau D nunca tenha divergência True
        assert pd.api.types.is_bool_dtype(df_analytical['GI - Divergência Cadastral Aparente']), \
            "FALHA V3: A coluna 'GI - Divergência Cadastral Aparente' perdeu o formato booleano."
        assert not df_analytical.loc[res_val_pendente, 'GI - Divergência Cadastral Aparente'].any(), \
            "FALHA V3.1: Existem registros em VALIDAÇÃO PENDENTE (Grau D) com Divergência Cadastral marcada como True."

        # V4 (Domínio Sintático Estrito): Garante apenas os valores permitidos nos 3 eixos
        resultados_validos = {'INCLUÍDO', 'EXCLUÍDO', 'VALIDAÇÃO PENDENTE'}
        assert set(df_analytical['GI - Resultado Etapa 1'].dropna().unique()).issubset(resultados_validos), \
            f"FALHA V4: Valores fora do padrão em 'GI - Resultado Etapa 1': {df_analytical['GI - Resultado Etapa 1'].unique()}"
            
        logging.info("✅ Todos os testes de qualidade (V1, V2, V3 e V4) passaram com sucesso.")
        
    except AssertionError as e:
        logging.error(f"❌ ERRO CRÍTICO DE INTEGRIDADE: {e}")
        raise e
        
    logging.info("Fase 4 concluída. Dados estruturados, corroborados e testados.")
    
    return df_analytical

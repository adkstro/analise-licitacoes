import os
import logging
from typing import Optional
import numpy as np
import pandas as pd

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

CAMINHO_VALIDACAO_HUMANA_PADRAO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'validacao_humana_r0_r6.xlsx'
)


def aplicar_retroalimentacao_humana(
    df_analytical: pd.DataFrame,
    caminho_validacao: Optional[str] = None
) -> pd.DataFrame:
    """
    Aplica overrides de validação humana (Seções 9 e 10 da Especificação Funcional)
    caso exista planilha/CSV de retroalimentação informada ou no diretório padrão.
    Colunas esperadas: Protocolo, Sinal (R0..R6), Decisao_Humana, Motivo_Decisao, Regra_Afetada.
    """
    caminhos_tentativa = [
        c for c in [
            caminho_validacao,
            CAMINHO_VALIDACAO_HUMANA_PADRAO,
            CAMINHO_VALIDACAO_HUMANA_PADRAO.replace('.xlsx', '.csv'),
            'validacao_humana_r0_r6.xlsx',
            'validacao_humana_r0_r6.csv'
        ] if c is not None
    ]

    caminho_encontrado = next((c for c in caminhos_tentativa if os.path.exists(c)), None)
    if not caminho_encontrado:
        return df_analytical

    try:
        if caminho_encontrado.endswith('.csv'):
            df_val = pd.read_csv(caminho_encontrado)
        else:
            df_val = pd.read_excel(caminho_encontrado)

        colunas_req = {'Protocolo', 'Sinal', 'Decisao_Humana'}
        if not colunas_req.issubset(set(df_val.columns)):
            logging.warning(
                f"Arquivo de validação humana '{caminho_encontrado}' não contém as colunas mínimas {colunas_req}."
            )
            return df_analytical

        df_val = df_val.dropna(subset=['Protocolo', 'Sinal', 'Decisao_Humana']).copy()
        if df_val.empty:
            return df_analytical

        df_val['Protocolo'] = df_val['Protocolo'].astype(str).str.strip()
        df_val['Sinal'] = df_val['Sinal'].astype(str).str.strip().str.upper()
        df_val['Decisao_Humana'] = df_val['Decisao_Humana'].astype(str).str.strip().str.upper()

        mapa_decisao = {
            'FALSO POSITIVO': 'AFASTADO',
            'AFASTADO': 'AFASTADO',
            'CONFIRMADO': 'CONFIRMADO',
            'PENDENTE DOCUMENTAL': 'PENDENTE DOCUMENTAL',
            'REVISAR': 'REVISAR',
            'NÃO IDENTIFICADO': 'NÃO IDENTIFICADO',
            'NAO IDENTIFICADO': 'NÃO IDENTIFICADO'
        }
        df_val['Status_Final'] = df_val['Decisao_Humana'].map(mapa_decisao)
        df_val = df_val[df_val['Status_Final'].notna()].drop_duplicates(subset=['Protocolo', 'Sinal'], keep='last')

        df_out = df_analytical.set_index('Protocolo')
        aplicados = 0

        for sinal in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
            sub = df_val[df_val['Sinal'] == sinal].set_index('Protocolo')
            if sub.empty:
                continue
            idx_comum = df_out.index.intersection(sub.index)
            if len(idx_comum) == 0:
                continue

            df_out.loc[idx_comum, f'{sinal}_STATUS'] = sub.loc[idx_comum, 'Status_Final']
            if 'Motivo_Decisao' in sub.columns:
                motivo = sub.loc[idx_comum, 'Motivo_Decisao'].fillna('').astype(str)
                regra_humana = 'Validação Humana: ' + motivo
                df_out.loc[idx_comum, f'{sinal}_REGRA'] = regra_humana
            aplicados += len(idx_comum)

        logging.info(f"Retroalimentação humana aplicada com sucesso em {aplicados} avaliações ({caminho_encontrado}).")
        return df_out.reset_index()

    except Exception as e:
        logging.warning(f"Erro ao processar planilha de validação humana '{caminho_encontrado}': {e}")
        return df_analytical


def consolidar_motor_r0_r6(
    df_analytical: pd.DataFrame,
    caminho_validacao_humana: Optional[str] = None
) -> pd.DataFrame:
    """
    Executa a resolução final da Camada 2 para os sinais R0 a R6, aplica eventual
    retroalimentação humana e calcula as dimensões D1–D5 e a prioridade qualitativa Q1–Q4.
    """
    s_objeto = df_analytical['Objeto'].fillna('').astype(str)
    s_modalidade = df_analytical['Modalidade'].fillna('').astype(str)
    s_tipo_objeto = (
        df_analytical['Tipo do Objeto'].fillna('').astype(str)
        if 'Tipo do Objeto' in df_analytical.columns
        else pd.Series('', index=df_analytical.index, dtype='object')
    )

    mod_eh_credenciamento = s_modalidade.str.contains(r'credenciamento|chamamento|chamada\s+publica', case=False, regex=True)
    mod_eh_inexigibilidade = s_modalidade.str.contains(r'inexigibilidade', case=False, regex=True)
    mod_cred_inex = mod_eh_credenciamento | mod_eh_inexigibilidade
    res_incluido = df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    res_excluido = df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO'
    res_val_pendente = df_analytical['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'
    escopo_credenciamento = mod_cred_inex | res_incluido

    # -------------------------------------------------------------------------
    # 1. SINCRONIZAÇÃO DE R0 COM A DIVERGÊNCIA CADASTRAL APARENTE (GI)
    # -------------------------------------------------------------------------
    mask_div_gi = df_analytical['GI - Divergência Cadastral Aparente'].fillna(False).astype(bool)
    mask_r0_de_gi = mask_div_gi & (df_analytical['R0_STATUS'] == 'NÃO IDENTIFICADO')
    if mask_r0_de_gi.any():
        df_analytical.loc[mask_r0_de_gi, 'R0_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r0_de_gi, 'R0_EVIDENCIA'] = (
            s_modalidade.loc[mask_r0_de_gi] + ' vs ' + df_analytical.loc[mask_r0_de_gi, 'GI - Resultado Etapa 1'].astype(str)
        )
        df_analytical.loc[mask_r0_de_gi, 'R0_REGRA'] = (
            'Divergência entre a Modalidade cadastrada e a classificação material da Etapa 1 (GI)'
        )

    # -------------------------------------------------------------------------
    # 2. RESOLUÇÃO CONTEXTUAL CAMADA 2 PARA CANDIDATOS R1 (FORÇA DE TRABALHO)
    # -------------------------------------------------------------------------
    mask_r1_cand = df_analytical['R1_STATUS'] == 'CANDIDATO'
    if mask_r1_cand.any():
        # Fora do universo de credenciamento/inexigibilidade -> AFASTADO
        mask_r1_fora = mask_r1_cand & (~escopo_credenciamento)
        if mask_r1_fora.any():
            df_analytical.loc[mask_r1_fora, 'R1_STATUS'] = 'AFASTADO'
            df_analytical.loc[mask_r1_fora, 'R1_REGRA'] = (
                'Afastado na Camada 2: Contratação em modalidade competitiva comum fora do escopo de credenciamento'
            )

        # Dentro do universo de credenciamento/inexigibilidade: avalia contexto acessório vs núcleo de mão de obra
        mask_r1_escopo = (df_analytical['R1_STATUS'] == 'CANDIDATO') & escopo_credenciamento
        if mask_r1_escopo.any():
            regex_r1_acessorio = (
                r'LOCAC[AÃ]O|FRETAMENTO|TRANSPORTE\s+(?:ESCOLAR|DE\s+PACIENTES|DE\s+PASSAGEIROS|UNIVERSITARIO)|'
                r'PASSAGENS?|HORAS?[\s/\-]+MAQUINAS?|CAMINH[OÕ]ES|RETROESCAVADEIRA|TRATOR|VEICULOS?|'
                r'PECAS\s+AUTOMOTIVAS|OFICINA|MANUTENC[AÃ]O\s+DE\s+VEICULOS|FORNECIMENTO\s+DE\s+REFEIC[OÕ]ES|'
                r'RESTAURANTE|LANCHES?|PLANT[OÕ]ES|EXAMES?|CONSULTAS?'
            )
            regex_r1_nucleo = (
                r'(?:CONTRATAC[AÃ]O|CREDENCIAMENTO|PRESTAC[AÃ]O\s+DE\s+SERVICOS?)\s+DE\s+'
                r'(?:PESSOAS?\s+FISICAS?\s+E[/\s]OU\s+JURIDICAS?\s+PARA\s+)?'
                r'(?:SERVICOS?\s+DE\s+|FUNCAO\s+DE\s+|PROFISSIONA(?:L|IS)\s+)?'
                r'(?:MOTORISTAS?|PEDREIROS?|SERVENTES?|PINTOR(?:ES)?|JARDINEIROS?|OPERADOR(?:ES)?|'
                r'SERVICOS\s+GERAIS|AUXILIAR(?:ES)?\s+DE\s+SERVICOS\s+GERAIS|MERENDEIRAS?|'
                r'PORTEIROS?|VIGIAS?|ZELADOR(?:ES)?|COVEIROS?|GARIS|RECEPCIONISTAS?|ELETRICISTAS?|ENCANADOR(?:ES)?|CARPINTEIROS?)'
            )

            cond_r1_acessorio = s_objeto.str.contains(regex_r1_acessorio, case=False, regex=True)
            cond_r1_nucleo = s_objeto.str.contains(regex_r1_nucleo, case=False, regex=True)

            mask_r1_afast = mask_r1_escopo & cond_r1_acessorio
            mask_r1_conf = mask_r1_escopo & (~cond_r1_acessorio) & (cond_r1_nucleo | res_excluido)
            mask_r1_rev = mask_r1_escopo & (~cond_r1_acessorio) & (~cond_r1_nucleo) & (~res_excluido)

            if mask_r1_afast.any():
                df_analytical.loc[mask_r1_afast, 'R1_STATUS'] = 'AFASTADO'
                df_analytical.loc[mask_r1_afast, 'R1_REGRA'] = (
                    'Afastado na Camada 2: Termo operacional inserido em contexto acessório (transporte, locação, manutenção veicular ou saúde)'
                )
            if mask_r1_conf.any():
                df_analytical.loc[mask_r1_conf, 'R1_STATUS'] = 'CONFIRMADO'
                df_analytical.loc[mask_r1_conf, 'R1_REGRA'] = (
                    'Confirmado na Camada 2: Força de trabalho operacional identificada como núcleo da contratação sob credenciamento/inexigibilidade'
                )
            if mask_r1_rev.any():
                df_analytical.loc[mask_r1_rev, 'R1_STATUS'] = 'REVISAR'
                df_analytical.loc[mask_r1_rev, 'R1_REGRA'] = (
                    'Revisão Manual na Camada 2: Menção a função operacional em objeto com múltiplos itens ou contexto ambíguo'
                )

    # -------------------------------------------------------------------------
    # 3. RESOLUÇÃO CONTEXTUAL CAMADA 2 PARA CANDIDATOS R2 (OBRAS E ENGENHARIA)
    # -------------------------------------------------------------------------
    mask_r2_cand = df_analytical['R2_STATUS'] == 'CANDIDATO'
    if mask_r2_cand.any():
        mask_r2_fora = mask_r2_cand & (~escopo_credenciamento)
        if mask_r2_fora.any():
            df_analytical.loc[mask_r2_fora, 'R2_STATUS'] = 'AFASTADO'
            df_analytical.loc[mask_r2_fora, 'R2_REGRA'] = (
                'Afastado na Camada 2: Obra/engenharia licitada em modalidade competitiva comum sem vínculo com credenciamento'
            )

        mask_r2_escopo = (df_analytical['R2_STATUS'] == 'CANDIDATO') & escopo_credenciamento
        if mask_r2_escopo.any():
            regex_r2_intelectual_ou_acessorio = (
                r'PROJETOS?\s+(?:DE\s+ENGENHARIA|ARQUITETONICOS?|BASICOS?|EXECUTIVOS?)|'
                r'CONSULTORIA|ASSESSORIA|FISCALIZAC[AÃ]O\s+DE\s+OBRAS?|LAUDOS?\s+TECNICOS?|'
                r'PERICIAS?|TOPOGRAFIA|GEORREFERENCIAMENTO|LOCAC[AÃ]O\s+DE|AQUISIC[AÃ]O\s+DE'
            )
            cond_r2_acessorio = s_objeto.str.contains(regex_r2_intelectual_ou_acessorio, case=False, regex=True)
            cond_tipo_obra = s_tipo_objeto.str.contains(r'OBRAS?\s+E\s+SERVICOS\s+DE\s+ENGENHARIA|\bOBRAS?\b', case=False, regex=True)

            mask_r2_afast = mask_r2_escopo & cond_r2_acessorio & (~cond_tipo_obra)
            mask_r2_conf = mask_r2_escopo & (~mask_r2_afast)

            if mask_r2_afast.any():
                df_analytical.loc[mask_r2_afast, 'R2_STATUS'] = 'AFASTADO'
                df_analytical.loc[mask_r2_afast, 'R2_REGRA'] = (
                    'Afastado na Camada 2: Serviço técnico intelectual (projeto/consultoria/laudo) sem execução direta de obra'
                )
            if mask_r2_conf.any():
                df_analytical.loc[mask_r2_conf, 'R2_STATUS'] = 'CONFIRMADO'
                df_analytical.loc[mask_r2_conf, 'R2_REGRA'] = (
                    'Confirmado na Camada 2: Execução de obra, engenharia ou manutenção predial vinculada a credenciamento/inexigibilidade'
                )

    # -------------------------------------------------------------------------
    # 4. COMPLEMENTO DE R4 (PENDENTE DOCUMENTAL EM EDITAIS DE CREDENCIAMENTO)
    # -------------------------------------------------------------------------
    # Quando o registro é um edital/chamamento aberto de credenciamento (pluralidade de prestadores),
    # mas o Objeto não explicita o critério objetivo de distribuição da demanda e não é seleção por terceiros
    regex_selecao_terceiros = r'VALE[\s\-]?(?:ALIMENTAC[AÃ]O|REFEIC[AÃ]O|CRECHE)|CART[AÃ]O\s+MAGNETICO|ESCOLHA\s+DO\s+(?:USUARIO|BENEFICIARIO)'
    tem_vinculo_originario = df_analytical['GI - Protocolo Relacionado'].notna()
    cond_edital_aberto = mod_eh_credenciamento | s_objeto.str.contains(r'CHAMAMENTO\s+PUBLICO|EDITAL\s+DE\s+CREDENCIAMENTO', case=False, regex=True)

    mask_r4_pendente = (
        (df_analytical['R4_STATUS'] == 'NÃO IDENTIFICADO')
        & res_incluido
        & cond_edital_aberto
        & (~tem_vinculo_originario)
        & (~s_objeto.str.contains(regex_selecao_terceiros, case=False, regex=True))
    )
    if mask_r4_pendente.any():
        df_analytical.loc[mask_r4_pendente, 'R4_STATUS'] = 'PENDENTE DOCUMENTAL'
        df_analytical.loc[mask_r4_pendente, 'R4_EVIDENCIA'] = 'Critério de distribuição omitido na síntese do Objeto'
        df_analytical.loc[mask_r4_pendente, 'R4_REGRA'] = (
            'Procedimento de credenciamento sem explicitação do critério objetivo de distribuição da demanda no Objeto (verificar Edital/TR)'
        )

    # -------------------------------------------------------------------------
    # 5. COMPLEMENTO DE R6 (TEXTOS VAGOS / GRAU D)
    # -------------------------------------------------------------------------
    mask_r6_grau_d = (df_analytical['R6_STATUS'] == 'NÃO IDENTIFICADO') & res_val_pendente
    if mask_r6_grau_d.any():
        df_analytical.loc[mask_r6_grau_d, 'R6_STATUS'] = 'REVISAR'
        df_analytical.loc[mask_r6_grau_d, 'R6_EVIDENCIA'] = 'Objeto genérico / Grau D'
        df_analytical.loc[mask_r6_grau_d, 'R6_REGRA'] = (
            'Objeto com descrição insuficiente para confirmar o cabimento material do credenciamento'
        )

    # -------------------------------------------------------------------------
    # 6. APLICAÇÃO DE RETROALIMENTAÇÃO HUMANA (SE EXISTIR PLANILHA DE OVERRIDE)
    # -------------------------------------------------------------------------
    df_analytical = aplicar_retroalimentacao_humana(df_analytical, caminho_validacao_humana)

    # -------------------------------------------------------------------------
    # 7. CONSOLIDAÇÃO DAS DIMENSÕES D1 A D5 (SEÇÃO 7 — SEM DUPLA CONTAGEM EM D2)
    # -------------------------------------------------------------------------
    df_analytical['D1_ENQUADRAMENTO'] = (
        (df_analytical['R0_STATUS'] == 'CONFIRMADO')
        | (df_analytical['R6_STATUS'].isin(['CONFIRMADO', 'REVISAR']))
    ).astype(bool)

    # Regra explícita da Seção 7.2: R1 e R2 contam apenas 1 vez na dimensão D2_EXECUCAO
    df_analytical['D2_EXECUCAO'] = (
        (df_analytical['R1_STATUS'] == 'CONFIRMADO')
        | (df_analytical['R2_STATUS'] == 'CONFIRMADO')
    ).astype(bool)

    df_analytical['D3_PRECIFICACAO'] = (df_analytical['R3_STATUS'] == 'CONFIRMADO').astype(bool)
    df_analytical['D4_DISTRIBUICAO'] = (df_analytical['R4_STATUS'] == 'CONFIRMADO').astype(bool)
    df_analytical['D5_RECORRENCIA'] = (df_analytical['R5_STATUS'] == 'CONFIRMADO').astype(bool)

    df_analytical['QTD_DIMENSOES'] = (
        df_analytical['D1_ENQUADRAMENTO'].astype(int)
        + df_analytical['D2_EXECUCAO'].astype(int)
        + df_analytical['D3_PRECIFICACAO'].astype(int)
        + df_analytical['D4_DISTRIBUICAO'].astype(int)
        + df_analytical['D5_RECORRENCIA'].astype(int)
    )

    # -------------------------------------------------------------------------
    # 8. CLASSIFICAÇÃO EM PRIORIDADE QUALITATIVA (Q1 A Q4 — SEÇÃO 8)
    # -------------------------------------------------------------------------
    colunas_status = [f'R{i}_STATUS' for i in range(7)]
    tem_pendente_doc = df_analytical[colunas_status].isin(['PENDENTE DOCUMENTAL']).any(axis=1)
    tem_revisar = df_analytical[colunas_status].isin(['REVISAR']).any(axis=1)
    tem_pendencia_ou_revisao = tem_pendente_doc | tem_revisar

    d1 = df_analytical['D1_ENQUADRAMENTO']
    d2 = df_analytical['D2_EXECUCAO']
    qtd_dim = df_analytical['QTD_DIMENSOES']
    r0_conf = df_analytical['R0_STATUS'] == 'CONFIRMADO'

    # Q1 — Muito alta: Patologia material de execução (D2) combinada com outra dimensão/pendência,
    # ou Divergência de Enquadramento (D1) combinada com múltiplas dimensões
    cond_q1 = (
        (d2 & (d1 | (qtd_dim >= 2) | tem_pendencia_ou_revisao))
        | (d1 & r0_conf & (qtd_dim >= 2))
        | (d1 & (qtd_dim >= 3))
    )

    # Q2 — Alta: D2 isolado, R0 confirmado isolado, D1 combinado com pendência/outra dimensão, ou 3+ dimensões estruturais
    cond_q2 = (~cond_q1) & (
        d2
        | r0_conf
        | (d1 & ((qtd_dim >= 2) | tem_pendente_doc))
        | (qtd_dim >= 3)
    )

    # Q3 — Média: Presença de dimensão estrutural (D1/D3/D4/D5) ou necessidade de verificação documental/revisão
    cond_q3 = (~cond_q1) & (~cond_q2) & ((qtd_dim >= 1) | tem_pendencia_ou_revisao)

    df_analytical['PRIORIDADE_QUALITATIVA'] = np.select(
        [cond_q1, cond_q2, cond_q3],
        ['Q1 — Muito alta', 'Q2 — Alta', 'Q3 — Média'],
        default='Q4 — Monitoramento'
    )
    df_analytical['PRIORIDADE_QUALITATIVA'] = df_analytical['PRIORIDADE_QUALITATIVA'].astype('object')

    df_analytical['NECESSITA_REVISAO_MANUAL'] = (
        tem_pendencia_ou_revisao
        | cond_q1
        | cond_q2
        | res_val_pendente
    ).astype(bool)

    # -------------------------------------------------------------------------
    # 9. CONSTRUÇÃO VETORIZADA DA JUSTIFICATIVA DE TRIAGEM (SÍNTESE AUDITÁVEL)
    # -------------------------------------------------------------------------
    partes_sinais = []
    nomes_curtos = {
        'R0': 'R0(Enquadramento)',
        'R1': 'R1(Mão de Obra)',
        'R2': 'R2(Obra/Engenharia)',
        'R3': 'R3(Mercado Fluido)',
        'R4': 'R4(Distribuição)',
        'R5': 'R5(Recorrência)',
        'R6': 'R6(Regime/Cabimento)'
    }
    acumulador_sinais = pd.Series('', index=df_analytical.index, dtype='object')
    for sinal, rotulo in nomes_curtos.items():
        st = df_analytical[f'{sinal}_STATUS']
        ativo = st.isin(['CONFIRMADO', 'PENDENTE DOCUMENTAL', 'REVISAR'])
        trecho = np.where(ativo, rotulo + ': ' + st.astype(str) + '; ', '')
        acumulador_sinais = acumulador_sinais + pd.Series(trecho, index=df_analytical.index, dtype='object')

    acumulador_sinais = acumulador_sinais.str.rstrip('; ')
    sem_sinais = acumulador_sinais == ''

    df_analytical['JUSTIFICATIVA_TRIAGEM'] = np.where(
        sem_sinais,
        df_analytical['PRIORIDADE_QUALITATIVA'].astype(str) + ' | Nenhum sinal crítico ou pendência identificados na triagem.',
        df_analytical['PRIORIDADE_QUALITATIVA'].astype(str)
        + ' (Dimensões ativas: ' + df_analytical['QTD_DIMENSOES'].astype(str) + ') | '
        + acumulador_sinais
    )
    df_analytical['JUSTIFICATIVA_TRIAGEM'] = df_analytical['JUSTIFICATIVA_TRIAGEM'].astype('object')

    return df_analytical


def executar_fase_4(
    df: pd.DataFrame,
    caminho_validacao_humana: Optional[str] = None
) -> pd.DataFrame:
    """
    Fase 4: Corroboração Interna, Consolidação do Motor R0–R6 (D1–D5 / Q1–Q4) e Testes de Integridade.
    Objetivo: Identificar 'contratações irmãs' (clusters de mercado fluido) para promover o grau de evidência,
    alinhar consistência dos 3 eixos GI, consolidar a matriz de triagem R0–R6 e executar os unit tests finais.
    """
    df_analytical = df.copy()

    # Garante tipagem 'object' nas colunas textuais GI e R0–R6 antes de atribuições vetorizadas
    colunas_gi_obj = [
        'GI - Fundamentação Contextual',
        'GI - Grau de Evidência',
        'GI - Resultado Etapa 1',
        'GI - Corroboração Utilizada',
        'GI - Protocolo Relacionado'
    ]
    for col_gi in colunas_gi_obj:
        if col_gi not in df_analytical.columns:
            df_analytical[col_gi] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
        else:
            df_analytical[col_gi] = df_analytical[col_gi].astype('object')

    for sinal in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        col_st = f'{sinal}_STATUS'
        col_ev = f'{sinal}_EVIDENCIA'
        col_rg = f'{sinal}_REGRA'
        if col_st not in df_analytical.columns:
            df_analytical[col_st] = pd.Series('NÃO IDENTIFICADO', index=df_analytical.index, dtype='object')
        else:
            df_analytical[col_st] = df_analytical[col_st].astype('object')
        for c_aux in (col_ev, col_rg):
            if c_aux not in df_analytical.columns:
                df_analytical[c_aux] = pd.Series(pd.NA, index=df_analytical.index, dtype='object')
            else:
                df_analytical[c_aux] = df_analytical[c_aux].astype('object')
    
    # -------------------------------------------------------------------------
    # 1. CORROBORAÇÃO INTERNA (EVIDÊNCIA DE MERCADO FLUIDO)
    # -------------------------------------------------------------------------
    mask_incluido = df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    df_incluidos = df_analytical[mask_incluido].copy()
    
    if not df_incluidos.empty:
        group_cols = ['Município', 'Jurisdicionado', 'Modalidade']
        df_incluidos['Tamanho_Grupo'] = df_incluidos.groupby(group_cols, dropna=False)['Protocolo'].transform('count')
        df_incluidos['Todos_Protocolos'] = df_incluidos.groupby(group_cols, dropna=False)['Protocolo'].transform(
            lambda x: ', '.join(x.astype(str))
        )
        
        mask_cluster = df_incluidos['Tamanho_Grupo'] > 1
        df_clusters = df_incluidos[mask_cluster]
        
        if not df_clusters.empty:
            protocolos_irmaos = [
                ', '.join([p.strip() for p in todos.split(',') if p.strip() != str(proprio)])
                for todos, proprio in zip(df_clusters['Todos_Protocolos'], df_clusters['Protocolo'])
            ]
            
            indices_promo = df_clusters.index
            
            df_analytical.loc[indices_promo, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
            df_analytical.loc[indices_promo, 'GI - Corroboração Utilizada'] = 'CORROBORAÇÃO INTERNA (CONTRATAÇÕES IRMÃS)'

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
    cond_grau_ab = df_analytical['GI - Grau de Evidência'].isin(['GRAU A — EVIDÊNCIA DIRETA', 'GRAU B — SINTOMÁTICO'])
    cond_grau_c = df_analytical['GI - Grau de Evidência'] == 'GRAU C — SEM EVIDÊNCIA'
    cond_grau_d = df_analytical['GI - Grau de Evidência'] == 'GRAU D — TEXTO VAGO/GENÉRICO'

    df_analytical.loc[cond_grau_ab, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
    df_analytical.loc[cond_grau_c, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO'
    df_analytical.loc[cond_grau_d, 'GI - Resultado Etapa 1'] = 'VALIDAÇÃO PENDENTE'

    modalidade_cred_inex = df_analytical['Modalidade'].fillna('').str.contains(
        r'inexigibilidade|credenciamento|chamamento|chamada\s+publica', case=False, regex=True
    )

    res_incluido = df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    res_excluido = df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO'
    res_val_pendente = df_analytical['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'

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
    # 3. CONSOLIDAÇÃO DO MOTOR DE TRIAGEM R0 A R6, DIMENSÕES D1–D5 E PRIORIDADE Q1–Q4
    # -------------------------------------------------------------------------
    df_analytical = consolidar_motor_r0_r6(df_analytical, caminho_validacao_humana)

    # -------------------------------------------------------------------------
    # 4. ASSERÇÕES E TESTES DE QUALIDADE (UNIT TESTS V1 A V6)
    # -------------------------------------------------------------------------
    try:
        # V1 (Unicidade Absoluta)
        assert df_analytical['Protocolo'].duplicated().sum() == 0, \
            "FALHA V1: Existem protocolos duplicados na base analítica após a reintegração."
        
        # V2 (Limpeza da Fila)
        mask_nao_avaliado = (
            df_analytical['GI - Grau de Evidência'].isin(['NA — NÃO AVALIADO (FILA)', 'GRAU D — NÃO AVALIADO (FILA)'])
            | (df_analytical['GI - Resultado Etapa 1'] == 'FILA PARA LLM')
        )
        assert not mask_nao_avaliado.any(), \
            f"FALHA V2: {int(mask_nao_avaliado.sum())} registros não processados encontrados na fila do LLM."
        
        # V3 (Tipagem e Regra de Divergência)
        assert pd.api.types.is_bool_dtype(df_analytical['GI - Divergência Cadastral Aparente']), \
            "FALHA V3: A coluna 'GI - Divergência Cadastral Aparente' perdeu o formato booleano."
        assert not df_analytical.loc[res_val_pendente, 'GI - Divergência Cadastral Aparente'].any(), \
            "FALHA V3.1: Existem registros em VALIDAÇÃO PENDENTE (Grau D) com Divergência Cadastral marcada como True."

        # V4 (Domínio Sintático Estrito GI)
        resultados_validos = {'INCLUÍDO', 'EXCLUÍDO', 'VALIDAÇÃO PENDENTE'}
        assert set(df_analytical['GI - Resultado Etapa 1'].dropna().unique()).issubset(resultados_validos), \
            f"FALHA V4: Valores fora do padrão em 'GI - Resultado Etapa 1': {df_analytical['GI - Resultado Etapa 1'].unique()}"

        # V5 (Domínio Sintático Estrito R0 a R6 — Nenhum CANDIDATO pode restar sem resolução)
        status_r_validos = {'CONFIRMADO', 'AFASTADO', 'PENDENTE DOCUMENTAL', 'REVISAR', 'NÃO IDENTIFICADO'}
        for sinal in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
            valores_sinal = set(df_analytical[f'{sinal}_STATUS'].dropna().unique())
            assert valores_sinal.issubset(status_r_validos), \
                f"FALHA V5: Status inválido em '{sinal}_STATUS': {valores_sinal - status_r_validos}"

        # V6 (Integridade das Dimensões D1–D5 e Prioridade Qualitativa Q1–Q4)
        prioridades_validas = {'Q1 — Muito alta', 'Q2 — Alta', 'Q3 — Média', 'Q4 — Monitoramento'}
        assert set(df_analytical['PRIORIDADE_QUALITATIVA'].dropna().unique()).issubset(prioridades_validas), \
            f"FALHA V6: Prioridades fora do domínio: {df_analytical['PRIORIDADE_QUALITATIVA'].unique()}"
            
        logging.info("✅ Todos os testes de qualidade (V1 a V6) passaram com sucesso.")
        
    except AssertionError as e:
        logging.error(f"❌ ERRO CRÍTICO DE INTEGRIDADE: {e}")
        raise e
        
    logging.info("Fase 4 concluída. Dados estruturados, corroborados e matriz R0–R6 consolidada.")
    
    return df_analytical

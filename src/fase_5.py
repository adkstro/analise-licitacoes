import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def executar_fase_5(df_analytical: pd.DataFrame, df_raw: pd.DataFrame) -> str:
    """
    Fase 5: Geração do Entregável Executivo.
    Exporta os dados validados e segmentados segundo a arquitetura de 3 eixos GI
    e a Matriz de Triagem R0–R6 (D1–D5 / Q1–Q4) para um Excel formatado.
    """
    # 1. NOMENCLATURA DO ARQUIVO
    data_atual = datetime.now().strftime("%Y-%m-%d")
    nome_arquivo = f"Auditoria_Credenciamento_Resultados_{data_atual}.xlsx"
    
    logging.info(f"Iniciando a geração do entregável executivo: {nome_arquivo}")
    
    # Prepara cópias para não alterar os originais e garante Protocolo como string
    df_ana = df_analytical.copy()
    df_bruta = df_raw.copy()
    df_ana['Protocolo'] = df_ana['Protocolo'].astype(str)
    df_bruta['Protocolo'] = df_bruta['Protocolo'].astype(str)

    # Garante a ordenação lógica das colunas GI e R0–R6 no entregável
    colunas_gi_ordenadas = [
        'GI - Fundamentação Contextual',
        'GI - Grau de Evidência',
        'GI - Resultado Etapa 1',
        'GI - Divergência Cadastral Aparente',
        'GI - Corroboração Utilizada',
        'GI - Protocolo Relacionado',
        'GI - Regime Jurídico Informado'
    ]
    colunas_r0_r6_ordenadas = []
    for s in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        colunas_r0_r6_ordenadas.extend([f'{s}_STATUS', f'{s}_EVIDENCIA', f'{s}_REGRA'])
    colunas_r0_r6_ordenadas.extend([
        'D1_ENQUADRAMENTO',
        'D2_EXECUCAO',
        'D3_PRECIFICACAO',
        'D4_DISTRIBUICAO',
        'D5_RECORRENCIA',
        'QTD_DIMENSOES',
        'PRIORIDADE_QUALITATIVA',
        'NECESSITA_REVISAO_MANUAL',
        'JUSTIFICATIVA_TRIAGEM'
    ])

    colunas_especiais = set(colunas_gi_ordenadas + colunas_r0_r6_ordenadas)
    colunas_base = [c for c in df_ana.columns if c not in colunas_especiais]
    colunas_gi_presentes = [c for c in colunas_gi_ordenadas if c in df_ana.columns]
    colunas_r_presentes = [c for c in colunas_r0_r6_ordenadas if c in df_ana.columns]
    df_ana = df_ana[colunas_base + colunas_gi_presentes + colunas_r_presentes]
    
    # 2. ROTEAMENTO DE ABAS (SPLIT DO DATAFRAME BASEADO NOS 3 EIXOS E MATRIZ R0–R6)
    mask_incluido = df_ana['GI - Resultado Etapa 1'] == 'INCLUÍDO'
    mask_excluido = df_ana['GI - Resultado Etapa 1'] == 'EXCLUÍDO'
    mask_val_pendente = df_ana['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'

    df_validado = df_ana[mask_incluido].copy()
    
    # 01_Resumo_Executivo (Métricas GI + Métricas do Motor de Triagem R0–R6)
    metricas = [
        'Total de Linhas Analisadas (Protocolos Únicos)',
        'Roteamento: INCLUÍDO',
        'Roteamento: EXCLUÍDO',
        'Roteamento: VALIDAÇÃO PENDENTE',
        'Grau A (Evidência Direta)',
        'Grau B (Sintomático)',
        'Grau C (Sem Evidência)',
        'Grau D (Texto Vago/Genérico)',
        'Alertas de Fraude (Divergência Cadastral Aparente = VERDADEIRO)'
    ]
    quantidades = [
        len(df_ana),
        int(mask_incluido.sum()),
        int(mask_excluido.sum()),
        int(mask_val_pendente.sum()),
        int((df_ana['GI - Grau de Evidência'] == 'GRAU A — EVIDÊNCIA DIRETA').sum()),
        int((df_ana['GI - Grau de Evidência'] == 'GRAU B — SINTOMÁTICO').sum()),
        int((df_ana['GI - Grau de Evidência'] == 'GRAU C — SEM EVIDÊNCIA').sum()),
        int((df_ana['GI - Grau de Evidência'] == 'GRAU D — TEXTO VAGO/GENÉRICO').sum()),
        int(df_ana['GI - Divergência Cadastral Aparente'].fillna(False).astype(bool).sum())
    ]

    if 'PRIORIDADE_QUALITATIVA' in df_ana.columns:
        metricas.extend([
            'Prioridade Qualitativa: Q1 — Muito alta',
            'Prioridade Qualitativa: Q2 — Alta',
            'Prioridade Qualitativa: Q3 — Média',
            'Prioridade Qualitativa: Q4 — Monitoramento',
            'Processos com Necessidade de Revisão Manual',
            'Dimensão D1 — Enquadramento / Cabimento',
            'Dimensão D2 — Execução (Força de Trabalho ou Obra/Engenharia)',
            'Dimensão D3 — Precificação / Mercado Fluido',
            'Dimensão D4 — Distribuição da Demanda Confirmada',
            'Dimensão D5 — Recorrência / Execução Continuada',
            'Sinal R0 Confirmado (Divergência de Enquadramento)',
            'Sinal R1 Confirmado (Força de Trabalho Operacional)',
            'Sinal R2 Confirmado (Obras / Engenharia / Manutenção Predial)',
            'Sinal R3 Confirmado (Mercado Fluido / Preço Variável)',
            'Sinal R4 Confirmado (Critério Objetivo de Distribuição)',
            'Sinal R4 Pendente Documental (Critério Omitido no Objeto)',
            'Sinal R5 Confirmado (Execução Continuada / Sob Demanda)',
            'Sinal R6 Revisar/Confirmado (Regime Especial / Cabimento)'
        ])
        quantidades.extend([
            int((df_ana['PRIORIDADE_QUALITATIVA'] == 'Q1 — Muito alta').sum()),
            int((df_ana['PRIORIDADE_QUALITATIVA'] == 'Q2 — Alta').sum()),
            int((df_ana['PRIORIDADE_QUALITATIVA'] == 'Q3 — Média').sum()),
            int((df_ana['PRIORIDADE_QUALITATIVA'] == 'Q4 — Monitoramento').sum()),
            int(df_ana['NECESSITA_REVISAO_MANUAL'].fillna(False).astype(bool).sum()),
            int(df_ana['D1_ENQUADRAMENTO'].fillna(False).astype(bool).sum()),
            int(df_ana['D2_EXECUCAO'].fillna(False).astype(bool).sum()),
            int(df_ana['D3_PRECIFICACAO'].fillna(False).astype(bool).sum()),
            int(df_ana['D4_DISTRIBUICAO'].fillna(False).astype(bool).sum()),
            int(df_ana['D5_RECORRENCIA'].fillna(False).astype(bool).sum()),
            int((df_ana['R0_STATUS'] == 'CONFIRMADO').sum()),
            int((df_ana['R1_STATUS'] == 'CONFIRMADO').sum()),
            int((df_ana['R2_STATUS'] == 'CONFIRMADO').sum()),
            int((df_ana['R3_STATUS'] == 'CONFIRMADO').sum()),
            int((df_ana['R4_STATUS'] == 'CONFIRMADO').sum()),
            int((df_ana['R4_STATUS'] == 'PENDENTE DOCUMENTAL').sum()),
            int((df_ana['R5_STATUS'] == 'CONFIRMADO').sum()),
            int(df_ana['R6_STATUS'].isin(['CONFIRMADO', 'REVISAR']).sum())
        ])

    df_resumo = pd.DataFrame({'Métrica': metricas, 'Quantidade': quantidades})
    
    # Sub-filtros para as abas
    df_grau_a = df_validado[df_validado['GI - Grau de Evidência'] == 'GRAU A — EVIDÊNCIA DIRETA']
    df_grau_b = df_validado[df_validado['GI - Grau de Evidência'] == 'GRAU B — SINTOMÁTICO']
    
    df_corroborados = df_ana[df_ana['GI - Corroboração Utilizada'].notna()]
    
    mask_alerta = (
        mask_val_pendente
        | (df_ana['GI - Grau de Evidência'] == 'GRAU D — TEXTO VAGO/GENÉRICO')
        | (df_ana['GI - Divergência Cadastral Aparente'] == True)
    )
    df_alertas = df_ana[mask_alerta]
    
    # Exclusões (Grau C) separadas entre Materialidade/Singularidade e Menção Incidental
    mask_mencao_incidental = df_ana['GI - Fundamentação Contextual'].fillna('').str.contains('incidental', case=False)
    df_exc_mat = df_ana[mask_excluido & (~mask_mencao_incidental)]
    df_exc_inc = df_ana[mask_excluido & mask_mencao_incidental]
    
    # Recorte da Base Bruta
    protocolos_validados = df_validado['Protocolo'].unique()
    df_recorte_bruto = df_bruta[df_bruta['Protocolo'].isin(protocolos_validados)]

    # 10_Matriz_Triagem_R0_R6 (Seção 9 da Especificação Funcional)
    colunas_matriz_secao_9 = [
        'Protocolo', 'No.Licitação', 'Jurisdicionado', 'Município', 'Modalidade',
        'Tipo do Objeto', 'Objeto', 'Valor', 'Qtd_Linhas_Origem',
        'GI - Resultado Etapa 1', 'GI - Grau de Evidência', 'GI - Divergência Cadastral Aparente',
        'GI - Regime Jurídico Informado'
    ] + colunas_r0_r6_ordenadas
    colunas_matriz_presentes = [c for c in colunas_matriz_secao_9 if c in df_ana.columns]

    # Filtra registros do universo relevante (INCLUÍDO, VALIDAÇÃO PENDENTE ou qualquer sinal ativo Q1–Q3)
    mask_relevante_matriz = (
        mask_incluido
        | mask_val_pendente
        | (df_ana.get('PRIORIDADE_QUALITATIVA', pd.Series('Q4 — Monitoramento', index=df_ana.index)) != 'Q4 — Monitoramento')
    )
    df_matriz_r0_r6 = df_ana.loc[mask_relevante_matriz, colunas_matriz_presentes].copy()
    if 'PRIORIDADE_QUALITATIVA' in df_matriz_r0_r6.columns and 'QTD_DIMENSOES' in df_matriz_r0_r6.columns:
        df_matriz_r0_r6 = df_matriz_r0_r6.sort_values(
            by=['PRIORIDADE_QUALITATIVA', 'QTD_DIMENSOES'],
            ascending=[True, False]
        )
    
    # Mapeamento do nome da aba (máx 31 caracteres no Excel) para o respectivo df
    abas = {
        '01_Resumo_Executivo': df_resumo,
        '02_Universo_Validado': df_validado,
        '03_Grau_A_Evidencia': df_grau_a,
        '04_Grau_B_Sintomatico': df_grau_b,
        '05_Corroborados_Python': df_corroborados,
        '06_Valid_Pendente_Alertas': df_alertas,
        '07_Exclusoes_Materialidade': df_exc_mat,
        '08_Falsos_Positivos': df_exc_inc,
        '09_Base_Bruta_Recorte': df_recorte_bruto,
        '10_Matriz_Triagem_R0_R6': df_matriz_r0_r6
    }
    
    # 3. REGRAS DE QUALIDADE E FORMATAÇÃO
    with pd.ExcelWriter(nome_arquivo, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formatações
        fmt_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        fmt_padrao = workbook.add_format({'valign': 'top'})
        fmt_string = workbook.add_format({'num_format': '@', 'valign': 'top'})
        fmt_alerta_fraude = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        
        for nome_aba, df_aba in abas.items():
            if df_aba.empty:
                df_aba = pd.DataFrame({'Mensagem': ['Nenhum registro atende aos critérios desta aba.']})
                
            df_aba.to_excel(writer, sheet_name=nome_aba, index=False)
            worksheet = writer.sheets[nome_aba]
            
            # Congela o cabeçalho
            worksheet.freeze_panes(1, 0)
            
            # Aplica Filtros Automáticos e Formatação de Colunas
            if not df_aba.empty and 'Mensagem' not in df_aba.columns:
                linhas = len(df_aba)
                colunas = len(df_aba.columns)
                worksheet.autofilter(0, 0, linhas, colunas - 1)
                
                # Ajuste Automático de Colunas
                for col_idx, col_name in enumerate(df_aba.columns):
                    # Proteção do tipo Protocolo para evitar perda de zeros
                    if col_name == 'Protocolo':
                        worksheet.set_column(col_idx, col_idx, 25, fmt_string)
                        continue
                        
                    # Estima tamanho máximo de forma vetorizada e segura contra NaN/pd.NA
                    tamanho_cabecalho = len(str(col_name))
                    tamanho_dado_max = df_aba[col_name].head(1000).fillna('').astype(str).str.len().max()
                    if pd.isna(tamanho_dado_max):
                        tamanho_dado_max = 0
                    tamanho_dado_max = int(tamanho_dado_max)
                    
                    tamanho_ideal = max(tamanho_cabecalho, tamanho_dado_max) + 2
                    
                    if tamanho_ideal > 70:
                        worksheet.set_column(col_idx, col_idx, 70, fmt_wrap)
                    else:
                        worksheet.set_column(col_idx, col_idx, tamanho_ideal, fmt_padrao)
                
                # Destaque de Fraude (Formatação Condicional) nas abas 02, 06, 07 e 10
                if nome_aba in ['02_Universo_Validado', '06_Valid_Pendente_Alertas', '07_Exclusoes_Materialidade', '10_Matriz_Triagem_R0_R6'] and 'GI - Divergência Cadastral Aparente' in df_aba.columns:
                    col_idx_div = df_aba.columns.get_loc('GI - Divergência Cadastral Aparente')
                    
                    def col_num_to_letter(n):
                        string = ""
                        while n >= 0:
                            string = chr(n % 26 + 65) + string
                            n = n // 26 - 1
                        return string
                        
                    letra_col = col_num_to_letter(col_idx_div)
                    
                    formula = f'=${letra_col}2=TRUE'
                    worksheet.conditional_format(1, 0, linhas, colunas - 1, {
                        'type': 'formula',
                        'criteria': formula,
                        'format': fmt_alerta_fraude
                    })

    logging.info(f"Fase 5 concluída com sucesso. Entregável Excel salvo: '{nome_arquivo}'.")
    return nome_arquivo

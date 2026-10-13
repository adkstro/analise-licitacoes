import pandas as pd
import re
import os
from datetime import datetime

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
FILE_PATH = r"c:\Users\akendall\Downloads\licitações ia\Base dedados de Licitação em 17-09.xlsx"
OUTPUT_DIR = r"c:\Users\akendall\Downloads\licitações ia"
TCE_NAME = "TCE-PB" # Based on the cities in sample
DATE_STR = datetime.now().strftime("%Y%m%d")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"Etapa_1_Universo_Validado_Credenciamentos_{TCE_NAME}_{DATE_STR}.xlsx")

def read_data(file_path):
    print("Lendo a base de dados...")
    df = pd.read_excel(file_path)
    return df

def get_col(df, possible_names):
    for col in df.columns:
        if col.strip().lower() in [n.lower() for n in possible_names]:
            return col
    return None

def analyze(df):
    print(f"Total de linhas originais: {len(df)}")
    if 'Protocolo' not in df.columns:
        raise ValueError("A coluna obrigatória 'Protocolo' não foi encontrada.")
    
    total_protocolos = df['Protocolo'].nunique()
    print(f"Total de Protocolos distintos: {total_protocolos}")
    
    # Identify duplicated lines
    protocolo_counts = df['Protocolo'].value_counts()
    repetidos = protocolo_counts[protocolo_counts > 1].index
    linhas_repetidas_count = len(df[df['Protocolo'].isin(repetidos)])
    print(f"Protocolos repetidos em múltiplas linhas: {len(repetidos)}")
    
    # Identify columns
    col_modalidade = get_col(df, ['Modalidade', 'Tipo da Licitação', 'Tipo de Modalidade'])
    col_objeto = get_col(df, ['Objeto', 'Descrição', 'Descrição do Objeto'])
    col_municipio = get_col(df, ['Município', 'Cidade', 'Municipio'])
    col_jurisdicionado = get_col(df, ['Jurisdicionado', 'Órgão', 'Orgao', 'Unidade Gestora', 'Ente'])
    
    # Consolidating into 1 row per Protocolo
    # For text fields, we can take the first value since they are usually identical for the same process
    # If there are variations, we will just use the first to represent the Protocolo's core characteristics.
    df_grouped = df.groupby('Protocolo', as_index=False).first()
    
    # Prepare analytical columns
    df_grouped['GI - Resultado Etapa 1'] = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'
    df_grouped['GI - Critério de Identificação'] = ''
    df_grouped['GI - Forma de Enquadramento'] = ''
    df_grouped['GI - Grau de Evidência'] = ''
    df_grouped['GI - Evidência Encontrada'] = ''
    df_grouped['GI - Fundamentação Contextual'] = ''
    df_grouped['GI - Regime Jurídico Informado'] = ''
    df_grouped['GI - Divergência Cadastral Aparente'] = ''
    df_grouped['GI - Corroboração Utilizada'] = ''
    df_grouped['GI - Protocolo Relacionado'] = ''
    df_grouped['GI - Necessita Validação Manual'] = 'Não'
    df_grouped['GI - Observação'] = ''
    
    # Helper to check text
    def contains(text, patterns):
        if not isinstance(text, str): return False
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def get_regime(modalidade_text):
        if not isinstance(modalidade_text, str): return 'Não Identificado'
        if '14.133' in modalidade_text or '14133' in modalidade_text: return 'Lei 14.133/2021'
        if '8.666' in modalidade_text or '8666' in modalidade_text or '10.520' in modalidade_text or '10520' in modalidade_text: return 'Lei 8.666/1993 e correlatas'
        if '13.303' in modalidade_text or '13303' in modalidade_text: return 'Lei 13.303/2016'
        return 'Outro / Não Identificado'

    # Step 5: Identification Rules
    # A - Modalidade
    mod_patterns = [r'credenciamento', r'procedimento auxiliar', r'chamamento']
    # B - Objeto
    obj_patterns = [r'credenciamento\s+de', r'credenciamento\s+para', r'credenciamento\s+visando', 
                    r'edital\s+de\s+credenciamento', r'procedimento\s+de\s+credenciamento', 
                    r'processo\s+de\s+credenciamento', r'contratação\s+de\s+credenciados',
                    r'profissionais\s+credenciados', r'empresas\s+credenciadas', 
                    r'pessoas?\s+físicas?\s+credenciadas?', r'pessoas?\s+jurídicas?\s+credenciadas?']
    # C - Decorrente
    dec_patterns = [r'conforme\s+credenciamento', r'oriundo\s+do\s+credenciamento',
                    r'decorrente\s+do\s+credenciamento', r'proveniente\s+do\s+credenciamento',
                    r'contratado.*credenciamento', r'credenciado.*edital', r'previamente\s+credenciado']
    # D - Fundamentos
    leg_patterns = [r'art\.?\s*74\s*,\s*iv', r'art\.?\s*74\s*inciso\s*iv', r'art\.?\s*79', 
                    r'contratação\s+paralela', r'não\s+excludente', r'seleção\s+a\s+critério\s+de\s+terceiros',
                    r'mercado\s+fluido']

    # Falsos positivos
    fp_patterns = [r'rede\s+credenciada', r'postos?\s+credenciados?', r'oficinas?\s+credenciadas?', 
                   r'estabelecimentos?\s+comerciais\s+credenciados?', r'hospital\s+credenciado', 
                   r'empresa\s+credenciada\s+pelo\s+fabricante', r'conselho\s+profissional',
                   r'crea', r'órgão\s+regulador', r'senatran', r'certificadora',
                   r'credenciais\s+de\s+acesso', r'participantes\s+de\s+evento',
                   r'cadastrar\s+ou\s+credenciar\s+terceiros']

    # Falsos negativos check patterns
    fn_patterns = [r'credenc', r'art\.?\s*74\s*,?\s*iv', r'art\.?\s*79']
    
    # Initialize counts
    candidatos = 0
    inc_a = 0
    inc_b = 0
    inc_c = 0
    val_pend = 0
    exc_inc = 0
    exc_sem = 0
    fn_ident = 0
    
    for idx, row in df_grouped.iterrows():
        modalidade = str(row[col_modalidade]) if col_modalidade and pd.notna(row[col_modalidade]) else ""
        objeto = str(row[col_objeto]) if col_objeto and pd.notna(row[col_objeto]) else ""
        
        regime = get_regime(modalidade)
        df_grouped.at[idx, 'GI - Regime Jurídico Informado'] = regime
        
        # Testar portas de entrada
        has_mod = contains(modalidade, mod_patterns)
        has_obj = contains(objeto, obj_patterns)
        has_dec = contains(objeto, dec_patterns)
        has_leg = contains(objeto, leg_patterns)
        
        is_candidate = has_mod or has_obj or has_dec or has_leg
        
        if is_candidate:
            candidatos += 1
            # Check falsos positivos (Menção Incidental)
            is_fp = contains(objeto, fp_patterns) and not (has_mod or has_obj or has_dec)
            if is_fp:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO — MENÇÃO INCIDENTAL'
                df_grouped.at[idx, 'GI - Observação'] = 'Possível menção incidental baseada em termos como rede credenciada/oficina etc.'
                exc_inc += 1
                continue
                
            # Classificação
            if has_mod:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
                df_grouped.at[idx, 'GI - Critério de Identificação'] = 'Modalidade Declarada'
                df_grouped.at[idx, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
                df_grouped.at[idx, 'GI - Evidência Encontrada'] = f"Modalidade: {modalidade}"
                inc_a += 1
            elif has_obj or has_dec:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
                df_grouped.at[idx, 'GI - Critério de Identificação'] = 'Objeto que Evidencia Credenciamento / Contratação Decorrente'
                df_grouped.at[idx, 'GI - Grau de Evidência'] = 'GRAU B — EVIDÊNCIA EXPLÍCITA/CONTEXTUAL'
                df_grouped.at[idx, 'GI - Evidência Encontrada'] = f"Termos no objeto: {objeto[:100]}..."
                df_grouped.at[idx, 'GI - Fundamentação Contextual'] = 'O texto do objeto descreve materialmente um processo de credenciamento ou contratação decorrente.'
                inc_b += 1
            elif has_leg:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'VALIDAÇÃO PENDENTE'
                df_grouped.at[idx, 'GI - Critério de Identificação'] = 'Fundamentos Jurídicos ou Sinais do Instituto'
                df_grouped.at[idx, 'GI - Grau de Evidência'] = 'GRAU D — VALIDAÇÃO PENDENTE'
                df_grouped.at[idx, 'GI - Evidência Encontrada'] = f"Fundamentação legal identificada: {objeto[:100]}..."
                df_grouped.at[idx, 'GI - Necessita Validação Manual'] = 'Sim'
                df_grouped.at[idx, 'GI - Observação'] = 'Revisar se a menção aos arts 74 IV ou 79 realmente configura o procedimento no caso concreto.'
                val_pend += 1
            else:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'VALIDAÇÃO PENDENTE'
                val_pend += 1
                
        else:
            # Check False Negatives
            if contains(objeto, fn_patterns) or contains(modalidade, fn_patterns):
                # We found a potential false negative
                fn_ident += 1
                # Let's see if we should include it
                if contains(objeto, fp_patterns):
                    df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO — MENÇÃO INCIDENTAL'
                    df_grouped.at[idx, 'GI - Observação'] = 'Identificado no teste de falsos negativos, mas trata-se de menção incidental.'
                    exc_inc += 1
                else:
                    df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'VALIDAÇÃO PENDENTE'
                    df_grouped.at[idx, 'GI - Critério de Identificação'] = 'Busca Reversa (Teste de Falsos Negativos)'
                    df_grouped.at[idx, 'GI - Grau de Evidência'] = 'GRAU D — VALIDAÇÃO PENDENTE'
                    df_grouped.at[idx, 'GI - Necessita Validação Manual'] = 'Sim'
                    val_pend += 1
            else:
                df_grouped.at[idx, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'
                exc_sem += 1
                
        # Divergência Cadastral
        if df_grouped.at[idx, 'GI - Resultado Etapa 1'] == 'INCLUÍDO':
            if contains(modalidade, [r'pregão', r'concorrência', r'dispensa']):
                df_grouped.at[idx, 'GI - Divergência Cadastral Aparente'] = 'Objeto indica credenciamento, mas Modalidade difere.'

    # 02_Universo_Validado
    df_validado = df_grouped[df_grouped['GI - Resultado Etapa 1'] == 'INCLUÍDO']
    
    # Duplicate check on Universe
    duplicados_validado = df_validado['Protocolo'].duplicated().sum()
    
    # Create the writer
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Aba 01_Resumo
        resumo_data = {
            'Indicador': [
                'Total de linhas da base',
                'Protocolos distintos',
                'Protocolos repetidos em múltiplas linhas da fonte',
                'Candidatos inicialmente identificados',
                'Incluídos Grau A',
                'Incluídos Grau B',
                'Incluídos Grau C',
                'Validação Pendente',
                'Exclusões por Menção Incidental',
                'Exclusões por ausência de evidência material',
                'Falsos negativos identificados',
                'Universo final validado',
                'Duplicidades de Protocolo no universo final'
            ],
            'Quantidade': [
                len(df),
                total_protocolos,
                len(repetidos),
                candidatos,
                inc_a,
                inc_b,
                inc_c,
                val_pend,
                exc_inc,
                exc_sem,
                fn_ident,
                len(df_validado),
                duplicados_validado
            ]
        }
        pd.DataFrame(resumo_data).to_excel(writer, sheet_name='01_Resumo', index=False)
        
        # 02_Universo_Validado
        df_validado.to_excel(writer, sheet_name='02_Universo_Validado', index=False)
        
        # 03_Grau_A
        df_validado[df_validado['GI - Grau de Evidência'] == 'GRAU A — EVIDÊNCIA DIRETA'].to_excel(writer, sheet_name='03_Grau_A', index=False)
        
        # 04_Grau_B
        df_validado[df_validado['GI - Grau de Evidência'] == 'GRAU B — EVIDÊNCIA EXPLÍCITA/CONTEXTUAL'].to_excel(writer, sheet_name='04_Grau_B', index=False)
        
        # 05_Grau_C
        df_validado[df_validado['GI - Grau de Evidência'] == 'GRAU C — EVIDÊNCIA CORROBORADA'].to_excel(writer, sheet_name='05_Grau_C', index=False)
        
        # 06_Validacao_Pendente
        df_grouped[df_grouped['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'].to_excel(writer, sheet_name='06_Validacao_Pendente', index=False)
        
        # 07_Exclusoes_Contextuais
        df_excl = df_grouped[df_grouped['GI - Resultado Etapa 1'].str.contains('EXCLUÍDO', na=False)]
        df_excl.to_excel(writer, sheet_name='07_Exclusoes_Contextuais', index=False)
        
        # 08_Controle_Negativo
        df_controle = df_grouped[df_grouped['GI - Resultado Etapa 1'] == 'EXCLUÍDO — MENÇÃO INCIDENTAL']
        df_controle.to_excel(writer, sheet_name='08_Controle_Negativo', index=False)
        
        # 09_Todos_Candidatos
        # Candidatos are those where we found something, or false negatives
        df_todos_cand = df_grouped[df_grouped['GI - Resultado Etapa 1'] != 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL']
        df_todos_cand.to_excel(writer, sheet_name='09_Todos_Candidatos', index=False)
        
        # 10_Base_Fonte_Recorte
        protocolos_candidatos = df_todos_cand['Protocolo'].unique()
        df_fonte_recorte = df[df['Protocolo'].isin(protocolos_candidatos)]
        df_fonte_recorte.to_excel(writer, sheet_name='10_Base_Fonte_Recorte', index=False)
        
        # 11_Metodologia
        metodologia = [
            f"Arquivo analisado: Base dedados de Licitação em 17-09.xlsx",
            f"Data da análise: {DATE_STR}",
            f"Chave primária: Protocolo",
            f"Quantidade de linhas na fonte: {len(df)}",
            f"Quantidade de Protocolos distintos: {total_protocolos}",
            "Critérios de identificação: Regras léxicas sobre Objeto e Modalidade.",
            "Graus de evidência: A (Modalidade), B (Objeto claro), C (Corroboração), D (Pendente).",
            "Falsos positivos: Tratamento para Menção Incidental (ex: rede credenciada).",
            "Falsos negativos: Busca reversa por expressões root.",
            "Teste de duplicidade: Protocolos únicos forçados por agregação 'first'."
        ]
        pd.DataFrame({"Metodologia Aplicada": metodologia}).to_excel(writer, sheet_name='11_Metodologia', index=False)
        
    print(f"\nArquivo '{OUTPUT_FILE}' gerado com sucesso!")
    print(f"Total Incluídos: {len(df_validado)}")
    print(f"Duplicidade no Universo: {duplicados_validado}")
    
    return OUTPUT_FILE

if __name__ == "__main__":
    df_raw = read_data(FILE_PATH)
    analyze(df_raw)
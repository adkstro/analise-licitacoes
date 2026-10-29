import warnings
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

import json
import logging
import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Força o urllib3 interno a ignorar o erro do proxy/firewall corporativo
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
requests.packages.urllib3.disable_warnings()

# Hack agressivo na sessão do Requests: desativa o `verify` em tempo de execução
old_merge_environment_settings = requests.Session.merge_environment_settings
def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings
requests.Session.merge_environment_settings = merge_environment_settings

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constante de configuração vetorial
SEMANTIC_THRESHOLD = 0.35

# Caminho padrão para a tabela parametrizável de regras R0 a R6
CAMINHO_REGRAS_R0_R6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regras_triagem_r0_r6.json')


def carregar_regras_r0_r6(caminho: str = CAMINHO_REGRAS_R0_R6) -> List[Dict[str, Any]]:
    """Carrega a tabela parametrizável de regras R0–R6 a partir do arquivo JSON."""
    if os.path.exists(caminho):
        try:
            with open(caminho, mode='r', encoding='utf-8') as f:
                dados = json.load(f)
                return [r for r in dados.get('regras', []) if r.get('ativo', True)]
        except Exception as e:
            logging.warning(f"Falha ao ler '{caminho}': {e}. Utilizando regras padrão em memória.")
    return []


def _extrair_primeira_evidencia(serie_texto: pd.Series, padrao_regex: str) -> pd.Series:
    """Extrai de forma vetorizada o primeiro trecho correspondente à expressão regular."""
    regex_captura = f"(?i)({padrao_regex})"
    extraido = serie_texto.str.extract(regex_captura, expand=False)
    if isinstance(extraido, pd.DataFrame):
        extraido = extraido.iloc[:, 0]
    return extraido.str.strip()


def executar_camada_1_r0_r6(
    df_analytical: pd.DataFrame,
    mask_vinculo_grau_a: pd.Series,
    citacao_extraida: pd.Series,
    mask_agri_grau_a: pd.Series
) -> pd.DataFrame:
    """
    Camada 1 do Motor de Triagem R0 a R6 (Determinística e Vetorizada):
    Detecta candidatos, aplica exceções óbvias de contexto e registra evidência e regra acionada.
    """
    regras = carregar_regras_r0_r6()
    regras_por_id = {r['regra_id']: r for r in regras}

    s_objeto = df_analytical['Objeto'].fillna('').astype(str)
    s_modalidade = df_analytical['Modalidade'].fillna('').astype(str)
    s_tipo_objeto = (
        df_analytical['Tipo do Objeto'].fillna('').astype(str)
        if 'Tipo do Objeto' in df_analytical.columns
        else pd.Series('', index=df_analytical.index, dtype='object')
    )
    s_regime = (
        df_analytical['GI - Regime Jurídico Informado'].fillna('').astype(str)
        if 'GI - Regime Jurídico Informado' in df_analytical.columns
        else pd.Series('', index=df_analytical.index, dtype='object')
    )
    texto_completo = s_modalidade + ' ' + s_tipo_objeto + ' ' + s_objeto

    mod_eh_credenciamento = s_modalidade.str.contains(r'credenciamento|chamamento|chamada\s+publica', case=False, regex=True)
    mod_eh_inexigibilidade = s_modalidade.str.contains(r'inexigibilidade', case=False, regex=True)
    mod_compativel = mod_eh_credenciamento | mod_eh_inexigibilidade

    # -------------------------------------------------------------------------
    # SINAL R0: DIVERGÊNCIA DE ENQUADRAMENTO OU DE PROCEDIMENTO
    # -------------------------------------------------------------------------
    # Caso 0A: Objeto individualiza número do credenciamento originário e modalidade é compatível -> AFASTADO
    mask_r0_afastado_vinculo = mask_vinculo_grau_a & mod_compativel
    if mask_r0_afastado_vinculo.any():
        df_analytical.loc[mask_r0_afastado_vinculo, 'R0_STATUS'] = 'AFASTADO'
        df_analytical.loc[mask_r0_afastado_vinculo, 'R0_EVIDENCIA'] = citacao_extraida.loc[mask_r0_afastado_vinculo].str.strip()
        df_analytical.loc[mask_r0_afastado_vinculo, 'R0_REGRA'] = (
            'Vínculo originário expresso: Objeto individualiza o número do procedimento de credenciamento em modalidade compatível'
        )

    # Caso 0B: R0_001 - Modalidade Credenciamento mas Objeto é Pré-Qualificação -> CONFIRMADO
    r0_001 = regras_por_id.get('R0_001', {
        'padrao_regex': r'PRE[\s\-]?QUALIFICAC[AÃ]O',
        'descricao_regra': 'Modalidade cadastrada como Credenciamento, mas o Objeto descreve procedimento de pré-qualificação'
    })
    ev_r0_001 = _extrair_primeira_evidencia(s_objeto, r0_001['padrao_regex'])
    mask_r0_001 = mod_eh_credenciamento & ev_r0_001.notna()
    if mask_r0_001.any():
        df_analytical.loc[mask_r0_001, 'R0_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r0_001, 'R0_EVIDENCIA'] = ev_r0_001.loc[mask_r0_001]
        df_analytical.loc[mask_r0_001, 'R0_REGRA'] = r0_001['descricao_regra']

    # Caso 0C: R0_002 - Inexigibilidade menciona credenciamento sem número do procedimento originário -> CONFIRMADO
    r0_002 = regras_por_id.get('R0_002', {
        'padrao_regex': r'CREDENCIAMENTO|CREDENCIAD[AO]S?',
        'excecao_regex': r'REDE\s+CREDENCIADA|OFICINAS?\s+CREDENCIADAS?|CREDENCIAD[AO]\s+PELO\s+FABRICANTE',
        'descricao_regra': 'Contratação por inexigibilidade menciona credenciamento sem individualizar o número do procedimento originário'
    })
    ev_r0_002 = _extrair_primeira_evidencia(s_objeto, r0_002['padrao_regex'])
    exc_r0_002 = s_objeto.str.contains(r0_002['excecao_regex'], case=False, regex=True) if r0_002.get('excecao_regex') else False
    mask_r0_002 = (
        (df_analytical['R0_STATUS'] == 'NÃO IDENTIFICADO')
        & mod_eh_inexigibilidade
        & (~mask_vinculo_grau_a)
        & ev_r0_002.notna()
        & (~exc_r0_002)
    )
    if mask_r0_002.any():
        df_analytical.loc[mask_r0_002, 'R0_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r0_002, 'R0_EVIDENCIA'] = ev_r0_002.loc[mask_r0_002]
        df_analytical.loc[mask_r0_002, 'R0_REGRA'] = r0_002['descricao_regra']

    # Caso 0D: R0_003 - Modalidade diferente de Credenciamento/Inexigibilidade com evidência material -> CONFIRMADO
    r0_003 = regras_por_id.get('R0_003', {
        'padrao_regex': r'CREDENCIAMENTO|CHAMAMENTO\s+PUBLICO|TABELA\s+(?:DO\s+)?SUS|SISTEMA\s+DE\s+RODIZIO',
        'excecao_regex': r'REDE\s+CREDENCIADA|OFICINAS?\s+CREDENCIADAS?|HOSPITAL\s+CREDENCIADO|CREDENCIAD[AO]\s+PELO\s+FABRICANTE',
        'descricao_regra': 'Modalidade diferente de Credenciamento com evidência material de credenciamento no Objeto'
    })
    ev_r0_003 = _extrair_primeira_evidencia(s_objeto, r0_003['padrao_regex'])
    exc_r0_003 = s_objeto.str.contains(r0_003['excecao_regex'], case=False, regex=True) if r0_003.get('excecao_regex') else False
    mask_r0_003 = (
        (df_analytical['R0_STATUS'] == 'NÃO IDENTIFICADO')
        & (~mod_compativel)
        & ((ev_r0_003.notna() & (~exc_r0_003)) | mask_vinculo_grau_a | mask_agri_grau_a)
    )
    if mask_r0_003.any():
        evidencia_comb = ev_r0_003.fillna(citacao_extraida).fillna('AGRICULTURA FAMILIAR / PNAE')
        df_analytical.loc[mask_r0_003, 'R0_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r0_003, 'R0_EVIDENCIA'] = (
            s_modalidade.loc[mask_r0_003] + ' vs ' + evidencia_comb.loc[mask_r0_003].astype(str)
        )
        df_analytical.loc[mask_r0_003, 'R0_REGRA'] = r0_003['descricao_regra']

    # Caso 0E: R0_004 - Regime jurídico defasado (Lei 8.666/93 ou 10.520/02) -> REVISAR
    r0_004 = regras_por_id.get('R0_004', {
        'padrao_regex': r'8\.?666|10\.?520',
        'descricao_regra': 'Regime jurídico informado refere-se a legislação revogada (diferente da Lei nº 14.133/2021)'
    })
    cond_regime_revogado = s_regime.str.contains(r0_004['padrao_regex'], case=False, regex=True)
    mask_r0_004 = (df_analytical['R0_STATUS'] == 'NÃO IDENTIFICADO') & cond_regime_revogado
    if mask_r0_004.any():
        df_analytical.loc[mask_r0_004, 'R0_STATUS'] = 'REVISAR'
        df_analytical.loc[mask_r0_004, 'R0_EVIDENCIA'] = s_regime.loc[mask_r0_004]
        df_analytical.loc[mask_r0_004, 'R0_REGRA'] = r0_004['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R1: INDÍCIO DE FORÇA DE TRABALHO OPERACIONAL
    # -------------------------------------------------------------------------
    r1_001 = regras_por_id.get('R1_001', {
        'padrao_regex': (
            r'\b(?:MOTORISTAS?|PEDREIROS?|SERVENTES?|PINTOR(?:ES)?|JARDINEIROS?|'
            r'OPERADOR(?:ES)?(?:\s+DE\s+MAQUINAS?)?|SERVICOS\s+GERAIS|AUXILIAR(?:ES)?\s+DE\s+SERVICOS\s+GERAIS|'
            r'MERENDEIRAS?|PORTEIROS?|VIGIAS?|ZELADOR(?:ES)?|COVEIROS?|GARIS|RECEPCIONISTAS?|'
            r'ELETRICISTAS?|ENCANADOR(?:ES)?|CARPINTEIROS?)\b'
        ),
        'excecao_regex': (
            r'(?:LOCAC[AÃ]O|FRETAMENTO|HORAS?[\s/\-]+MAQUINAS?|CAMINH[OÕ]ES|VEICULOS?|RETROESCAVADEIRA|TRATOR|'
            r'MOTONIVELADORA|TRANSPORTE).*?(?:COM\s+MOTORISTA|COM\s+OPERADOR|MOTORISTA[/\s]OPERADOR)|'
            r'(?:COM\s+MOTORISTA|COM\s+OPERADOR).*?(?:CONTA\s+DA\s+CONTRATADA|INCLUS[AO])|'
            r'PLANT[OÕ]ES\s+MEDICOS|CONSULTAS\s+MEDICAS|FORNECIMENTO\s+DE\s+REFEIC[OÕ]ES|BUFFET|'
            r'EXECUC[AÃ]O\s+DE\s+OBRA|CONSTRUC[AÃ]O\s+DE\s+(?:UNIDADES|ESCOLA|PRACA|UBS)'
        ),
        'descricao_regra': 'Função operacional identificada como possível objeto principal da contratação (substituição de pessoal)'
    })
    ev_r1 = _extrair_primeira_evidencia(s_objeto, r1_001['padrao_regex'])
    mask_r1_match = ev_r1.notna()
    if mask_r1_match.any():
        exc_r1 = s_objeto.str.contains(r1_001['excecao_regex'], case=False, regex=True) if r1_001.get('excecao_regex') else pd.Series(False, index=df_analytical.index)
        mask_r1_afastado = mask_r1_match & exc_r1
        mask_r1_candidato = mask_r1_match & (~exc_r1)

        if mask_r1_afastado.any():
            df_analytical.loc[mask_r1_afastado, 'R1_STATUS'] = 'AFASTADO'
            df_analytical.loc[mask_r1_afastado, 'R1_EVIDENCIA'] = ev_r1.loc[mask_r1_afastado]
            df_analytical.loc[mask_r1_afastado, 'R1_REGRA'] = (
                'Exceção Contextual: Menção a função operacional é meramente acessória a locação de equipamento/veículo, transporte ou obra'
            )

        if mask_r1_candidato.any():
            df_analytical.loc[mask_r1_candidato, 'R1_STATUS'] = 'CANDIDATO'
            df_analytical.loc[mask_r1_candidato, 'R1_EVIDENCIA'] = ev_r1.loc[mask_r1_candidato]
            df_analytical.loc[mask_r1_candidato, 'R1_REGRA'] = r1_001['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R2: INDÍCIO DE OBRAS, ENGENHARIA OU MANUTENÇÃO PREDIAL
    # -------------------------------------------------------------------------
    r2_001 = regras_por_id.get('R2_001', {
        'padrao_regex': (
            r'\b(?:CONSTRUC[AÃ]O\s+CIVIL|CONSTRUC[AÃ]O\s+DE|REFORMA\s+(?:DE|PREDIAL|GERAL)|PAVIMENTAC[AÃ]O|'
            r'ASFALTO|MANUTENC[AÃ]O\s+PREDIAL|MANUTENC[AÃ]O\s+DE\s+PREDIOS\s+PUBLICOS|CONSERVAC[AÃ]O\s+CIVIL|'
            r'EXECUC[AÃ]O\s+DE\s+OBRAS?|SERVICOS\s+DE\s+ENGENHARIA|TERRAPLANAGEM|DRENAGEM\s+PLUVIAL)\b'
        ),
        'excecao_regex': (
            r'AQUISIC[AÃ]O\s+(?:DE\s+)?MATERIA(?:L|IS)\s+DE\s+CONSTRUC[AÃ]O|PRE[\s\-]?QUALIFICAC[AÃ]O|'
            r'LOCAC[AÃ]O\s+DE\s+(?:EQUIPAMENTOS?|MAQUINAS?|VEICULOS?)|HORAS?[\s/\-]+MAQUINAS?|'
            r'ELABORAC[AÃ]O\s+DE\s+PROJETOS?\s+DE\s+ENGENHARIA|ASSESSORIA\s+EM\s+ENGENHARIA'
        ),
        'descricao_regra': 'Núcleo do objeto relacionado a execução de obra, engenharia ou manutenção civil'
    })
    ev_r2_obj = _extrair_primeira_evidencia(s_objeto, r2_001['padrao_regex'])
    ev_r2_tipo = _extrair_primeira_evidencia(s_tipo_objeto, r'OBRAS?\s+E\s+SERVICOS\s+DE\s+ENGENHARIA|\bOBRAS?\b|\bENGENHARIA\b')
    ev_r2 = ev_r2_obj.fillna(ev_r2_tipo)
    mask_r2_match = ev_r2.notna()

    if mask_r2_match.any():
        exc_r2 = s_objeto.str.contains(r2_001['excecao_regex'], case=False, regex=True) if r2_001.get('excecao_regex') else pd.Series(False, index=df_analytical.index)
        mask_r2_afastado = mask_r2_match & exc_r2
        mask_r2_candidato = mask_r2_match & (~exc_r2)

        if mask_r2_afastado.any():
            df_analytical.loc[mask_r2_afastado, 'R2_STATUS'] = 'AFASTADO'
            df_analytical.loc[mask_r2_afastado, 'R2_EVIDENCIA'] = ev_r2.loc[mask_r2_afastado]
            df_analytical.loc[mask_r2_afastado, 'R2_REGRA'] = (
                'Exceção Contextual: Aquisição isolada de material de construção, locação de máquinas, projeto intelectual ou pré-qualificação'
            )

        if mask_r2_candidato.any():
            df_analytical.loc[mask_r2_candidato, 'R2_STATUS'] = 'CANDIDATO'
            df_analytical.loc[mask_r2_candidato, 'R2_EVIDENCIA'] = ev_r2.loc[mask_r2_candidato]
            df_analytical.loc[mask_r2_candidato, 'R2_REGRA'] = r2_001['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R3: MERCADO FLUIDO / PRECIFICAÇÃO VARIÁVEL
    # -------------------------------------------------------------------------
    r3_001 = regras_por_id.get('R3_001', {
        'padrao_regex': (
            r'\b(?:MERCADO\s+FLUIDO|PRECOS?\s+VARIAVE(?:L|IS)|COTAC[AÃ]O\s+PERIODICA|ATUALIZAC[AÃ]O\s+DE\s+PRECOS?|'
            r'TABELA\s+DINAMICA|DESCONTO\s+SOBRE\s+(?:A\s+)?TABELA|MAIOR\s+PERCENTUAL\s+DE\s+DESCONTO\s+SOBRE\s+TABELA|'
            r'TABELA\s+ABC\s*FARMA|TABELA\s+CMED|TABELA\s+ANP|TABELA\s+SINAPI)\b'
        ),
        'descricao_regra': 'Mecanismo de precificação variável ou índice/tabela dinâmica explicitamente mencionado no objeto'
    })
    ev_r3_001 = _extrair_primeira_evidencia(s_objeto, r3_001['padrao_regex'])
    mask_r3_001 = ev_r3_001.notna()
    if mask_r3_001.any():
        df_analytical.loc[mask_r3_001, 'R3_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r3_001, 'R3_EVIDENCIA'] = ev_r3_001.loc[mask_r3_001]
        df_analytical.loc[mask_r3_001, 'R3_REGRA'] = r3_001['descricao_regra']

    r3_002 = regras_por_id.get('R3_002', {
        'padrao_regex': r'\b(?:PRECO\s+VIGENTE|PRECOS?\s+DE\s+MERCADO|CONFORME\s+TABELA\s+VIGENTE|INDICE\s+DE\s+ATUALIZAC[AÃ]O)\b',
        'excecao_regex': r'TABELA\s+(?:DO\s+)?SUS|TABELA\s+FIXA',
        'descricao_regra': 'Há referência a tabela ou preço vigente, mas a dinâmica de atualização não é suficientemente compreensível no objeto'
    })
    ev_r3_002 = _extrair_primeira_evidencia(s_objeto, r3_002['padrao_regex'])
    exc_r3_002 = s_objeto.str.contains(r3_002['excecao_regex'], case=False, regex=True) if r3_002.get('excecao_regex') else False
    mask_r3_002 = (df_analytical['R3_STATUS'] == 'NÃO IDENTIFICADO') & ev_r3_002.notna() & (~exc_r3_002)
    if mask_r3_002.any():
        df_analytical.loc[mask_r3_002, 'R3_STATUS'] = 'PENDENTE DOCUMENTAL'
        df_analytical.loc[mask_r3_002, 'R3_EVIDENCIA'] = ev_r3_002.loc[mask_r3_002]
        df_analytical.loc[mask_r3_002, 'R3_REGRA'] = r3_002['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R4: CRITÉRIO OBJETIVO DE DISTRIBUIÇÃO DA DEMANDA
    # -------------------------------------------------------------------------
    r4_001 = regras_por_id.get('R4_001', {
        'padrao_regex': (
            r'\b(?:SISTEMA\s+DE\s+RODIZIO|RODIZIO\s+(?:PERIODICO|ENTRE|DE)|ORDEM\s+DE\s+CONVOCAC[AÃ]O|'
            r'DISTRIBUIC[AÃ]O\s+(?:ADMINISTRATIVA|EQUITATIVA|IGUALITARIA)|SORTEIO\s+(?:PUBLICO|ENTRE)|'
            r'ESCALA\s+DE\s+(?:RODIZIO|ATENDIMENTO|REVEZAMENTO)|SISTEMA\s+DE\s+ALTERNANCIA|'
            r'CRITERIO\s+DE\s+ALOCAC[AÃ]O\s+DA\s+DEMANDA)\b'
        ),
        'descricao_regra': 'Evidência explícita de rodízio, sorteio, escala ou critério objetivo de alocação da demanda entre credenciados'
    })
    ev_r4_001 = _extrair_primeira_evidencia(s_objeto, r4_001['padrao_regex'])
    mask_r4_001 = ev_r4_001.notna()
    if mask_r4_001.any():
        df_analytical.loc[mask_r4_001, 'R4_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r4_001, 'R4_EVIDENCIA'] = ev_r4_001.loc[mask_r4_001]
        df_analytical.loc[mask_r4_001, 'R4_REGRA'] = r4_001['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R5: EXECUÇÃO CONTINUADA OU SOB DEMANDA
    # -------------------------------------------------------------------------
    r5_001 = regras_por_id.get('R5_001', {
        'padrao_regex': (
            r'\b(?:CONFORME\s+(?:A\s+)?DEMANDA|QUANDO\s+HOUVER\s+DEMANDA|SOB\s+DEMANDA|QUANDO\s+NECESSARIO|'
            r'CONFORME\s+(?:A\s+)?NECESSIDADE|EVENTUAL\s+CONTRATAC[AÃ]O|FUTURA\s+E\s+EVENTUAL|'
            r'DURANTE\s+12\s*\(?DOZE\)?\s*MESES|PELO\s+PERIODO\s+DE\s+12\s*\(?DOZE\)?\s*MESES|'
            r'PRESTAC[AÃ]O\s+CONTINUADA|SERVICOS\s+CONTINUOS|EXECUC[AÃ]O\s+PERIODICA|SUCESSIVAMENTE|'
            r'DURANTE\s+A\s+VIGENCIA|PARCELADAMENTE)\b'
        ),
        'excecao_regex': (
            r'\b(?:EVENTO\s+UNICO|AQUISIC[AÃ]O\s+PONTUAL|OBRA\s+UNICA|ENTREGA\s+IMEDIATA\s+E\s+INTEGRAL|'
            r'APRESENTAC[AÃ]O\s+UNICA|SHOW\s+ARTISTICO\s+DA\s+(?:BANDA|DUPLA|CANTOR))\b'
        ),
        'descricao_regra': 'Objeto apresenta expressões de execução continuada ou acionamento recorrente sob demanda'
    })
    ev_r5_pos = _extrair_primeira_evidencia(s_objeto, r5_001['padrao_regex'])
    ev_r5_exc = _extrair_primeira_evidencia(s_objeto, r5_001['excecao_regex']) if r5_001.get('excecao_regex') else pd.Series(pd.NA, index=df_analytical.index)

    mask_r5_afastado = ev_r5_exc.notna()
    mask_r5_confirmado = (~mask_r5_afastado) & ev_r5_pos.notna()

    if mask_r5_afastado.any():
        df_analytical.loc[mask_r5_afastado, 'R5_STATUS'] = 'AFASTADO'
        df_analytical.loc[mask_r5_afastado, 'R5_EVIDENCIA'] = ev_r5_exc.loc[mask_r5_afastado]
        df_analytical.loc[mask_r5_afastado, 'R5_REGRA'] = (
            'Exceção Contextual: Contratação pontual para evento único, apresentação específica ou entrega imediata'
        )
    if mask_r5_confirmado.any():
        df_analytical.loc[mask_r5_confirmado, 'R5_STATUS'] = 'CONFIRMADO'
        df_analytical.loc[mask_r5_confirmado, 'R5_EVIDENCIA'] = ev_r5_pos.loc[mask_r5_confirmado]
        df_analytical.loc[mask_r5_confirmado, 'R5_REGRA'] = r5_001['descricao_regra']

    # -------------------------------------------------------------------------
    # SINAL R6: CABIMENTO MATERIAL OU REGIME ESPECIAL A VALIDAR
    # -------------------------------------------------------------------------
    r6_001 = regras_por_id.get('R6_001', {
        'padrao_regex': (
            r'\b(?:13\.?303|13\.?019|ORGANIZAC[OÕ]ES\s+DA\s+SOCIEDADE\s+CIVIL|\bOSCS?\b|'
            r'PRE[\s\-]?QUALIFICAC[AÃ]O|PERMISS[AÃ]O\s+DE\s+USO|CESS[AÃ]O\s+DE\s+USO\s+DE\s+BOX|'
            r'LEILOEIROS?\s+OFICIA(?:L|IS)|INSTITUIC[OÕ]ES\s+FINANCEIRAS?.*?(?:CONSIGNADO|CREDITO))\b'
        ),
        'descricao_regra': 'Procedimento submetido a regime especial (Lei 13.303/16, Lei 13.019/14, pré-qualificação ou objeto atípico) que requer validação de cabimento'
    })
    ev_r6_001 = _extrair_primeira_evidencia(texto_completo, r6_001['padrao_regex'])
    mask_r6_001 = ev_r6_001.notna()
    if mask_r6_001.any():
        df_analytical.loc[mask_r6_001, 'R6_STATUS'] = 'REVISAR'
        df_analytical.loc[mask_r6_001, 'R6_EVIDENCIA'] = ev_r6_001.loc[mask_r6_001]
        df_analytical.loc[mask_r6_001, 'R6_REGRA'] = r6_001['descricao_regra']

    return df_analytical


def executar_fase_2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fase 2: Triagem em Cascata (Determinismo, Camada 1 R0–R6 e Vetorização).
    Objetivo: Classificar diretamente casos de evidência cabal (vínculo expresso a protocolo
    de credenciamento no objeto e aquisição de gêneros alimentícios da agricultura familiar/PNAE -> INCLUÍDO / GRAU A),
    executar a Camada 1 do Motor R0–R6, descartar ruídos e isolar a zona cinzenta para avaliação do LLM.
    """
    df_analytical = df.copy()

    # Garante que colunas textuais GI e R0–R6 existam e tenham dtype 'object' (evita LossySetitemError)
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
    
    # Prepara textos para busca combinada (Modalidade + Objeto)
    texto_busca = (df_analytical['Modalidade'].fillna('') + ' ' + df_analytical['Objeto'].fillna('')).astype(str)
    s_objeto = df_analytical['Objeto'].fillna('').astype(str)
    
    # Cria a flag temporária de candidato para a fila do LLM
    is_candidate = pd.Series(False, index=df_analytical.index)

    # -------------------------------------------------------------------------
    # 1A. INCLUSÃO DIRETA (GRAU A): VÍNCULO EXPRESSO A PROTOCOLO DE CREDENCIAMENTO NO OBJETO
    # -------------------------------------------------------------------------
    regex_vinculo_credenciamento = (
        r'(?i)((?:(?:EDITAL|PROCESSO(?:\s+AUXILIAR|\s+ADMINISTRATIVO)?|CHAMAMENTO\s+PUBLICO)\s+(?:DE\s+|[-–/]\s*)?)?'
        r'CREDENCIAMENT[O0]'
        r'(?:\s+(?:DE\s+(?:SERVICOS?|COMPRAS?|SAUDE|EXAMES?)|PUBLICO|ELETRONICO|ESPECIAL|MEDICO))?'
        r'\s*(?:N[O0º°\.]*|NUM(?:ERO)?\.?)?\s*[:\.\-]?\s*'
        r'((?:[A-Z0-9]{2,15}[/-])?\d{1,6}(?:\.\d{1,4})?\s*[/-]\s*(?:20)?\d{2,4}))'
    )

    extracao_vinculo = s_objeto.str.extract(regex_vinculo_credenciamento, expand=True)
    citacao_extraida = extracao_vinculo[0]
    protocolo_rel_extraido = extracao_vinculo[1]
    mask_vinculo_grau_a = citacao_extraida.notna()

    if mask_vinculo_grau_a.any():
        modalidade_compativel = df_analytical.loc[mask_vinculo_grau_a, 'Modalidade'].fillna('').str.contains(
            r'inexigibilidade|credenciamento|chamamento|chamada\s+publica', case=False, regex=True
        )
        divergencia_vinculo = (~modalidade_compativel).values

        citacao_formatada = citacao_extraida.loc[mask_vinculo_grau_a].str.strip().str.upper()
        numero_protocolo_rel = protocolo_rel_extraido.loc[mask_vinculo_grau_a].str.replace(r'\s+', '', regex=True).str.upper()

        justificativa_padrao = (
            'Incluído deterministicamente (Evidência Direta): O objeto vincula expressamente a contratação ao instrumento formal ('
            + citacao_formatada + ').'
        )
        justificativa_divergente = (
            'Incluído deterministicamente (Evidência Direta): O objeto vincula expressamente a contratação ao instrumento formal ('
            + citacao_formatada + '), porém a modalidade cadastrada diverge do instituto.'
        )

        df_analytical.loc[mask_vinculo_grau_a, 'GI - Fundamentação Contextual'] = np.where(
            divergencia_vinculo, justificativa_divergente, justificativa_padrao
        )
        df_analytical.loc[mask_vinculo_grau_a, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
        df_analytical.loc[mask_vinculo_grau_a, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
        df_analytical.loc[mask_vinculo_grau_a, 'GI - Divergência Cadastral Aparente'] = divergencia_vinculo
        df_analytical.loc[mask_vinculo_grau_a, 'GI - Protocolo Relacionado'] = numero_protocolo_rel

        logging.info(
            f"Filtro Direto Grau A (Vínculo a Protocolo de Credenciamento): {int(mask_vinculo_grau_a.sum())} "
            f"registros classificados diretamente como INCLUÍDO (Divergências: {int(divergencia_vinculo.sum())})."
        )

    # -------------------------------------------------------------------------
    # 1B. INCLUSÃO DIRETA (GRAU A): AQUISIÇÃO DE GÊNEROS ALIMENTÍCIOS DA AGRICULTURA FAMILIAR (PNAE / PAA)
    # -------------------------------------------------------------------------
    regex_alimentos_pnae = (
        r'GENEROS?\s+ALIMENTICIOS?|ALIMENTOS?|ALIMENTACAO\s+ESCOLAR|MERENDA\s+ESCOLAR|'
        r'HORTIFRUTI(?:GRANJEIROS?)?|OVOS|FRUTAS|VERDURAS|LEGUMES|PAES|BOLOS|\bPNAE\b|\bPAA\b'
    )
    regex_agricultura_familiar = (
        r'AGRICULTURA\s+FAMILIAR|EMPREENDEDOR(?:ES)?\s+FAMILIAR(?:ES)?(?:\s+RURA(?:L|IS))?|'
        r'PRODUTOR(?:ES)?\s+RURA(?:L|IS)|AGRICULTOR(?:ES)?\s+FAMILIAR(?:ES)?|'
        r'COOPERATIVAS?\s+(?:DE\s+|E\s+ASSOCIAC(?:AO|OES)\s+DE\s+)?PRODUTORES\s+RURAIS|'
        r'ASSOCIAC(?:AO|OES)\s+DE\s+PRODUTORES\s+RURAIS|CHAMADA\s+PUBLICA\s+DE\s+HORTIFRUTI|'
        r'LEI\s*(?:FEDERAL\s*)?(?:N[O0º°\.]*\s*)?11\.?947'
    )
    regex_excecoes_nao_alimentos = (
        r'CONSULTORIA|ASSESSORIA|APOIO\s+TECNICO|HORA\s+MAQUINA|TRATOR\s+AGRICOLA|ATACADISTA|DISTRIBUIDORA\s+COMERCIAL'
    )

    mask_ainda_nao_classificado = df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO'
    cond_alimentos = s_objeto.str.contains(regex_alimentos_pnae, case=False, regex=True)
    cond_agri_familiar = s_objeto.str.contains(regex_agricultura_familiar, case=False, regex=True)
    cond_sem_excecao = ~s_objeto.str.contains(regex_excecoes_nao_alimentos, case=False, regex=True)

    mask_agri_grau_a = mask_ainda_nao_classificado & cond_alimentos & cond_agri_familiar & cond_sem_excecao

    if mask_agri_grau_a.any():
        modalidade_compativel_agri = df_analytical.loc[mask_agri_grau_a, 'Modalidade'].fillna('').str.contains(
            r'inexigibilidade|credenciamento|chamamento|chamada\s+publica', case=False, regex=True
        )
        divergencia_agri = (~modalidade_compativel_agri).values

        justificativa_agri_regular = (
            'Incluído deterministicamente (Evidência Direta - Agricultura Familiar/PNAE): '
            'Aquisição de gêneros alimentícios da agricultura familiar/empreendedor familiar rural (Chamada Pública/Credenciamento).'
        )
        justificativa_agri_divergente = (
            'Incluído deterministicamente (Evidência Direta - Agricultura Familiar/PNAE): '
            'Aquisição de gêneros alimentícios da agricultura familiar/PNAE sob lógica de Chamada Pública/Credenciamento, '
            'porém cadastrada em modalidade divergente.'
        )

        df_analytical.loc[mask_agri_grau_a, 'GI - Fundamentação Contextual'] = np.where(
            divergencia_agri, justificativa_agri_divergente, justificativa_agri_regular
        )
        df_analytical.loc[mask_agri_grau_a, 'GI - Grau de Evidência'] = 'GRAU A — EVIDÊNCIA DIRETA'
        df_analytical.loc[mask_agri_grau_a, 'GI - Resultado Etapa 1'] = 'INCLUÍDO'
        df_analytical.loc[mask_agri_grau_a, 'GI - Divergência Cadastral Aparente'] = divergencia_agri

        logging.info(
            f"Filtro Direto Grau A (Agricultura Familiar / PNAE): {int(mask_agri_grau_a.sum())} "
            f"registros classificados diretamente como INCLUÍDO (Divergências: {int(divergencia_agri.sum())})."
        )

    # -------------------------------------------------------------------------
    # 1C. EXECUÇÃO DA CAMADA 1 DO MOTOR R0 A R6 (DETECÇÃO E EXCEÇÕES CONTEXTUAIS)
    # -------------------------------------------------------------------------
    df_analytical = executar_camada_1_r0_r6(
        df_analytical=df_analytical,
        mask_vinculo_grau_a=mask_vinculo_grau_a,
        citacao_extraida=citacao_extraida,
        mask_agri_grau_a=mask_agri_grau_a
    )
    
    # -------------------------------------------------------------------------
    # 2. A ROTA EXPRESSA (CANDIDATOS OBRIGATÓRIOS PARA A FILA DO LLM)
    # -------------------------------------------------------------------------
    mask_nao_classificado = df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO'

    cond_modalidade = df_analytical['Modalidade'].str.contains(r'inexigibilidade|credenciamento', case=False, na=False)
    
    # Citações legais referentes a credenciamento (Art. 79 da 14.133 e Art. 25 da 8.666)
    cond_leis = texto_busca.str.contains(r'art[\.\s]+79.*?14\.?133|art[\.\s]+25.*?8\.?666', case=False, regex=True, na=False)
    
    # Atualiza is_candidate apenas para quem ainda não foi classificado nos Passos 1A e 1B
    is_candidate = is_candidate | (mask_nao_classificado & (cond_modalidade | cond_leis))
    
    # -------------------------------------------------------------------------
    # 3. FALSOS POSITIVOS (MENÇÃO INCIDENTAL)
    # -------------------------------------------------------------------------
    regex_incidental = r'rede\s+credenciada|oficinas?\s+credenciadas?|hospital\s+credenciado|credenciad[ao]\s+pelo\s+fabricante'
    cond_incidental = s_objeto.str.contains(regex_incidental, case=False, regex=True, na=False)
    
    # Aplica exclusão apenas se ainda estiver NÃO CLASSIFICADO e não for candidato da Rota Expressa
    mask_excluir_incidental = mask_nao_classificado & cond_incidental & (~is_candidate)
    df_analytical.loc[mask_excluir_incidental, 'GI - Fundamentação Contextual'] = (
        'Excluído deterministicamente (Menção Incidental): O termo credenciado descreve apenas a rede/prestador e não a modalidade de contratação.'
    )
    df_analytical.loc[mask_excluir_incidental, 'GI - Grau de Evidência'] = 'GRAU C — SEM EVIDÊNCIA'
    df_analytical.loc[mask_excluir_incidental, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO'
    df_analytical.loc[mask_excluir_incidental, 'GI - Divergência Cadastral Aparente'] = False
    
    # -------------------------------------------------------------------------
    # 4. A PENEIRA GROSSA POSITIVA (SINTOMAS DA CAMUFLAGEM)
    # -------------------------------------------------------------------------
    sintomas = [
        # Operacionais
        r'tabelas?\s+de\s+preços?', r'tabela\s+(?:do\s+)?sus', r'valor(?:es)?\s+(?:fixados?|fixos?|prefixados?|preestabelecidos?)', 
        r'preços?\s+padronizados?', r'cachês?\s+(?:padronizados?|fixados?|preestabelecidos?)', r'sem\s+exclusão', 
        r'aberto\s+a\s+todos', r'pluralidade\s+de\s+(?:interessados|contratados|prestadores)', r'inviabilidade\s+de\s+competiç[ãa]o', 
        r'chamamento\s+público', r'(?:edital|aviso)\s+de\s+chamamento', r'forma\s+complementar', r'distribuição\s+equitativa', 
        r'sistema\s+de\s+rodízio', r'banco\s+de\s+(?:profissionais|fornecedores|prestadores)', r'cadastro\s+de\s+(?:prestadores|interessados|fornecedores)',
        # Saúde e Exames
        r'serviços?\s+(?:especializados?\s+)?de\s+saúde', r'consultas\s+médicas', r'exames\s+(?:de\s+ultrassonografia|laboratoriais|clínicos|laudos)', 
        r'laudos\s+médicos', r'próteses\s+dentárias', r'\bsus\b',
        # Eventos e Shows
        r'atraç(?:ão|ões)\s+artística?', r'shows?\s+(?:artístico|musical)', r'banda\s+[a-z]+', r'apresentação\s+(?:em\s+praça\s+pública|musical|artística)', 
        r'festividades?', r'são\s+joão', r'arraial', r'forró', r'festa\s+da\s+padroeira',
        # Serviços Sociais e Transporte
        r'oficineiros?', r'fretamento\s+de\s+veículos?',
        # Alimentação Direta Residual
        r'agricultura\s+familiar', r'empreendedor\s+familiar', r'pnae', r'alimentação\s+escolar'
    ]
    regex_sintomas = '|'.join(sintomas)
    
    # Só avalia quem ainda não é candidato e continua NÃO CLASSIFICADO
    mask_validos = (~is_candidate) & (df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO')
    cond_sintomas = texto_busca.str.contains(regex_sintomas, case=False, regex=True, na=False)
    
    is_candidate = is_candidate | (mask_validos & cond_sintomas)
    
    # -------------------------------------------------------------------------
    # 5. A PENEIRA GROSSA NEGATIVA (DESCARTE LÉXICO)
    # -------------------------------------------------------------------------
    descartes = [
        r'pavimentação', r'asfalto', r'material\s+de\s+expediente', r'combustível', r'peças\s+automotivas', 
        r'locação\s+de\s+veículos', r'aquisiç[ãa]o\s+de\s+equipamentos?', r'medicamentos?', r'material\s+médico', 
        r'manutenção\s+(?:preventiva|corretiva)', r'lixo\s+hospitalar', r'cestas\s+básicas', r'registro\s+de\s+preços?'
    ]
    regex_descarte = '|'.join(descartes)
    
    mask_avaliar_descarte = (~is_candidate) & (df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO')
    cond_descarte = s_objeto.str.contains(regex_descarte, case=False, regex=True, na=False)
    
    mask_excluir_descarte = mask_avaliar_descarte & cond_descarte
    df_analytical.loc[mask_excluir_descarte, 'GI - Fundamentação Contextual'] = (
        'Excluído deterministicamente (Descarte Léxico): Objeto singular e fechado incompatível com mercado fluido.'
    )
    df_analytical.loc[mask_excluir_descarte, 'GI - Grau de Evidência'] = 'GRAU C — SEM EVIDÊNCIA'
    df_analytical.loc[mask_excluir_descarte, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO'
    df_analytical.loc[mask_excluir_descarte, 'GI - Divergência Cadastral Aparente'] = False
    
    # -------------------------------------------------------------------------
    # 6. A PENEIRA FINA (BUSCA VETORIAL / EMBEDDINGS)
    # -------------------------------------------------------------------------
    mask_residuais = (~is_candidate) & (df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO')
    indices_residuais = df_analytical[mask_residuais].index
    
    if not indices_residuais.empty:
        logging.info(f"Processando {len(indices_residuais)} registros na Peneira Fina Semântica...")
        try:
            # Inicializa o modelo multilíngue otimizado para português
            modelo_nlp = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Frases Âncora (Target) definindo o conceito material de credenciamento
            ancoras = [
                "credenciamento de pessoas físicas ou jurídicas para prestação de serviços",
                "inexigibilidade de licitação por inviabilidade de competição devido à pluralidade de interessados",
                "chamamento público para cadastramento de fornecedores com preço fixado pela administração",
                "contratação mediante sistema de rodízio ou distribuição equitativa de demandas"
            ]
            
            # Codifica as âncoras e calcula o vetor médio que representa a intenção
            embeddings_ancoras = modelo_nlp.encode(ancoras, convert_to_tensor=True)
            embedding_target = embeddings_ancoras.mean(dim=0)
            
            # Codifica os objetos residuais em lote
            textos_residuais = df_analytical.loc[indices_residuais, 'Objeto'].fillna('').tolist()
            embeddings_residuais = modelo_nlp.encode(textos_residuais, convert_to_tensor=True)
            
            # Calcula similaridade do cosseno
            similaridades = util.cos_sim(embeddings_residuais, embedding_target).cpu().numpy().flatten()
            
            # Classifica com base no threshold configurável
            aprovados_vetorial = similaridades >= SEMANTIC_THRESHOLD
            indices_aprovados = indices_residuais[aprovados_vetorial]
            indices_reprovados = indices_residuais[~aprovados_vetorial]
            
            # Atualiza flags e status
            is_candidate.loc[indices_aprovados] = True
            df_analytical.loc[indices_reprovados, 'GI - Fundamentação Contextual'] = (
                'Excluído na Peneira Vetorial: Objeto singular sem aderência semântica ao instituto do credenciamento.'
            )
            df_analytical.loc[indices_reprovados, 'GI - Grau de Evidência'] = 'GRAU C — SEM EVIDÊNCIA'
            df_analytical.loc[indices_reprovados, 'GI - Resultado Etapa 1'] = 'EXCLUÍDO'
            df_analytical.loc[indices_reprovados, 'GI - Divergência Cadastral Aparente'] = False
            
        except Exception as e:
            logging.error(f"Erro na inferência vetorial: {e}. Os residuais não puderam ser vetorizados.")
            # Fallback seguro: se falhar a IA local, envia pra fila pra não perder possíveis fraudes
            is_candidate.loc[indices_residuais] = True

    # Para registros excluídos deterministicamente que NÃO pertencem ao universo de Credenciamento/Inexigibilidade,
    # afasta candidatos residuais R1/R2 (pois obras ou serviços em Concorrência/Pregão comum não são falso credenciamento)
    mask_excluidos_fora_escopo = (df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO') & (~cond_modalidade)
    mask_r1_fora = mask_excluidos_fora_escopo & (df_analytical['R1_STATUS'] == 'CANDIDATO')
    if mask_r1_fora.any():
        df_analytical.loc[mask_r1_fora, 'R1_STATUS'] = 'AFASTADO'
        df_analytical.loc[mask_r1_fora, 'R1_REGRA'] = (
            'Afastado na Camada 2: Processo em modalidade competitiva comum sem vínculo com credenciamento'
        )
    mask_r2_fora = mask_excluidos_fora_escopo & (df_analytical['R2_STATUS'] == 'CANDIDATO')
    if mask_r2_fora.any():
        df_analytical.loc[mask_r2_fora, 'R2_STATUS'] = 'AFASTADO'
        df_analytical.loc[mask_r2_fora, 'R2_REGRA'] = (
            'Afastado na Camada 2: Obra/engenharia licitada em modalidade competitiva comum sem vínculo com credenciamento'
        )
    
    # -------------------------------------------------------------------------
    # 7. ISOLAMENTO DA FILA SEMÂNTICA (ZONA CINZENTA)
    # -------------------------------------------------------------------------
    df_analytical.loc[is_candidate, 'GI - Grau de Evidência'] = 'NA — NÃO AVALIADO (FILA)'
    df_analytical.loc[is_candidate, 'GI - Resultado Etapa 1'] = 'FILA PARA LLM'
    
    # Extrai o subconjunto isolado para o LLM com as colunas obrigatórias + sinais candidatos R1 e R2
    colunas_fila = [
        'Protocolo', 'Modalidade', 'Objeto', 'GI - Regime Jurídico Informado',
        'Valor', 'R1_STATUS', 'R1_EVIDENCIA', 'R2_STATUS', 'R2_EVIDENCIA'
    ]
    colunas_presentes = [c for c in colunas_fila if c in df_analytical.columns]
    
    df_fila_llm = df_analytical.loc[is_candidate, colunas_presentes].copy()
    
    qtde_incluidos_direto = int((df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO').sum())
    qtde_excluidos = int((df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO').sum())
    logging.info(
        f"Fase 2 concluída. Incluídos Diretos (Grau A): {qtde_incluidos_direto} | "
        f"Descartados (Grau C): {qtde_excluidos} | Direcionados para LLM: {len(df_fila_llm)}"
    )
    
    return df_analytical, df_fila_llm

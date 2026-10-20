import pandas as pd
import numpy as np
import logging
import os
from typing import Tuple
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

def executar_fase_2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fase 2: Triagem em Cascata (Determinismo e Vetorização).
    Objetivo: Classificar diretamente casos de evidência cabal (vínculo expresso a protocolo
    de credenciamento no objeto -> INCLUÍDO / GRAU A), descartar ruídos e isolar a zona cinzenta
    para avaliação posterior do LLM.
    """
    df_analytical = df.copy()

    # Garante que colunas textuais GI tenham dtype 'object' (evita LossySetitemError quando iniciadas com NaN/float64)
    for col_gi in ['GI - Fundamentação Contextual', 'GI - Grau de Evidência', 'GI - Resultado Etapa 1', 'GI - Corroboração Utilizada', 'GI - Protocolo Relacionado']:
        if col_gi in df_analytical.columns:
            df_analytical[col_gi] = df_analytical[col_gi].astype('object')
    
    # Prepara textos para busca combinada (Modalidade + Objeto)
    texto_busca = (df_analytical['Modalidade'].fillna('') + ' ' + df_analytical['Objeto'].fillna('')).astype(str)
    
    # Cria a flag temporária de candidato para a fila do LLM
    is_candidate = pd.Series(False, index=df_analytical.index)

    # -------------------------------------------------------------------------
    # 1. INCLUSÃO DIRETA (GRAU A): VÍNCULO EXPRESSO A PROTOCOLO DE CREDENCIAMENTO NO OBJETO
    # -------------------------------------------------------------------------
    # Grupo 1: Citação completa (ex: "EDITAL DE CREDENCIAMENTO NO 002/2025")
    # Grupo 2: Número do protocolo/processo relacionado (ex: "002/2025", "00009/2025", "001-2026")
    regex_vinculo_credenciamento = (
        r'(?i)((?:(?:EDITAL|PROCESSO(?:\s+AUXILIAR|\s+ADMINISTRATIVO)?|CHAMAMENTO\s+PUBLICO)\s+(?:DE\s+|[-–/]\s*)?)?'
        r'CREDENCIAMENT[O0]'
        r'(?:\s+(?:DE\s+(?:SERVICOS?|COMPRAS?|SAUDE|EXAMES?)|PUBLICO|ELETRONICO|ESPECIAL|MEDICO))?'
        r'\s*(?:N[O0º°\.]*|NUM(?:ERO)?\.?)?\s*[:\.\-]?\s*'
        r'((?:[A-Z0-9]{2,15}[/-])?\d{1,6}(?:\.\d{1,4})?\s*[/-]\s*(?:20)?\d{2,4}))'
    )

    extracao_vinculo = df_analytical['Objeto'].fillna('').str.extract(regex_vinculo_credenciamento, expand=True)
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
    # 2. A ROTA EXPRESSA (CANDIDATOS OBRIGATÓRIOS PARA A FILA DO LLM)
    # -------------------------------------------------------------------------
    mask_nao_classificado = df_analytical['GI - Resultado Etapa 1'] == 'NÃO CLASSIFICADO'

    cond_modalidade = df_analytical['Modalidade'].str.contains(r'inexigibilidade|credenciamento', case=False, na=False)
    
    # Citações legais referentes a credenciamento (Art. 79 da 14.133 e Art. 25 da 8.666)
    cond_leis = texto_busca.str.contains(r'art[\.\s]+79.*?14\.?133|art[\.\s]+25.*?8\.?666', case=False, regex=True, na=False)
    
    # Atualiza is_candidate apenas para quem ainda não foi classificado no Passo 1
    is_candidate = is_candidate | (mask_nao_classificado & (cond_modalidade | cond_leis))
    
    # -------------------------------------------------------------------------
    # 3. FALSOS POSITIVOS (MENÇÃO INCIDENTAL)
    # -------------------------------------------------------------------------
    regex_incidental = r'rede\s+credenciada|oficinas?\s+credenciadas?|hospital\s+credenciado|credenciad[ao]\s+pelo\s+fabricante'
    cond_incidental = df_analytical['Objeto'].str.contains(regex_incidental, case=False, regex=True, na=False)
    
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
        # Alimentação Direta
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
    cond_descarte = df_analytical['Objeto'].str.contains(regex_descarte, case=False, regex=True, na=False)
    
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
    
    # -------------------------------------------------------------------------
    # 7. ISOLAMENTO DA FILA SEMÂNTICA (ZONA CINZENTA)
    # -------------------------------------------------------------------------
    df_analytical.loc[is_candidate, 'GI - Grau de Evidência'] = 'NA — NÃO AVALIADO (FILA)'
    df_analytical.loc[is_candidate, 'GI - Resultado Etapa 1'] = 'FILA PARA LLM'
    
    # Extrai o subconjunto isolado para o LLM com as colunas obrigatórias
    colunas_fila = ['Protocolo', 'Modalidade', 'Objeto', 'GI - Regime Jurídico Informado', 'Valor']
    
    # Garante que as colunas existem antes de filtrar
    colunas_presentes = [c for c in colunas_fila if c in df_analytical.columns]
    
    df_fila_llm = df_analytical.loc[is_candidate, colunas_presentes].copy()
    
    qtde_incluidos_direto = int((df_analytical['GI - Resultado Etapa 1'] == 'INCLUÍDO').sum())
    qtde_excluidos = int((df_analytical['GI - Resultado Etapa 1'] == 'EXCLUÍDO').sum())
    logging.info(
        f"Fase 2 concluída. Incluídos Diretos (Grau A): {qtde_incluidos_direto} | "
        f"Descartados (Grau C): {qtde_excluidos} | Direcionados para LLM: {len(df_fila_llm)}"
    )
    
    return df_analytical, df_fila_llm

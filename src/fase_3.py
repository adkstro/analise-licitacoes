import asyncio
import logging
import os
import pandas as pd
from typing import Literal, Optional
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. DEFINIÇÃO DO SCHEMA PYDANTIC (CHAIN OF THOUGHT)
class AnaliseCredenciamento(BaseModel):
    justificativa_sintetica: str = Field(..., description="Raciocínio passo a passo analisando a essência material do texto frente à modalidade informada. Máx 40 palavras.")
    divergencia_cadastral: bool = Field(..., description="True se a essência material do texto contradiz a Modalidade informada.")
    resultado_etapa_1: Literal['INCLUÍDO', 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL', 'EXCLUÍDO — MENÇÃO INCIDENTAL']
    grau_evidencia: Optional[Literal['GRAU A — EVIDÊNCIA DIRETA', 'GRAU B — SINTOMÁTICO', 'GRAU C — TEXTO VAGO/GENÉRICO']] = None

# 2. O SYSTEM PROMPT (MANUAL DO AUDITOR DE CONTROLE EXTERNO)
SYSTEM_PROMPT = """Você é um Auditor Sênior de Controle Externo de um Tribunal de Contas no Brasil, especialista em licitações (Lei 14.133/21). Sua missão é analisar textos sujos e confusos de contratações públicas para encontrar o uso (regular ou fraudulento) do instituto do 'Credenciamento' e da 'Inexigibilidade'.

O que você deve procurar (Os 3 Cenários do Art. 79 da Lei 14.133/21):

1. Regras de Credenciamento (Art. 79) — Foco na Pluralidade:
O credenciamento só deve ser utilizado quando a competição tradicional for inviável. Identifique se o objeto se enquadra em um destes cenários:

    Contratações paralelas e não excludentes: A demanda do órgão é tão grande que é mais vantajoso contratar todos os fornecedores possíveis por um preço fixo, distribuindo a demanda (ex: exames de laboratório, plantões médicos, rodízio de serviços).

    Seleção a critério de terceiros: O beneficiário final (cidadão/servidor) é quem escolhe o prestador credenciado (ex: vale-creche, vale-refeição, assistência social, óticas).

    Mercados fluidos: Preços que sofrem flutuações constantes, onde o pregão seria lento demais (ex: passagens aéreas, aquisição direta de produtos agrícolas/produtor rural).

2. Regras de Inexigibilidade (Art. 74) — Foco na Exclusividade e Singularidade:
A inexigibilidade ocorre quando a disputa é logicamente impossível. Identifique se o objeto se enquadra em:

    Fornecedor Exclusivo: Apenas uma empresa comercializa o bem/serviço. É fraude se houver exigência de marca específica apenas para simular exclusividade.

    Artista Consagrado: Contratação direta consagrada pela crítica/público, feita diretamente ou via empresário exclusivo. É fraude o uso de cartas de exclusividade restritas a um único dia/evento (atravessadores).

    Notória Especialização: Serviços técnicos predominantemente intelectuais e singulares (ex: pareceristas, auditorias, consultorias complexas). É fraude o seu uso para serviços comuns ou de publicidade e propaganda (proibido).

    Aquisição/Locação de Imóveis: Instalações ou localização geográfica tornam um imóvel específico a única opção viável.

Atenção à Padronização e Continuidade:
Não assuma que um serviço é contínuo e padronizado se não houver indícios claros no texto. Preste extrema atenção a termos que quebram a lógica do credenciamento, como contratações de caráter emergencial.

A inexigibilidade permite a contratação direta de um artista específico ou um profissional de notória especialização. Se o objeto não for único ou exclusivo, a inexigibilidade é inaplicável.

REGRAS ESPECÍFICAS PARA AGRICULTURA FAMILIAR (PNAE / PAA / CHAMADA PÚBLICA):
A aquisição de gêneros alimentícios da agricultura familiar e do empreendedor familiar rural (no âmbito do PNAE — Alimentação Escolar ou do PAA) é realizada por Chamada Pública/Credenciamento, caracterizada por preço fixado pelo órgão e pluralidade de agricultores/cooperativas credenciados:
1. Alinhamento Material: Se o objeto citar "agricultura familiar", "PNAE", "PAA", "empreendedor familiar rural", "chamada pública de hortifruti" ou "cooperativa/associação de produtores rurais":
   - Se a Modalidade for 'Credenciamento', 'Chamada Pública' ou 'Inexigibilidade/Dispensa', classifique estritamente como resultado_etapa_1 = 'INCLUÍDO' e grau_evidencia = 'GRAU A — EVIDÊNCIA DIRETA' (ou 'GRAU B' se o texto for sucinto).
2. Inexigibilidade/Dispensa Camuflada: Se a Modalidade estiver cadastrada como 'Dispensa', 'Inexigibilidade' ou 'Outras', mas o texto descrever aquisição de alimentos da agricultura familiar/PNAE em lote com múltiplos produtores sob tabela predefinida, marque resultado_etapa_1 = 'INCLUÍDO' e divergencia_cadastral = True (pois o gestor utilizou uma dispensa/inexigibilidade genérica para encobrir um procedimento com lógica de credenciamento público).
3. Falso Credenciamento na Agricultura: Se a Modalidade for 'Credenciamento', mas o objeto indicar a compra fechada/exclusiva com uma empresa atacadista/distribuidora comercial tradicional (que NÃO seja agricultor familiar ou cooperativa rural enquadrada legalmente), marque divergencia_cadastral = True e resultado_etapa_1 = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'.

O Ponto Crítico de Auditoria (A Caça à Fraude):
Prefeituras mentem no cadastro. Baseie-se SEMPRE na realidade material do texto e no preço. Como não há disputa de lances, o edital deve ser permanentemente aberto a novos interessados.

    Falso Credenciamento: Se a Modalidade é 'Credenciamento', mas o texto descreve uma compra fechada, obra, pavimentação, serviço com vencedor único, situação emergencial, ou apresenta barreiras artificiais de entrada para limitar o número de credenciados (direcionamento). Marque divergencia_cadastral = True e resultado_etapa_1 = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'.

    A Regra da Dúvida (In Dubio Pro Auditor): Prefeitos frequentemente usam textos extremamente vagos ou genéricos (ex: 'prestação de serviços técnicos', 'serviços diversos') para ocultar contratações milionárias sem licitação. Se o texto for vago demais e não permitir uma análise profunda, NÃO o exclua. Classifique estritamente como resultado_etapa_1 = 'INCLUÍDO' e atribua o grau_evidencia = 'GRAU C — TEXTO VAGO/GENÉRICO'.

    Inexigibilidade Camuflada: Se a Modalidade é 'Inexigibilidade' ou 'Dispensa', mas o texto descreve serviços em massa (plantões médicos sob demanda, shows para o São João, tabela do SUS), isso deveria ser um Credenciamento. Marque como INCLUÍDO e aponte divergencia_cadastral = True.

Regras de Classificação (Grau):

    GRAU A: Inequívoca pluralidade, tabela fixa, sem exclusividade no texto.

    GRAU B: Fortes indícios no jargão (ex: contratação de médicos em regime de plantão, atração musical para evento).

    GRAU C: Texto excessivamente vago ou genérico, não sendo possível cravar a natureza da contratação apenas com o objeto."""


@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5))
async def inferir_registro(client: genai.Client, protocolo: str, prompt_usuario: str, semaphore: asyncio.Semaphore) -> dict:
    """Executa a inferência na API protegida por retry e semáforo (concorrência)."""
    async with semaphore:
        try:
            # Novo padrão google-genai com suporte nativo a Pydantic
            resposta = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.1-flash-lite",
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AnaliseCredenciamento,
                    temperature=0.1
                )
            )
            
            # O objeto Pydantic instanciado fica em resposta.parsed
            analise = resposta.parsed
            
            return {
                'Protocolo': protocolo,
                'GI - Fundamentação Contextual': analise.justificativa_sintetica,
                'GI - Divergência Cadastral Aparente': analise.divergencia_cadastral,
                'GI - Resultado Etapa 1': analise.resultado_etapa_1,
                'GI - Grau de Evidência': analise.grau_evidencia if analise.grau_evidencia else pd.NA
            }
        except Exception as e:
            logging.error(f"Erro no Protocolo {protocolo}: {e}")
            raise e

# 3. MOTOR DE INFERÊNCIA ASSÍNCRONO
async def orquestrar_inferencia(df_fila: pd.DataFrame) -> pd.DataFrame:
    """Prepara as tarefas, envia em batch e agrupa os resultados."""
    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv()
    
    # Configuração do Gemini via novo SDK (google-genai) sem Instructor
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env.")
        
    client = genai.Client(api_key=api_key)
    
    # Limitador de concorrência (Rate Limits)
    semaphore = asyncio.Semaphore(10)
    tasks = []
    
    for _, row in df_fila.iterrows():
        prompt = (
            f"Protocolo: {row.get('Protocolo')}\n"
            f"Modalidade: {row.get('Modalidade')}\n"
            f"Objeto: {row.get('Objeto')}\n"
            f"Regime Jurídico: {row.get('GI - Regime Jurídico Informado')}\n"
            f"Valor Licitação: {row.get('Valor')}"
        )
        tasks.append(inferir_registro(client, row['Protocolo'], prompt, semaphore))
        
    logging.info(f"Iniciando inferência de {len(tasks)} registros no Gemini API...")
    
    # Dispara e aguarda todas as requisições
    resultados = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filtra os sucessos e avisa os erros irrecuperáveis
    sucessos = [r for r in resultados if isinstance(r, dict)]
    erros = len(resultados) - len(sucessos)
    
    if erros > 0:
        logging.warning(f"{erros} registros falharam após todas as tentativas de retry.")
        
    return pd.DataFrame(sucessos)

def executar_fase_3(df_analytical: pd.DataFrame, df_fila_llm: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 3: Função principal. Orquestra a execução assíncrona da inteligência artificial
    e aplica o merge (reintegração) ao dataframe principal.
    """
    if df_fila_llm.empty:
        logging.info("A fila do LLM está vazia. Nenhuma inferência será realizada.")
        return df_analytical
        
    # Tratamento para iniciar loop assíncrono dentro de scripts síncronos / notebooks
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        df_resultados = loop.run_until_complete(orquestrar_inferencia(df_fila_llm))
    except RuntimeError:
        df_resultados = asyncio.run(orquestrar_inferencia(df_fila_llm))
        
    if df_resultados.empty:
        logging.error("Nenhum resultado válido foi retornado pelo LLM.")
        return df_analytical
        
    # 4. REINTEGRAÇÃO E MERGE NO PANDAS
    logging.info("Realizando merge estruturado no DataFrame analítico...")
    
    # Seta o index como chave primária para garantir update perfeito
    reset_depois = False
    if 'Protocolo' in df_analytical.columns:
        df_analytical.set_index('Protocolo', inplace=True)
        reset_depois = True
        
    if 'Protocolo' in df_resultados.columns:
        df_resultados.set_index('Protocolo', inplace=True)
        
    protocolos_comuns = df_resultados.index.intersection(df_analytical.index)
    colunas_comuns = df_resultados.columns.intersection(df_analytical.columns)
    
    # Coerção defensiva de dtypes para evitar LossySetitemError/TypeError em colunas nulas
    for col in colunas_comuns:
        if df_analytical[col].dtype != df_resultados[col].dtype:
            df_analytical[col] = df_analytical[col].astype('object')
            
    # Atribuição direta garante que valores None/pd.NA também limpem o 'GRAU D'
    df_analytical.loc[protocolos_comuns, colunas_comuns] = df_resultados.loc[protocolos_comuns, colunas_comuns]
    
    if reset_depois:
        df_analytical.reset_index(inplace=True)
    
    logging.info("Fase 3 concluída com sucesso. Base analítica enriquecida com insights semânticos.")
    
    return df_analytical
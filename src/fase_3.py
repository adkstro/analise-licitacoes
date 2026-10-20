import asyncio
import json
import logging
import os
import time
from typing import Literal, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Arquivo padrão de persistência incremental (Checkpoint)
ARQUIVO_CHECKPOINT_PADRAO = "checkpoint_fase_3.jsonl"

# 1. DEFINIÇÃO DO SCHEMA PYDANTIC (CHAIN OF THOUGHT)
class AnaliseCredenciamento(BaseModel):
    justificativa_sintetica: str = Field(..., description="Raciocínio passo a passo analisando a essência material do texto frente à modalidade informada. Máx 40 palavras.")
    divergencia_cadastral: bool = Field(..., description="True se a essência material do texto contradiz a Modalidade informada.")
    resultado_etapa_1: Literal['INCLUÍDO', 'EXCLUÍDO', 'VALIDAÇÃO PENDENTE']
    grau_evidencia: Optional[Literal['GRAU A — INEQUÍVOCA', 'GRAU B — SINTOMÁTICO', 'GRAU C — SEM EVIDÊNCIA', 'GRAU D — TEXTO VAGO/GENÉRICO']] = None

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
    A inexigibilidade permite a contratação direta de um artista específico ou um profissional de notória especialização. Se o objeto não for único ou exclusivo, a inexigibilidade é inaplicável.

Atenção à Padronização e Continuidade:
Não assuma que um serviço é contínuo e padronizado se não houver indícios claros no texto. Preste extrema atenção a termos que quebram a lógica do credenciamento, como contratações de caráter emergencial.

REGRAS ESPECÍFICAS PARA AGRICULTURA FAMILIAR (PNAE / PAA / CHAMADA PÚBLICA):
A aquisição de gêneros alimentícios da agricultura familiar e do empreendedor familiar rural (no âmbito do PNAE — Alimentação Escolar ou do PAA) é realizada por Chamada Pública/Credenciamento, caracterizada por preço fixado pelo órgão e pluralidade de agricultores/cooperativas credenciados:
1. Alinhamento Material: Se o objeto citar "agricultura familiar", "PNAE", "PAA", "empreendedor familiar rural", "chamada pública de hortifruti" ou "cooperativa/associação de produtores rurais":
   - Se a Modalidade for 'Credenciamento', 'Chamada Pública' ou 'Inexigibilidade/Dispensa', classifique estritamente como resultado_etapa_1 = 'INCLUÍDO' e grau_evidencia = 'GRAU A — EVIDÊNCIA DIRETA' (ou 'GRAU B' se o texto for sucinto).
2. Inexigibilidade/Dispensa Camuflada: Se a Modalidade estiver cadastrada como 'Dispensa', 'Inexigibilidade' ou 'Outras', mas o texto descrever aquisição de alimentos da agricultura familiar/PNAE em lote com múltiplos produtores sob tabela predefinida, marque resultado_etapa_1 = 'INCLUÍDO' e divergencia_cadastral = True.
3. Falso Credenciamento na Agricultura: Se a Modalidade for 'Credenciamento', mas o objeto indicar a compra fechada/exclusiva com uma empresa atacadista/distribuidora comercial tradicional (que NÃO seja agricultor familiar ou cooperativa rural enquadrada legalmente), marque divergencia_cadastral = True e resultado_etapa_1 = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'.

O Ponto Crítico de Auditoria (A Caça à Fraude):
Prefeituras mentem no cadastro. Baseie-se SEMPRE na realidade material do texto e no preço. Como não há disputa de lances, o edital deve ser permanentemente aberto a novos interessados.
    Falso Credenciamento: Se a Modalidade é 'Credenciamento', mas o texto descreve uma compra fechada, obra, pavimentação, serviço com vencedor único, situação emergencial, ou apresenta barreiras artificiais de entrada para limitar o número de credenciados (direcionamento). Marque divergencia_cadastral = True e resultado_etapa_1 = 'EXCLUÍDO — SEM EVIDÊNCIA MATERIAL'.
    A Regra da Dúvida (In Dubio Pro Auditor): Prefeitos frequentemente usam textos extremamente vagos ou genéricos (ex: 'prestação de serviços técnicos', 'serviços diversos') para ocultar contratações milionárias sem licitação. Se o texto for vago demais e não permitir uma análise profunda, NÃO o exclua. Classifique estritamente como resultado_etapa_1 = 'INCLUÍDO' e atribua o grau_evidencia = 'GRAU C — TEXTO VAGO/GENÉRICO'.
    Inexigibilidade Camuflada: Se a Modalidade é 'Inexigibilidade' ou 'Dispensa', mas o texto descreve serviços em massa (plantões médicos sob demanda, shows para o São João, tabela do SUS), isso deveria ser um Credenciamento. Marque como INCLUÍDO e aponte divergencia_cadastral = True.

Regras de Classificação (Grau):
    GRAU A: Inequívoca pluralidade, tabela fixa, sem exclusividade no texto.
    GRAU B: Fortes indícios no jargão (ex: contratação de médicos em regime de plantão, atração musical para evento).
    GRAU C: O texto é perfeitamente claro, mas descreve algo singular incompatível com o instituto (uma obra fechada, aquisição de um carro específico). A justificativa_sintetica deve explicar por que a singularidade do objeto impede o credenciamento.
    GRAU D: Texto excessivamente vago ou genérico, não sendo possível cravar a natureza da contratação apenas com o objeto. Esta é a justificativa exata do roteamento "VALIDAÇÃO PENDENTE"."""

# 3. CONTROLADOR DE TAXA ASSÍNCRONO E CHECKPOINT INCREMENTAL
class RateLimiterAsync:
    """
    Controlador de cadência para respeitar o teto de 15 RPM da chave gratuita.
    Garante intervalo seguro entre chamadas (~4.28s para teto de 14 RPM).
    """
    def __init__(self, max_rpm: int = 14):
        self.intervalo = 60.0 / max_rpm
        self.ultimo_disparo = 0.0
        self.lock = asyncio.Lock()

    async def aguardar(self):
        async with self.lock:
            loop = asyncio.get_running_loop()
            agora = loop.time()
            decorrido = agora - self.ultimo_disparo
            if decorrido < self.intervalo:
                espera = self.intervalo - decorrido
                await asyncio.sleep(espera)
            self.ultimo_disparo = loop.time()


class GerenciadorCheckpoint:
    """
    Gerencia a leitura e gravação atômica em disco (JSONL) de cada protocolo
    processado, garantindo tolerância a falhas e retomada automática.
    """
    def __init__(self, caminho_arquivo: str = ARQUIVO_CHECKPOINT_PADRAO):
        self.caminho_arquivo = caminho_arquivo
        self.lock = asyncio.Lock()

    def carregar_processados(self) -> pd.DataFrame:
        """Lê os registros já salvos no arquivo de checkpoint, se existir."""
        if not os.path.exists(self.caminho_arquivo):
            return pd.DataFrame()

        registros = []
        try:
            with open(self.caminho_arquivo, mode="r", encoding="utf-8") as f:
                for linha in f:
                    linha_limpa = linha.strip()
                    if linha_limpa:
                        try:
                            registros.append(json.loads(linha_limpa))
                        except json.JSONDecodeError:
                            continue
            if not registros:
                return pd.DataFrame()

            df_ckpt = pd.DataFrame(registros)
            df_ckpt['Protocolo'] = df_ckpt['Protocolo'].astype(str)
            # Mantém a última versão válida caso haja reprocessamento
            df_ckpt = df_ckpt.drop_duplicates(subset=['Protocolo'], keep='last')
            df_ckpt['GI - Grau de Evidência'] = df_ckpt['GI - Grau de Evidência'].fillna(pd.NA)
            return df_ckpt
        except Exception as e:
            logging.warning(f"Não foi possível ler o arquivo de checkpoint ({self.caminho_arquivo}): {e}")
            return pd.DataFrame()

    async def salvar_registro(self, resultado: dict) -> None:
        """Grava imediatamente o resultado de um protocolo em disco de forma thread/async-safe."""
        async with self.lock:
            try:
                registro_serializavel = {
                    k: (None if pd.isna(v) else v) for k, v in resultado.items()
                }
                linha_json = json.dumps(registro_serializavel, ensure_ascii=False) + "\n"
                await asyncio.to_thread(self._anexar_em_disco, linha_json)
            except Exception as e:
                logging.error(f"Falha ao gravar checkpoint para protocolo {resultado.get('Protocolo')}: {e}")

    def _anexar_em_disco(self, linha: str) -> None:
        with open(self.caminho_arquivo, mode="a", encoding="utf-8") as f:
            f.write(linha)
            f.flush()


@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5), reraise=True)
async def _chamar_api_gemini(
    client: genai.Client,
    modelo: str,
    prompt_usuario: str,
    rate_limiter: Optional[RateLimiterAsync] = None
):
    """Realiza a chamada à API protegida por retry e pelo limitador de cadência."""
    if rate_limiter is not None:
        await rate_limiter.aguardar()

    return await asyncio.to_thread(
        client.models.generate_content,
        model=modelo,
        contents=prompt_usuario,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AnaliseCredenciamento,
            temperature=0.1
        )
    )


async def inferir_registro(
    client: genai.Client,
    modelo: str,
    protocolo: str,
    prompt_usuario: str,
    semaphore: asyncio.Semaphore,
    checkpoint: GerenciadorCheckpoint,
    rate_limiter: Optional[RateLimiterAsync] = None
) -> dict:
    """Executa a inferência na API e persiste o resultado imediatamente no checkpoint."""
    async with semaphore:
        try:
            resposta = await _chamar_api_gemini(client, modelo, prompt_usuario, rate_limiter)
            analise = resposta.parsed

            resultado = {
                'Protocolo': str(protocolo),
                'GI - Fundamentação Contextual': analise.justificativa_sintetica,
                'GI - Divergência Cadastral Aparente': bool(analise.divergencia_cadastral),
                'GI - Resultado Etapa 1': analise.resultado_etapa_1,
                'GI - Grau de Evidência': analise.grau_evidencia if analise.grau_evidencia else pd.NA
            }

            # Salva imediatamente no disco para resistir a quedas/interrupções
            await checkpoint.salvar_registro(resultado)
            return resultado

        except Exception as e:
            logging.error(f"Erro no Protocolo {protocolo}: {e}")
            raise e


# 4. MOTOR DE INFERÊNCIA ASSÍNCRONO COM RETOMADA AUTOMÁTICA
async def orquestrar_inferencia(
    df_fila: pd.DataFrame,
    modo_gratuito: bool = False,
    modelo: Optional[str] = None,
    limite_rpd: int = 500,
    caminho_checkpoint: str = ARQUIVO_CHECKPOINT_PADRAO
) -> pd.DataFrame:
    """
    Identifica protocolos já processados no checkpoint, filtra apenas os pendentes,
    controla limites de quota (RPM/RPD) e retorna o conjunto consolidado.
    """
    load_dotenv()

    checkpoint = GerenciadorCheckpoint(caminho_checkpoint)
    df_historico = checkpoint.carregar_processados()

    df_fila_copia = df_fila.copy()
    df_fila_copia['Protocolo'] = df_fila_copia['Protocolo'].astype(str)
    total_fila = len(df_fila_copia)

    # Verifica quais protocolos da fila atual já constam no checkpoint
    if not df_historico.empty:
        mask_ja_processados = df_fila_copia['Protocolo'].isin(df_historico['Protocolo'])
        qtde_ja_processados = int(mask_ja_processados.sum())
        df_pendentes = df_fila_copia[~mask_ja_processados].copy()
        logging.info(
            f"[CHECKPOINT] Progresso reconhecido: {qtde_ja_processados}/{total_fila} registros já processados. "
            f"Restam {len(df_pendentes)} registros pendentes."
        )
    else:
        df_pendentes = df_fila_copia
        logging.info(f"[CHECKPOINT] Nenhum histórico prévio encontrado. Total na fila: {total_fila} registros.")

    # Se todos os registros da fila já foram processados, retorna imediatamente sem chamar API
    if df_pendentes.empty:
        logging.info("Todos os registros da fila já foram processados anteriormente! Nenhuma chamada à API necessária.")
        return df_historico[df_historico['Protocolo'].isin(df_fila_copia['Protocolo'])].copy()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env.")

    client = genai.Client(api_key=api_key)

    # Identifica modo via parâmetro ou variável de ambiente
    if not modo_gratuito:
        env_modo = os.environ.get("GEMINI_MODO_GRATUITO", "false").strip().lower()
        modo_gratuito = env_modo in ("true", "1", "t", "sim", "yes")

    # Configuração de Modelo e Restrições de Quota sobre os registros PENDENTES
    if modo_gratuito:
        modelo_final = modelo or "gemini-3.1-flash-lite"
        total_pendentes = len(df_pendentes)
        if total_pendentes > limite_rpd:
            logging.warning(
                f"[MODO GRATUITO] Limitando execução atual de {total_pendentes} pendentes para os próximos "
                f"{limite_rpd} registros para respeitar a cota diária (500 RPD). "
                f"{total_pendentes - limite_rpd} registros ficarão para a próxima execução."
            )
            df_processar = df_pendentes.head(limite_rpd).copy()
        else:
            df_processar = df_pendentes.copy()

        semaphore = asyncio.Semaphore(1)
        rate_limiter = RateLimiterAsync(max_rpm=14)
        logging.info(f"[MODO GRATUITO ATIVO] Modelo: {modelo_final} | Limite: 15 RPM / 250K TPM / {limite_rpd} RPD.")
    else:
        modelo_final = modelo or "gemini-2.5-flash"
        df_processar = df_pendentes.copy()
        semaphore = asyncio.Semaphore(10)
        rate_limiter = None
        logging.info(f"[MODO PAGO ATIVO] Modelo: {modelo_final} | Concorrência paralela: 10 tarefas.")

    # Construção vetorizada dos prompts (sem .iterrows())
    prompts_series = (
        "Protocolo: " + df_processar['Protocolo'].fillna('').astype(str) + "\n"
        + "Modalidade: " + df_processar['Modalidade'].fillna('').astype(str) + "\n"
        + "Objeto: " + df_processar['Objeto'].fillna('').astype(str) + "\n"
        + "Regime Jurídico: " + df_processar['GI - Regime Jurídico Informado'].fillna('').astype(str) + "\n"
        + "Valor Licitação: " + df_processar['Valor'].fillna(0.0).astype(str)
    )

    tasks = [
        inferir_registro(client, modelo_final, proto, prompt, semaphore, checkpoint, rate_limiter)
        for proto, prompt in zip(df_processar['Protocolo'], prompts_series)
    ]

    logging.info(f"Iniciando inferência de {len(tasks)} registros pendentes no Gemini API ({modelo_final})...")

    try:
        resultados = await asyncio.gather(*tasks, return_exceptions=True)
        sucessos = [r for r in resultados if isinstance(r, dict)]
        erros = len(resultados) - len(sucessos)

        if erros > 0:
            logging.warning(
                f"{erros} registros falharam nesta execução. "
                f"Eles não foram gravados no checkpoint e serão reprocessados automaticamente na próxima execução."
            )
    except ( asyncio.CancelledError, KeyboardInterrupt ):
        logging.warning("Execução interrompida pelo usuário! Os registros concluídos até o momento foram preservados no checkpoint.")

    # Recarrega o estado consolidado diretamente do checkpoint atualizado
    df_consolidado = checkpoint.carregar_processados()
    if df_consolidado.empty:
        return pd.DataFrame()

    return df_consolidado[df_consolidado['Protocolo'].isin(df_fila_copia['Protocolo'])].copy()


# 5. FUNÇÃO PRINCIPAL DE ENTRADA DA FASE 3
def executar_fase_3(
    df_analytical: pd.DataFrame,
    df_fila_llm: pd.DataFrame,
    modo_gratuito: bool = False,
    modelo: Optional[str] = None,
    limite_rpd: int = 500,
    caminho_checkpoint: str = ARQUIVO_CHECKPOINT_PADRAO
) -> pd.DataFrame:
    """
    Fase 3: Julgamento Semântico via LLM (Gemini) com tolerância a interrupções.
    Reconhece automaticamente linhas já processadas (no DataFrame ou em disco via checkpoint)
    e continua exatamente de onde parou.
    """
    if df_fila_llm.empty:
        logging.info("A fila do LLM está vazia. Nenhuma inferência será realizada.")
        return df_analytical

    df_analytical_out = df_analytical.copy()
    df_analytical_out['Protocolo'] = df_analytical_out['Protocolo'].astype(str)

    df_fila_preparada = df_fila_llm.copy()
    df_fila_preparada['Protocolo'] = df_fila_preparada['Protocolo'].astype(str)

    # Reconhece registros que eventualmente já estejam preenchidos em memória no df_analytical
    status_pendentes = {'NÃO CLASSIFICADO', 'FILA PARA LLM'}
    mask_ja_avaliados_mem = (
        ~df_analytical_out['GI - Resultado Etapa 1'].isin(status_pendentes)
        & df_analytical_out['GI - Fundamentação Contextual'].notna()
    )
    protocolos_avaliados_mem = set(df_analytical_out.loc[mask_ja_avaliados_mem, 'Protocolo'])

    if protocolos_avaliados_mem:
        mask_em_fila = df_fila_preparada['Protocolo'].isin(protocolos_avaliados_mem)
        qtde_mem = int(mask_em_fila.sum())
        if qtde_mem > 0:
            logging.info(f"[MEMÓRIA] {qtde_mem} registros da fila já constam como processados no DataFrame analítico.")
            df_fila_preparada = df_fila_preparada[~mask_em_fila].copy()

    # Compatibilidade com event loops existentes (Jupyter Notebook / scripts síncronos)
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        df_resultados = loop.run_until_complete(
            orquestrar_inferencia(
                df_fila_preparada,
                modo_gratuito=modo_gratuito,
                modelo=modelo,
                limite_rpd=limite_rpd,
                caminho_checkpoint=caminho_checkpoint
            )
        )
    except RuntimeError:
        df_resultados = asyncio.run(
            orquestrar_inferencia(
                df_fila_preparada,
                modo_gratuito=modo_gratuito,
                modelo=modelo,
                limite_rpd=limite_rpd,
                caminho_checkpoint=caminho_checkpoint
            )
        )

    if df_resultados.empty:
        logging.warning("Nenhum resultado disponível para mesclagem nesta execução.")
        return df_analytical_out

    # Reintegração estruturada no DataFrame analítico
    logging.info(f"Realizando merge estruturado de {len(df_resultados)} registros no DataFrame analítico...")
    df_analytical_out.set_index('Protocolo', inplace=True)
    df_resultados.set_index('Protocolo', inplace=True)

    df_analytical_out.update(df_resultados)
    df_analytical_out.reset_index(inplace=True)

    assert df_analytical_out['Protocolo'].duplicated().sum() == 0, "Erro: Protocolos duplicados após reintegração da Fase 3."

    logging.info("Fase 3 concluída com sucesso. Base analítica enriquecida com insights semânticos.")
    return df_analytical_out

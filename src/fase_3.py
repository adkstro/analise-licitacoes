import asyncio
import json
import logging
import os
import sys
import time
from typing import Literal, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configuração de log principal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Silencia logs verbosos de bibliotecas internas (AFC, urllib3, httpx) para manter a barra limpa
for logger_name in ["google", "google.genai", "urllib3", "httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Arquivo padrão de persistência incremental (Checkpoint)
ARQUIVO_CHECKPOINT_PADRAO = "checkpoint_fase_3.jsonl"

# 1. DEFINIÇÃO DO SCHEMA PYDANTIC NA ORDEM ESTRITA DE RACIOCÍNIO (CHAIN OF THOUGHT)
class AnaliseCredenciamento(BaseModel):
    justificativa_sintetica: str = Field(
        ...,
        description="1. GI - Fundamentação Contextual: Raciocínio passo a passo analisando a essência material do texto frente à modalidade informada. Se Grau C, explique por que a singularidade impede o credenciamento. Se Grau D, aponte a omissão de detalhes. Máx 40 palavras."
    )
    grau_evidencia: Literal[
        'GRAU A — EVIDÊNCIA DIRETA',
        'GRAU B — SINTOMÁTICO',
        'GRAU C — SEM EVIDÊNCIA',
        'GRAU D — TEXTO VAGO/GENÉRICO'
    ] = Field(
        ...,
        description="2. GI - Grau de Evidência: Força da evidência material no texto (Grau A: prova direta/vínculo expresso; Grau B: indícios/jargões; Grau C: objeto singular/fechado sem evidência; Grau D: texto vago/genérico)."
    )
    resultado_etapa_1: Literal[
        'INCLUÍDO',
        'EXCLUÍDO',
        'VALIDAÇÃO PENDENTE'
    ] = Field(
        ...,
        description="3. GI - Resultado Etapa 1: Roteamento logístico derivado do Grau (Grau A ou B -> INCLUÍDO; Grau C -> EXCLUÍDO; Grau D -> VALIDAÇÃO PENDENTE)."
    )
    divergencia_cadastral: bool = Field(
        ...,
        description="4. GI - Divergência Cadastral Aparente: True quando o resultado_etapa_1 contradiz a modalidade cadastrada (ex: INCLUÍDO mas modalidade é Pregão; ou EXCLUÍDO mas modalidade é Credenciamento/Inexigibilidade). False quando concorda ou quando resultado_etapa_1 é VALIDAÇÃO PENDENTE (Grau D)."
    )

# 2. O SYSTEM PROMPT (MANUAL DO AUDITOR DE CONTROLE EXTERNO)
SYSTEM_PROMPT = """Você é um Auditor Sênior de Controle Externo de um Tribunal de Contas no Brasil, especialista em licitações (Lei 14.133/21 e Lei 8.666/93). Sua missão é analisar textos de contratações públicas para identificar a presença material do instituto do 'Credenciamento' e da 'Inexigibilidade', avaliando sua regularidade ou eventual fraude/divergência cadastral.

O que você deve procurar (Os Cenários Legais):

1. Regras de Credenciamento (Art. 79 da Lei 14.133/21) — Foco na Pluralidade e Mercado Fluido:
O credenciamento só deve ser utilizado quando a competição tradicional for inviável. Identifique se o objeto se enquadra em um destes cenários:
    - Contratações paralelas e não excludentes: A demanda do órgão é tão grande que é mais vantajoso contratar todos os fornecedores possíveis por um preço fixo/tabelado, distribuindo a demanda (ex: exames de laboratório, consultas/plantões médicos, tabela SUS, rodízio de serviços, oficineiros).
    - Seleção a critério de terceiros: O beneficiário final (cidadão/servidor) é quem escolhe o prestador credenciado (ex: vale-creche, vale-refeição, assistência social, óticas).
    - Mercados fluidos: Preços que sofrem flutuações constantes, onde o pregão seria lento demais (ex: passagens aéreas, aquisição direta de produtos agrícolas/produtor rural).
    - Vinculação expressa a processo originário: O texto vincula a contratação a um edital de credenciamento ou inexigibilidade já existente (ex: "...conforme Credenciamento nº 0008/2026").

2. Regras de Inexigibilidade (Art. 74 da Lei 14.133/21) — Foco na Exclusividade e Singularidade:
A inexigibilidade ocorre quando a disputa é logicamente impossível:
    - Fornecedor Exclusivo: Apenas uma empresa comercializa o bem/serviço. É fraude se houver exigência de marca específica apenas para simular exclusividade.
    - Artista Consagrado: Contratação direta de profissional do setor artístico consagrado pela crítica/público, feita diretamente ou via empresário exclusivo. É fraude o uso de cartas de exclusividade restritas a um único dia/evento (atravessadores).
    - Notória Especialização: Serviços técnicos predominantemente intelectuais e singulares (ex: pareceristas, auditorias, consultorias complexas). É proibido seu uso para serviços comuns ou de publicidade e propaganda.
    - Aquisição/Locação de Imóveis: Instalações ou localização geográfica tornam um imóvel específico a única opção viável.

3. Regras Específicas para Agricultura Familiar (PNAE / PAA / Chamada Pública):
A aquisição de gêneros alimentícios da agricultura familiar e do empreendedor familiar rural (no âmbito do PNAE — Alimentação Escolar ou do PAA) é realizada por Chamada Pública/Credenciamento, caracterizada por preço fixado pelo órgão e pluralidade de agricultores/cooperativas credenciados:
    - Alinhamento Material: Se o objeto citar "agricultura familiar", "PNAE", "PAA", "empreendedor familiar rural", "chamada pública de hortifruti" ou "cooperativa/associação de produtores rurais", classifique como GRAU A — EVIDÊNCIA DIRETA (ou GRAU B — SINTOMÁTICO se o texto for sucinto) e resultado_etapa_1 = 'INCLUÍDO'.
    - Falso Credenciamento na Agricultura: Se a Modalidade for 'Credenciamento', mas o objeto indicar compra fechada/exclusiva com empresa atacadista/distribuidora comercial tradicional (que NÃO seja agricultor familiar ou cooperativa rural), classifique como GRAU C — SEM EVIDÊNCIA, resultado_etapa_1 = 'EXCLUÍDO' e divergencia_cadastral = True.

=====================================================================
ARQUITETURA DE DECISÃO EM 3 EIXOS (PREENCHA NESTA EXATA ORDEM LÓGICA):
=====================================================================

PASSO 1 — Fundamentação Contextual (justificativa_sintetica):
Escreva o raciocínio analítico (máx. 40 palavras) confrontando a essência material do Objeto com a Modalidade informada.

PASSO 2 — A Métrica da Matéria (grau_evidencia):
Responda: O quão forte é a evidência da presença material de credenciamento/inexigibilidade no texto?
- 'GRAU A — EVIDÊNCIA DIRETA': O texto prova pluralidade, tabela fixa e ausência de exclusividade, ou inclui o protocolo em um processo de inexigibilidade/credenciamento já existente (ex: "...conforme credenciamento Nº 0008/2026"). Ação: Roteamento = INCLUÍDO.
- 'GRAU B — SINTOMÁTICO': O texto possui fortes indícios e jargões burocráticos do instituto (ex: plantões, shows de padroeira), mas faltam detalhes matemáticos. Ação: Roteamento = INCLUÍDO.
- 'GRAU C — SEM EVIDÊNCIA': O texto é claro, mas descreve algo singular e fechado (ex: obra, compra de carro específico, material de expediente), incompatível com mercado fluido. Ação: Roteamento = EXCLUÍDO. A justificativa_sintetica deve explicar por que a singularidade do objeto impede o credenciamento.
- 'GRAU D — TEXTO VAGO/GENÉRICO': O texto é tão genérico (ex: "serviços diversos") que impossibilita a inferência. Ação: Roteamento = VALIDAÇÃO PENDENTE. A justificativa_sintetica deve apontar a omissão de detalhes. Se for Grau D, obrigatoriamente marque divergencia_cadastral = False (na dúvida absoluta, não se presume fraude automática).

PASSO 3 — O Roteamento Logístico (resultado_etapa_1):
Derive matematicamente do Grau de Evidência atribuído no Passo 2:
- Se grau_evidencia for 'GRAU A — EVIDÊNCIA DIRETA' ou 'GRAU B — SINTOMÁTICO' -> resultado_etapa_1 = 'INCLUÍDO'.
- Se grau_evidencia for 'GRAU C — SEM EVIDÊNCIA' -> resultado_etapa_1 = 'EXCLUÍDO'.
- Se grau_evidencia for 'GRAU D — TEXTO VAGO/GENÉRICO' -> resultado_etapa_1 = 'VALIDAÇÃO PENDENTE'.

PASSO 4 — O Alerta de Fraude (divergencia_cadastral):
Responda: O cadastro (Modalidade) contradiz a realidade material (resultado_etapa_1)?
- True (VERDADEIRO): Quando o resultado da etapa 1 contradiz a modalidade cadastrada (ex: resultado INCLUÍDO — o objeto reflete o conceito material de inexigibilidade/credenciamento — mas a modalidade cadastrada é Pregão/Concorrência/Dispensa comum; ou resultado EXCLUÍDO — objeto singular/fechado Grau C — mas a modalidade cadastrada é Credenciamento/Inexigibilidade).
- False (FALSO - default): Quando o resultado da etapa 1 concorda com a modalidade cadastrada (ex: resultado EXCLUÍDO e a modalidade cadastrada é qualquer uma exceto inexigibilidade ou credenciamento; ou resultado INCLUÍDO e a modalidade é inexigibilidade ou credenciamento) OU quando o resultado da etapa 1 é VALIDAÇÃO PENDENTE (Grau D)."""


def _formatar_tempo(segundos: float) -> str:
    """Converte segundos em formato legível (ex: 1h 12m 05s ou 04m 32s)."""
    if segundos < 0 or np.isnan(segundos) or np.isinf(segundos):
        return "--:--"
    segundos_int = int(round(segundos))
    horas, resto = divmod(segundos_int, 3600)
    minutos, segs = divmod(resto, 60)
    if horas > 0:
        return f"{horas}h {minutos:02d}m {segs:02d}s"
    return f"{minutos:02d}m {segs:02d}s"


def _impor_consistencia_regras(
    grau: str,
    resultado: str,
    divergencia: bool
) -> tuple[str, str, bool]:
    """
    Trava determinística pós-LLM para garantir 100% de aderência matemática
    entre Grau de Evidência, Resultado Etapa 1 e Divergência Cadastral.
    """
    if grau in ('GRAU A — EVIDÊNCIA DIRETA', 'GRAU B — SINTOMÁTICO'):
        resultado_corrigido = 'INCLUÍDO'
        divergencia_corrigida = bool(divergencia)
    elif grau == 'GRAU C — SEM EVIDÊNCIA':
        resultado_corrigido = 'EXCLUÍDO'
        divergencia_corrigida = bool(divergencia)
    else:
        grau = 'GRAU D — TEXTO VAGO/GENÉRICO'
        resultado_corrigido = 'VALIDAÇÃO PENDENTE'
        divergencia_corrigida = False

    return grau, resultado_corrigido, divergencia_corrigida


# 3. CONTROLADOR DE TAXA, CHECKPOINT E MONITOR DE PROGRESSO
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
            df_ckpt = df_ckpt.drop_duplicates(subset=['Protocolo'], keep='last')
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


class MonitorProgresso:
    """
    Exibe em tempo real o andamento da Fase 3, com barra visual, contadores de classificação,
    progresso global da base e estimativa dinâmica de tempo restante (ETA).
    """
    def __init__(self, total_lote: int, ja_processados_antes: int, total_fila_geral: int, modo_gratuito: bool):
        self.total_lote = total_lote
        self.ja_processados_antes = ja_processados_antes
        self.total_fila_geral = total_fila_geral
        self.modo_gratuito = modo_gratuito
        self.inicio = time.time()
        self.lock = asyncio.Lock()

        self.concluidos_lote = 0
        self.sucessos = 0
        self.erros = 0
        self.contagem_resultados = {
            'INCLUÍDO': 0,
            'EXCLUÍDO': 0,
            'VALIDAÇÃO PENDENTE': 0
        }
        self.divergencias = 0

        self.pbar = tqdm(
            total=total_lote,
            desc="🔍 Auditando com Gemini",
            unit="proc",
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

    async def registrar_sucesso(self, resultado: dict) -> None:
        async with self.lock:
            self.concluidos_lote += 1
            self.sucessos += 1
            res_etapa = resultado.get('GI - Resultado Etapa 1', '')
            if res_etapa in self.contagem_resultados:
                self.contagem_resultados[res_etapa] += 1
            if resultado.get('GI - Divergência Cadastral Aparente'):
                self.divergencias += 1

            self._atualizar_barra()

    async def registrar_erro(self, protocolo: str) -> None:
        async with self.lock:
            self.concluidos_lote += 1
            self.erros += 1
            self._atualizar_barra()

    def _atualizar_barra(self) -> None:
        total_acumulado_global = self.ja_processados_antes + self.sucessos
        faltam_lote = self.total_lote - self.concluidos_lote
        self.pbar.set_postfix_str(
            f"Inc:{self.contagem_resultados['INCLUÍDO']} "
            f"Exc:{self.contagem_resultados['EXCLUÍDO']} "
            f"Pend:{self.contagem_resultados['VALIDAÇÃO PENDENTE']} "
            f"Div:{self.divergencias} "
            f"Erros:{self.erros} | "
            f"Global:{total_acumulado_global}/{self.total_fila_geral} "
            f"(Faltam {faltam_lote} no lote)"
        )
        self.pbar.update(1)

    def fechar(self) -> None:
        self.pbar.close()


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
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
    monitor: MonitorProgresso,
    rate_limiter: Optional[RateLimiterAsync] = None
) -> dict:
    """Executa a inferência na API, valida regras, persiste no checkpoint e atualiza o painel de progresso."""
    async with semaphore:
        try:
            resposta = await _chamar_api_gemini(client, modelo, prompt_usuario, rate_limiter)
            analise = resposta.parsed

            grau_final, resultado_final, divergencia_final = _impor_consistencia_regras(
                analise.grau_evidencia,
                analise.resultado_etapa_1,
                analise.divergencia_cadastral
            )

            resultado = {
                'Protocolo': str(protocolo),
                'GI - Fundamentação Contextual': analise.justificativa_sintetica,
                'GI - Grau de Evidência': grau_final,
                'GI - Resultado Etapa 1': resultado_final,
                'GI - Divergência Cadastral Aparente': divergencia_final
            }

            await checkpoint.salvar_registro(resultado)
            await monitor.registrar_sucesso(resultado)
            return resultado

        except Exception as e:
            await monitor.registrar_erro(str(protocolo))
            logging.error(f"Falha definitiva no Protocolo {protocolo} após tentativas de retry: {e}")
            raise e


# 4. MOTOR DE INFERÊNCIA ASSÍNCRONO COM RETOMADA AUTOMÁTICA E PAINEL DE PROGRESSO
async def orquestrar_inferencia(
    df_fila: pd.DataFrame,
    modo_gratuito: bool = False,
    modelo: Optional[str] = None,
    limite_rpd: int = 500,
    caminho_checkpoint: str = ARQUIVO_CHECKPOINT_PADRAO,
    ja_processados_memoria: int = 0
) -> pd.DataFrame:
    """
    Identifica protocolos já processados no checkpoint, exibe painel diagnóstico detalhado,
    controla limites de quota (RPM/RPD) e retorna o conjunto consolidado.
    """
    load_dotenv()

    checkpoint = GerenciadorCheckpoint(caminho_checkpoint)
    df_historico = checkpoint.carregar_processados()

    df_fila_copia = df_fila.copy()
    df_fila_copia['Protocolo'] = df_fila_copia['Protocolo'].astype(str)
    total_fila_atual = len(df_fila_copia)
    total_geral = total_fila_atual + ja_processados_memoria

    # Verifica quais protocolos da fila atual já constam no checkpoint em disco
    if not df_historico.empty:
        mask_ja_processados = df_fila_copia['Protocolo'].isin(df_historico['Protocolo'])
        qtde_ckpt = int(mask_ja_processados.sum())
        df_pendentes = df_fila_copia[~mask_ja_processados].copy()
    else:
        qtde_ckpt = 0
        df_pendentes = df_fila_copia

    ja_processados_total = ja_processados_memoria + qtde_ckpt
    perc_inicial = (ja_processados_total / total_geral * 100) if total_geral > 0 else 100.0

    # Identifica modo via parâmetro ou variável de ambiente
    if not modo_gratuito:
        env_modo = os.environ.get("GEMINI_MODO_GRATUITO", "false").strip().lower()
        modo_gratuito = env_modo in ("true", "1", "t", "sim", "yes")

    # Atualizado para gemini-3.8-flash (ativo) para modo pago e gemini-3.1-flash-lite para gratuito
    modelo_final = modelo or ("gemini-3.1-flash-lite" if modo_gratuito else "gemini-3.8-flash")

    print("\n" + "=" * 75)
    print("📊 PAINEL DE CONTROLE — FASE 3 (JULGAMENTO SEMÂNTICO VIA LLM)")
    print("=" * 75)
    print(f" • Total de protocolos na fila da IA : {total_geral}")
    print(f" • Já processados (Checkpoint/Mem.)  : {ja_processados_total} ({perc_inicial:.1f}%)")
    print(f" • Pendentes para concluir a base    : {len(df_pendentes)}")

    if df_pendentes.empty:
        print(" • Status                            : ✅ 100% CONCLUÍDO! Nenhuma chamada necessária.")
        print("=" * 75 + "\n")
        return df_historico[df_historico['Protocolo'].isin(df_fila_copia['Protocolo'])].copy()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env.")

    client = genai.Client(api_key=api_key)

    # Configuração de Modelo, Velocidade e Estimativa de Tempo
    if modo_gratuito:
        total_pendentes = len(df_pendentes)
        df_processar = df_pendentes.head(limite_rpd).copy() if total_pendentes > limite_rpd else df_pendentes.copy()
        restantes_apos_lote = total_pendentes - len(df_processar)

        semaphore = asyncio.Semaphore(1)
        rate_limiter = RateLimiterAsync(max_rpm=14)
        tempo_estimado_seg = len(df_processar) * (60.0 / 14.0)

        print(f" • Modo de Execução                  : GRATUITO ({modelo_final})")
        print(f" • Limites de Quota Aplicados        : 14 RPM (~4.3s/req) | Máx {limite_rpd} req/dia")
        print(f" • Lote desta sessão                 : {len(df_processar)} protocolos")
        if restantes_apos_lote > 0:
            print(f" • Restantes para próximas sessões   : {restantes_apos_lote} protocolos (proteção de 500 RPD)")
        print(f" • Tempo estimado para este lote     : ~{_formatar_tempo(tempo_estimado_seg)}")
    else:
        df_processar = df_pendentes.copy()
        semaphore = asyncio.Semaphore(10)
        rate_limiter = None
        tempo_estimado_seg = len(df_processar) * 0.4

        print(f" • Modo de Execução                  : PAGO / ALTA VELOCIDADE ({modelo_final})")
        print(f" • Concorrência Paralela             : 10 requisições simultâneas")
        print(f" • Lote desta sessão                 : {len(df_processar)} protocolos (100% dos pendentes)")
        print(f" • Tempo estimado para este lote     : ~{_formatar_tempo(tempo_estimado_seg)}")

    print(" • Tolerância a falhas               : Ativa (salvo em disco a cada protocolo)")
    print("=" * 75 + "\n")
    sys.stdout.flush()

    # Construção vetorizada dos prompts
    prompts_series = (
        "Protocolo: " + df_processar['Protocolo'].fillna('').astype(str) + "\n"
        + "Modalidade: " + df_processar['Modalidade'].fillna('').astype(str) + "\n"
        + "Objeto: " + df_processar['Objeto'].fillna('').astype(str) + "\n"
        + "Regime Jurídico: " + df_processar['GI - Regime Jurídico Informado'].fillna('').astype(str) + "\n"
        + "Valor Licitação: " + df_processar['Valor'].fillna(0.0).astype(str)
    )

    monitor = MonitorProgresso(
        total_lote=len(df_processar),
        ja_processados_antes=ja_processados_total,
        total_fila_geral=total_geral,
        modo_gratuito=modo_gratuito
    )

    tasks = [
        inferir_registro(client, modelo_final, proto, prompt, semaphore, checkpoint, monitor, rate_limiter)
        for proto, prompt in zip(df_processar['Protocolo'], prompts_series)
    ]

    inicio_execucao = time.time()
    interrompido = False

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except (asyncio.CancelledError, KeyboardInterrupt):
        interrompido = True
    finally:
        monitor.fechar()

    duracao_real = time.time() - inicio_execucao
    df_consolidado = checkpoint.carregar_processados()

    concluidos_agora = monitor.sucessos
    total_concluido_final = ja_processados_total + concluidos_agora
    faltam_geral_final = max(0, total_geral - total_concluido_final)
    perc_final = (total_concluido_final / total_geral * 100) if total_geral > 0 else 100.0

    print("\n" + "=" * 75)
    if interrompido:
        print("⚠️ EXECUÇÃO PAUSADA PELO USUÁRIO (PROGRESSO SALVO COM SEGURANÇA)")
    else:
        print("✅ RESUMO DE ENCERRAMENTO DA SESSÃO — FASE 3")
    print("=" * 75)
    print(f" • Tempo decorrido nesta sessão      : {_formatar_tempo(duracao_real)}")
    print(f" • Novos protocolos analisados agora : {concluidos_agora} de {len(df_processar)} (Falhas: {monitor.erros})")
    print(f"   ├─ INCLUÍDO                       : {monitor.contagem_resultados['INCLUÍDO']}")
    print(f"   ├─ EXCLUÍDO                       : {monitor.contagem_resultados['EXCLUÍDO']}")
    print(f"   ├─ VALIDAÇÃO PENDENTE             : {monitor.contagem_resultados['VALIDAÇÃO PENDENTE']}")
    print(f"   └─ Alertas de Divergência         : {monitor.divergencias}")
    print("-" * 75)
    print(f" • Progresso Global Acumulado        : {total_concluido_final}/{total_geral} ({perc_final:.1f}%)")
    if faltam_geral_final > 0:
        print(f" • Faltam para terminar toda a fila  : {faltam_geral_final} protocolos (basta rodar novamente)")
    else:
        print(" • Status da Fila                    : 🎉 TODOS OS PROTOCOLOS FORAM PROCESSADOS!")
    print("=" * 75 + "\n")
    sys.stdout.flush()

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
    Fase 3: Julgamento Semântico via LLM (Gemini) com tolerância a interrupções
    e monitoramento visual detalhado de progresso e tempo restante (ETA).
    """
    if df_fila_llm.empty:
        logging.info("A fila do LLM está vazia. Nenhuma inferência será realizada.")
        return df_analytical

    df_analytical_out = df_analytical.copy()
    df_analytical_out['Protocolo'] = df_analytical_out['Protocolo'].astype(str)

    # Garante que colunas textuais/booleanas de destino tenham dtype 'object' antes do update()
    # (previne LossySetitemError / TypeError caso uma coluna de nulos tenha sido inferida como float64 pelo Pandas)
    colunas_alvo_update = [
        'GI - Fundamentação Contextual',
        'GI - Grau de Evidência',
        'GI - Resultado Etapa 1',
        'GI - Divergência Cadastral Aparente',
        'GI - Corroboração Utilizada',
        'GI - Protocolo Relacionado'
    ]
    for col_alvo in colunas_alvo_update:
        if col_alvo in df_analytical_out.columns:
            df_analytical_out[col_alvo] = df_analytical_out[col_alvo].astype('object')

    df_fila_preparada = df_fila_llm.copy()
    df_fila_preparada['Protocolo'] = df_fila_preparada['Protocolo'].astype(str)

    # Reconhece registros que eventualmente já estejam preenchidos em memória no df_analytical
    status_pendentes = {'NÃO CLASSIFICADO', 'FILA PARA LLM'}
    mask_ja_avaliados_mem = (
        ~df_analytical_out['GI - Resultado Etapa 1'].isin(status_pendentes)
        & df_analytical_out['GI - Fundamentação Contextual'].notna()
    )
    protocolos_avaliados_mem = set(df_analytical_out.loc[mask_ja_avaliados_mem, 'Protocolo'])

    qtde_mem = 0
    if protocolos_avaliados_mem:
        mask_em_fila = df_fila_preparada['Protocolo'].isin(protocolos_avaliados_mem)
        qtde_mem = int(mask_em_fila.sum())
        if qtde_mem > 0:
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
                caminho_checkpoint=caminho_checkpoint,
                ja_processados_memoria=qtde_mem
            )
        )
    except RuntimeError:
        df_resultados = asyncio.run(
            orquestrar_inferencia(
                df_fila_preparada,
                modo_gratuito=modo_gratuito,
                modelo=modelo,
                limite_rpd=limite_rpd,
                caminho_checkpoint=caminho_checkpoint,
                ja_processados_memoria=qtde_mem
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

    # Normalização pós-merge vetorizada: garante integridade da regra de Divergência Cadastral Aparente
    mask_val_pendente = df_analytical_out['GI - Resultado Etapa 1'] == 'VALIDAÇÃO PENDENTE'
    df_analytical_out.loc[mask_val_pendente, 'GI - Divergência Cadastral Aparente'] = False
    df_analytical_out['GI - Divergência Cadastral Aparente'] = (
        df_analytical_out['GI - Divergência Cadastral Aparente'].fillna(False).astype(bool)
    )

    assert df_analytical_out['Protocolo'].duplicated().sum() == 0, "Erro: Protocolos duplicados após reintegração da Fase 3."

    logging.info("Fase 3 concluída com sucesso. Base analítica enriquecida com insights semânticos.")
    return df_analytical_out

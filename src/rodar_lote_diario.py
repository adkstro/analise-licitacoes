import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
import pandas as pd
from dotenv import load_dotenv

# Garante que o diretório src/ esteja no sys.path e define caminhos absolutos seguros
DIR_SRC = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_SRC)
if DIR_SRC not in sys.path:
    sys.path.insert(0, DIR_SRC)

from fase_1 import executar_fase_1, ler_base_dados
from fase_2 import executar_fase_2
from fase_3 import executar_fase_3
from fase_4 import executar_fase_4
from fase_5 import executar_fase_5

# Caminhos padrão de arquivos e caches
ARQUIVO_FONTE_PADRAO = os.path.join(DIR_RAIZ, "Base dedados de Licitação em 17-09.xlsx")
CACHE_ANALYTICAL_PKL = os.path.join(DIR_SRC, "cache_fase2_analytical.pkl")
CACHE_QUEUE_PKL = os.path.join(DIR_SRC, "cache_fase2_queue.pkl")
CAMINHO_CHECKPOINT = os.path.join(DIR_SRC, "checkpoint_fase_3.jsonl")
CAMINHO_ENV = os.path.join(DIR_SRC, ".env")
CAMINHO_LOG = os.path.join(DIR_SRC, "log_lote_diario.log")

# Data limite: Terça-feira, 03/11/2026
DATA_LIMITE_PADRAO = date(2026, 11, 3)

# Configuração de log (Console + Arquivo persistente)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CAMINHO_LOG, encoding="utf-8")
    ]
)


def carregar_ou_gerar_base_fase_2(
    caminho_fonte: str = ARQUIVO_FONTE_PADRAO,
    forcar_recalculo: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os DataFrames das Fases 1 e 2 a partir do cache local (.pkl) em < 1 segundo,
    ou executa as Fases 1 e 2 na primeira vez e salva o cache para os dias seguintes.
    """
    df_raw = ler_base_dados(caminho_fonte)

    if (
        not forcar_recalculo
        and os.path.exists(CACHE_ANALYTICAL_PKL)
        and os.path.exists(CACHE_QUEUE_PKL)
    ):
        logging.info("Carregando base consolidada (Fase 1 + Fase 2 + Camada 1 R0–R6) do cache local...")
        df_analytical = pd.read_pickle(CACHE_ANALYTICAL_PKL)
        df_queue = pd.read_pickle(CACHE_QUEUE_PKL)
        logging.info(
            f"Cache carregado instantaneamente: {len(df_analytical)} protocolos únicos | "
            f"{len(df_queue)} protocolos na fila total do LLM."
        )
        return df_analytical, df_queue, df_raw

    logging.info("Cache não encontrado ou recálculo solicitado. Executando Fase 1 e Fase 2 no dataset completo...")
    df_analytical, df_raw = executar_fase_1(df_raw)
    df_analytical, df_queue = executar_fase_2(df_analytical)

    df_analytical.to_pickle(CACHE_ANALYTICAL_PKL)
    df_queue.to_pickle(CACHE_QUEUE_PKL)
    logging.info(f"Cache salvo com sucesso em '{CACHE_ANALYTICAL_PKL}' para acelerar as próximas execuções diárias.")

    return df_analytical, df_queue, df_raw


def executar_lote_do_dia(
    limite_diario: int = 500,
    gerar_relatorio_parcial: bool = True,
    forcar_recalculo_fase2: bool = False
) -> pd.DataFrame:
    """
    Executa um lote de até `limite_diario` (500) protocolos pendentes do dataset completo
    no modo gratuito (14 RPM, modelo gemini-3.1-flash-lite), salvando cada protocolo
    no checkpoint_fase_3.jsonl e atualizando o Excel parcial com os registros já concluídos.
    """
    load_dotenv(CAMINHO_ENV)
    load_dotenv()

    logging.info("=" * 75)
    logging.info(f"🚀 INICIANDO ROTINA DIÁRIA — LOTE DE ATÉ {limite_diario} PROTOCOLOS (MODO GRATUITO)")
    logging.info("=" * 75)

    df_analytical, df_queue, df_raw = carregar_ou_gerar_base_fase_2(
        forcar_recalculo=forcar_recalculo_fase2
    )

    # Executa a Fase 3 no modo gratuito com teto de 500 requisições no dia
    df_analytical_atualizado = executar_fase_3(
        df_analytical=df_analytical,
        df_fila_llm=df_queue,
        modo_gratuito=True,
        limite_rpd=limite_diario,
        caminho_checkpoint=CAMINHO_CHECKPOINT
    )

    if gerar_relatorio_parcial:
        # Filtra todos os protocolos que já possuem classificação concluída (Fase 2 direta + Checkpoint Fase 3)
        mask_concluidos = ~df_analytical_atualizado['GI - Resultado Etapa 1'].isin(
            ['NÃO CLASSIFICADO', 'FILA PARA LLM']
        )
        df_concluidos = df_analytical_atualizado[mask_concluidos].copy()
        logging.info(
            f"Consolidando Fase 4 e Fase 5 sobre os {len(df_concluidos)} protocolos já concluídos até o momento..."
        )
        df_concluidos_fase4 = executar_fase_4(df_concluidos)
        arquivo_excel = executar_fase_5(df_concluidos_fase4, df_raw)
        logging.info(f"📊 Relatório acumulado atualizado com sucesso: {arquivo_excel}")

    return df_analytical_atualizado


def executar_ciclo_ate_data_limite(
    limite_diario: int = 500,
    horario_diario: str = "09:00",
    data_limite: date = DATA_LIMITE_PADRAO
) -> None:
    """
    Roda imediatamente o lote de hoje (500 linhas) e agenda os próximos disparos diários
    no horário especificado até a data limite (terça-feira, 03/11/2026).
    """
    hora_alvo, minuto_alvo = map(int, horario_diario.split(":"))

    while True:
        hoje = date.today()
        if hoje > data_limite:
            logging.info(f"Data limite ({data_limite.strftime('%d/%m/%Y')}) atingida. Encerrando rotina diária.")
            break

        logging.info(f"Executando lote referente ao dia {hoje.strftime('%d/%m/%Y')} (Meta até {data_limite.strftime('%d/%m/%Y')})...")
        try:
            executar_lote_do_dia(limite_diario=limite_diario)
        except Exception as e:
            logging.error(f"Erro durante a execução do lote de {hoje.strftime('%d/%m/%Y')}: {e}")

        amanha = hoje + timedelta(days=1)
        if amanha > data_limite:
            logging.info(
                f"✅ Último lote programado ({hoje.strftime('%d/%m/%Y')}) concluído! "
                f"Rotina finalizada dentro do prazo (terça-feira, {data_limite.strftime('%d/%m/%Y')})."
            )
            break

        proximo_disparo = datetime.combine(amanha, datetime.min.time()).replace(
            hour=hora_alvo, minute=minuto_alvo, second=0
        )
        segundos_espera = max(60.0, (proximo_disparo - datetime.now()).total_seconds())
        horas_esp = int(segundos_espera // 3600)
        mins_esp = int((segundos_espera % 3600) // 60)

        logging.info(
            f"⏳ Próximo lote de {limite_diario} linhas agendado para {proximo_disparo.strftime('%d/%m/%Y às %H:%M')} "
            f"(em {horas_esp}h {mins_esp:02d}m)."
        )
        time.sleep(segundos_espera)


def registrar_tarefa_windows(horario_diario: str = "09:00", limite_diario: int = 500) -> bool:
    """
    Cria uma tarefa no Agendador de Tarefas do Windows (schtasks) para disparar o script
    todos os dias no horário configurado até terça-feira (03/11/2026).
    """
    python_exe = os.path.join(DIR_RAIZ, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    script_path = os.path.abspath(__file__)
    nome_tarefa = "Auditoria_Credenciamento_500_Diario"
    comando_acao = f'"{python_exe}" "{script_path}" --unico --limite {limite_diario}'

    cmd = [
        "schtasks", "/Create",
        "/SC", "DAILY",
        "/TN", nome_tarefa,
        "/TR", comando_acao,
        "/ST", horario_diario,
        "/ED", "03/11/2026",
        "/F"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logging.info(
                f"✅ Tarefa '{nome_tarefa}' registrada no Agendador do Windows (todos os dias às {horario_diario} até 03/11/2026)."
            )
            return True
        else:
            logging.warning(f"Não foi possível registrar via schtasks ({res.stderr.strip()}). Usando agendador Python.")
            return False
    except Exception as e:
        logging.warning(f"Exceção ao chamar schtasks: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rotina para processar 500 linhas/dia do dataset completo no modo gratuito até terça-feira."
    )
    parser.add_argument(
        "--unico",
        action="store_true",
        help="Executa apenas o lote de hoje (500 linhas) e encerra."
    )
    parser.add_argument(
        "--loop-ate-terca",
        action="store_true",
        help="Executa o lote de hoje imediatamente e mantém agendado diariamente até terça-feira (03/11/2026)."
    )
    parser.add_argument(
        "--agendar-windows",
        action="store_true",
        help="Registra a tarefa diária no Agendador de Tarefas do Windows até terça-feira e já roda o lote de hoje."
    )
    parser.add_argument(
        "--horario",
        type=str,
        default="09:00",
        help="Horário diário para os próximos lotes no formato HH:MM (padrão: 09:00)."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=500,
        help="Quantidade máxima de protocolos por dia no modo gratuito (padrão: 500)."
    )
    parser.add_argument(
        "--recalcular-fase2",
        action="store_true",
        help="Força a reexecução das Fases 1 e 2 ignorando o cache local."
    )

    args = parser.parse_args()

    if args.agendar_windows:
        registrar_tarefa_windows(horario_diario=args.horario, limite_diario=args.limite)
        executar_lote_do_dia(limite_diario=args.limite, forcar_recalculo_fase2=args.recalcular_fase2)
    elif args.loop_ate_terca:
        executar_ciclo_ate_data_limite(limite_diario=args.limite, horario_diario=args.horario)
    else:
        executar_lote_do_dia(limite_diario=args.limite, forcar_recalculo_fase2=args.recalcular_fase2)

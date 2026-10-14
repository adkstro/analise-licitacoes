import asyncio
import logging
import inspect

# Importações dos módulos do pipeline
from fase_1 import executar_fase_1
from fase_2 import executar_fase_2
from fase_3 import executar_fase_3
from fase_4 import executar_fase_4
from fase_5 import executar_fase_5

# Configuração global de log do orquestrador
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

async def executar_pipeline(caminho_arquivo: str):
    """
    Orquestrador principal do pipeline de auditoria de credenciamentos.
    Gerencia o fluxo de dados (DataFrames) entre as 5 fases do projeto.
    """
    try:
        logging.info("==================================================")
        logging.info("INICIANDO PIPELINE DE AUDITORIA DE CREDENCIAMENTO")
        logging.info("==================================================")

        # FASE 1: Ingestão e Higienização
        logging.info("Iniciando Fase 1: Ingestão e Higienização...")
        df_analytical, df_raw = executar_fase_1(caminho_arquivo)

        # FASE 2: Triagem Determinística e Semântica Vetorial
        logging.info("Iniciando Fase 2: Triagem Determinística e Peneira Vetorial...")
        df_analytical, df_queue = executar_fase_2(df_analytical)

        # FASE 3: Inferência Semântica via LLM (Gemini)
        logging.info("Iniciando Fase 3: Julgamento Semântico via LLM (Assíncrono)...")
        # Validação arquitetural: garante execução correta dependendo de como a função foi importada/definida
        if inspect.iscoroutinefunction(executar_fase_3):
            df_analytical = await executar_fase_3(df_analytical, df_queue)
        else:
            df_analytical = executar_fase_3(df_analytical, df_queue)

        # FASE 4: Corroboração Interna e Testes de Integridade
        logging.info("Iniciando Fase 4: Corroboração Interna e Asserções de Qualidade...")
        df_analytical = executar_fase_4(df_analytical)

        # FASE 5: Geração do Entregável Executivo
        logging.info("Iniciando Fase 5: Geração do Relatório Executivo Excel...")
        executar_fase_5(df_analytical, df_raw)

        logging.info("==================================================")
        logging.info("PIPELINE CONCLUÍDO COM SUCESSO!")
        logging.info("==================================================")

    except AssertionError as ae:
        logging.error(f"FALHA DE INTEGRIDADE (TESTE REPROVADO): {ae}")
        logging.error("O processamento foi interrompido para evitar corrupção de dados.")
    except Exception as e:
        logging.error(f"FALHA CRÍTICA NA EXECUÇÃO DO PIPELINE: {e}")
        logging.error("O processamento foi interrompido. Verifique os logs anteriores.")

if __name__ == '__main__':
    # Defina o caminho do arquivo de origem aqui
    ARQUIVO_FONTE = "../Base dedados de Licitação em 17-09.xlsx"
    
    # Inicia o event loop do orquestrador assíncrono
    asyncio.run(executar_pipeline(ARQUIVO_FONTE))
import time
from workers.base_worker import BaseWorker
from workers.analyser.strategy_coordinator import AnalyserStrategyCoordinator
from core.services.data_pipeline_service import DataPipelineService

class Analyser(BaseWorker):
    """
    Main orchestrator child process: Analyser.
    Coordinates UDS client connection to Executer, state syncing,
    and consumes tick data from the DataPipeline UDS service client wrapper.

    On each closed 1-minute candle:
      - Rolls data into a 840-candle queue (14 hours).
      - Every 10 candles: calls the Blackbox (Big Brother) for full analysis.
    """
    HEARTBEAT_PERIOD = 2
    MAIN_LOOP_SLEEP = 0.01  # Faster sleep for tick data

    def work(self):
        # 1. Initialize modular components. AnalyserStrategyCoordinator owns
        # its own ExecutorService (persistent connection, connects/
        # reconnects lazily on send - see core/services/executer_service.py),
        # so there's no separate connection-management class needed here.
        self.dp_service = DataPipelineService()
        self.strategy_coord = AnalyserStrategyCoordinator(worker_instance=self)

        last_heartbeat = 0

        try:
            while not self.shutdown:
                self.check_broadcast()
                now = time.time()

                # Heartbeat check
                if now - last_heartbeat >= self.HEARTBEAT_PERIOD:
                    self.send_heartbeat()
                    last_heartbeat = now

                # 2. Check and maintain connection to DataPipeline
                self.dp_service.connect()

                # 3. Check for incoming streaming packets from DataPipeline service
                packets = self.dp_service.read_dollar_bars()
                for packet in packets:
                    self.strategy_coord.process_dollar_bar(packet)

                time.sleep(self.MAIN_LOOP_SLEEP)
        finally:
            self.strategy_coord.executor_service.close()
            self.dp_service.close()
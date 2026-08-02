import json
import os
import asyncio
import dataclasses
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
from workers.base_worker import BaseWorker
from workers.data_pipeline.dollar_bar_builder import DollarBarBuilder

class DataPipeline(BaseWorker):
    """
    Child process: DataPipeline (Asynchronous).
    Maintains UDS socket `/tmp/data_pipeline_service.sock` to broadcast bars.
    Subscribes to Binance Futures `@aggTrade` WebSocket, feeds them through
    ImbalanceBarBuilder -> MacroBarAggregator, and broadcasts completed bars.

    Trading is fixed to DEFAULT_COINS (BTCUSDT/ETHUSDT) - DataPipeline
    doesn't track shared-memory position state at all; it just streams
    both symbols unconditionally for as long as it runs. See
    core/shm.py's SharedMemoryService docstring for why the old dynamic
    subscribe/unsubscribe-on-trade-close behavior was removed.
    """
    HEARTBEAT_PERIOD = 2
    UDS_SOCKET_PATH = "/tmp/data_pipeline_service.sock"
    WS_CONNECT_RETRY_INTERVAL = 2
    WS_SIGNALS_POLL_INTERVAL = 0.1
    DEFAULT_COINS = ["BTCUSDT", "ETHUSDT"]

    def work(self):
        """Launches the asynchronous event loop to run DataPipeline tasks."""
        asyncio.run(self.async_work())

    async def async_work(self):
        self.stopped = False
        self.active_clients = []
        self.ubwa = None
        self.server = None
        self.stream_id = None

        # Registered symbols
        self.registered_symbols = set()
        self.orderbooks = {}
        self.funding_rates = {}
        self.builders = {}

        # 1. Register initial symbols
        initial_coins = self.DEFAULT_COINS
        self.send_log(f"Initializing DataPipeline for target symbols: {initial_coins}")
        for sym in initial_coins:
            self._register_symbol(sym)

        # 2. Start WebSocket Api Manager
        self.ubwa = BinanceWebSocketApiManager(
            exchange="binance.com-futures",
            output_default="dict",
            enable_stream_signal_buffer=True
        )

        markets = [s.lower() for s in initial_coins]
        self.stream_id = self.ubwa.create_stream(
            channels=["aggTrade", "depth20@100ms", "markPrice@1s"],
            markets=markets
        )
        self.send_log(f"WebSocket stream created (id={self.stream_id}). Subscribed to {initial_coins} @aggTrade, @depth20@100ms, and @markPrice@1s.")

        # 3. Run concurrent tasks
        tasks = [
            asyncio.create_task(self.heartbeat_loop()),
            asyncio.create_task(self.broadcast_loop()),
            asyncio.create_task(self.uds_server_loop()),
            asyncio.create_task(self.ws_loop()),
            asyncio.create_task(self.ws_signals_loop()),
            asyncio.create_task(self.threshold_update_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.send_log("DataPipeline async tasks cancelled.")
        finally:
            self.stopped = True
            if self.server:
                self.server.close()
            try:
                if os.path.exists(self.UDS_SOCKET_PATH):
                    os.unlink(self.UDS_SOCKET_PATH)
            except OSError:
                pass
            for writer in list(self.active_clients):
                try:
                    writer.close()
                except Exception:
                    pass
            if self.ubwa:
                self.ubwa.stop_manager_with_all_streams()

    def _register_symbol(self, symbol: str):
        symbol = symbol.upper()
        self.registered_symbols.add(symbol)
        if symbol not in self.builders:
            self.builders[symbol] = DollarBarBuilder(symbol)

    def _deregister_symbol(self, symbol: str):
        symbol = symbol.upper()
        self.registered_symbols.discard(symbol)
        self.builders.pop(symbol, None)

    async def heartbeat_loop(self):
        while not self.stopped:
            self.send_heartbeat()
            await asyncio.sleep(self.HEARTBEAT_PERIOD)

    async def broadcast_loop(self):
        while not self.stopped:
            self.check_broadcast()
            await asyncio.sleep(self.WS_SIGNALS_POLL_INTERVAL)

    async def ws_signals_loop(self):
        self.send_log("DataPipeline WebSocket signals listener started.")
        while not self.stopped:
            if self.ubwa:
                try:
                    signal = self.ubwa.pop_stream_signal_from_stream_signal_buffer()
                    if signal:
                        self.send_log(f"UBWS Signal received: {signal}")
                        sig_type = signal.get("type")
                        if sig_type in ("CONNECT", "DISCONNECT"):
                            self.send_log(f"{sig_type} signal received. Clearing local OrderBook cache.")
                            self.orderbooks.clear()
                            if sig_type == "CONNECT":
                                self.send_log("WebSocket CONNECT signal processed.")
                except Exception as e:
                    self.send_log(f"Error popping stream signal: {e}", errnum=1)
            await asyncio.sleep(self.WS_SIGNALS_POLL_INTERVAL)

    async def threshold_update_loop(self):
        self.send_log("Starting weekly volume threshold updater loop.")
        while not self.stopped:
            try:
                import urllib.request
                for sym in list(self.builders.keys()):
                    try:
                        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&limit=7"
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        
                        def fetch_data():
                            with urllib.request.urlopen(req, timeout=10) as response:
                                return json.loads(response.read().decode('utf-8'))
                        
                        data = await asyncio.to_thread(fetch_data)
                        if data:
                            # kline index 7 is quote_asset_volume
                            total_q_vol = sum(float(k[7]) for k in data)
                            avg_daily_vol = total_q_vol / max(len(data), 1)
                            self.builders[sym].update_threshold(avg_daily_vol)
                            
                    except Exception as e:
                        self.send_log(f"Error updating 7d threshold for {sym}: {e}", errnum=1)
                        
                self.send_log("Updated dollar bar thresholds from 7-day rolling average volume.")
            except Exception as e:
                self.send_log(f"Error in threshold loop: {e}", errnum=1)
                
            await asyncio.sleep(21600)  # 6 hours

    async def uds_server_loop(self):
        socket_path = self.UDS_SOCKET_PATH
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass

        self.server = await asyncio.start_unix_server(self.handle_client, path=socket_path)
        self.send_log(f"DataPipeline UDS Service Server listening on {socket_path}")
        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader, writer):
        """
        Analyser connects here purely to receive the dollar_bar broadcast
        stream (see _broadcast_uds) - it never sends anything back to
        DataPipeline. This loop's only remaining job is to notice when the
        client disconnects (readline() returns empty) so the writer gets
        cleaned up out of active_clients. There is no defined protocol for
        pipeline-bound messages anymore, so anything a client does send is
        logged as unexpected rather than acted on.
        """
        self.active_clients.append(writer)
        try:
            while not self.stopped:
                line = await reader.readline()
                if not line:
                    break
                self.send_log(f"DataPipeline received unexpected client data: {line[:200]!r}", errnum=1)
        except asyncio.CancelledError:
            pass
        finally:
            if writer in self.active_clients:
                self.active_clients.remove(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def ws_loop(self):
        self.send_log("DataPipeline WebSocket consumer loop started.")
        while not self.stopped:
            try:
                if self.ubwa:
                    data = self.ubwa.pop_stream_data_from_stream_buffer()
                    if not data:
                        await asyncio.sleep(self.WS_SIGNALS_POLL_INTERVAL)
                        continue

                    # Data can be either aggTrade or depthUpdate (which doesn't always have 'e')
                    payload = data.get("data", data)
                    
                    if not isinstance(payload, dict):
                        continue

                    # Handle depth updates (Binance partial book depth streams don't always have 'e'='depthUpdate', but they always have 'bids' and 'asks')
                    if "bids" in payload and "asks" in payload:
                        symbol = payload.get("s", payload.get("symbol", "")).upper()
                        if symbol:
                            bids = payload["bids"]
                            asks = payload["asks"]
                            # Sum quantities (which are at index 1)
                            total_bid_qty = sum(float(b[1]) for b in bids)
                            total_ask_qty = sum(float(a[1]) for a in asks)
                            total_qty = total_bid_qty + total_ask_qty
                            q_imbalance = (total_bid_qty - total_ask_qty) / total_qty if total_qty > 0 else 0.0
                            self.orderbooks[symbol] = q_imbalance
                        continue

                    # Handle markPriceUpdate
                    if payload.get("e") == "markPriceUpdate":
                        symbol = payload.get("s", "").upper()
                        if symbol and "r" in payload:
                            try:
                                self.funding_rates[symbol] = float(payload["r"])
                            except (ValueError, TypeError):
                                pass
                        continue

                    # Handle aggTrade
                    if payload.get("e") == "aggTrade":
                        symbol = payload.get("s", "").upper()
                        if symbol not in self.registered_symbols:
                            continue

                        builder = self.builders.get(symbol)
                        if builder:
                            bar_packet = builder.feed(payload)
                            if bar_packet:
                                bar_packet["quote_imbalance"] = self.orderbooks.get(symbol, 0.0)
                                bar_packet["fundingRate"] = self.funding_rates.get(symbol, 0.0)
                                payload_out = (json.dumps({"type": "dollar_bar", "data": bar_packet}) + "\n").encode('utf-8')
                                self._broadcast_uds(payload_out)

                else:
                    await asyncio.sleep(self.WS_SIGNALS_POLL_INTERVAL)
            except Exception as e:
                self.send_log(f"Error in ws_loop: {e}", errnum=1)
                await asyncio.sleep(self.WS_CONNECT_RETRY_INTERVAL)

    def _broadcast_uds(self, payload: bytes):
        if not self.active_clients:
            return
        for writer in list(self.active_clients):
            try:
                writer.write(payload)
                # Note: drain() can be called in non-blocking send or ignored if buffer size is small
            except Exception:
                if writer in self.active_clients:
                    self.active_clients.remove(writer)


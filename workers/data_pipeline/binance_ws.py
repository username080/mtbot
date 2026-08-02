import asyncio
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

class BinanceWebSocketManager:
    """
    Manager for multiple Binance WebSocket streams dynamically using
    unicorn-binance-websocket-api.
    Groups symbols sharing the same resolution onto the same websocket stream.
    """
    EXCHANGE_DEFAULT = "binance.com-futures"
    OUTPUT_DEFAULT = "dict"

    def __init__(self):
        self.ubwa = BinanceWebSocketApiManager(
            exchange=self.EXCHANGE_DEFAULT,
            output_default=self.OUTPUT_DEFAULT,
            enable_stream_signal_buffer=True
        )
        # Maps resolution -> {"stream_id": stream_id, "symbols": set([symbols])}
        self.streams = {}

    def subscribe(self, symbols: list, resolution: str = "1m"):
        """
        Dynamically subscribes a list of symbols to a stream corresponding to the resolution.
        """
        if not symbols:
            return
            
        channel = f"kline_{resolution}"
        markets = [s.lower() for s in symbols]

        if resolution not in self.streams:
            # Create a new websocket stream for this resolution
            stream_id = self.ubwa.create_stream(channels=[channel], markets=markets)
            self.streams[resolution] = {
                "stream_id": stream_id,
                "symbols": set(symbols)
            }
        else:
            stream_info = self.streams[resolution]
            stream_id = stream_info["stream_id"]
            
            # Find symbols that are not already subscribed to this stream
            to_sub = [s for s in symbols if s not in stream_info["symbols"]]
            if to_sub:
                markets_to_sub = [s.lower() for s in to_sub]
                self.ubwa.subscribe_to_stream(stream_id, channels=[channel], markets=markets_to_sub)
                stream_info["symbols"].update(to_sub)

    def unsubscribe(self, symbols: list, resolution: str = "1m"):
        """
        Dynamically unsubscribes a list of symbols from the stream for the resolution.
        """
        if resolution in self.streams:
            stream_info = self.streams[resolution]
            stream_id = stream_info["stream_id"]
            channel = f"kline_{resolution}"
            
            # Find symbols that are currently subscribed to this stream
            to_unsub = [s for s in symbols if s in stream_info["symbols"]]
            if to_unsub:
                markets_to_unsub = [s.lower() for s in to_unsub]
                self.ubwa.unsubscribe_from_stream(stream_id, channels=[channel], markets=markets_to_unsub)
                for s in to_unsub:
                    stream_info["symbols"].remove(s)
            
            # If no symbols remain for this stream, shut it down
            if not stream_info["symbols"]:
                self.ubwa.stop_stream(stream_id)
                self.ubwa.delete_stream_id_with_buffer(stream_id)
                del self.streams[resolution]

    def pop_data(self):
        """Pops data from the combined stream buffer."""
        if self.ubwa:
            return self.ubwa.pop_stream_data_from_stream_buffer()
        return None

    def pop_signal(self):
        """Pops connection signal events from the stream signal buffer."""
        if self.ubwa:
            return self.ubwa.pop_stream_signal_from_stream_signal_buffer()
        return None

    def close(self):
        """Stops the manager and closes all WebSocket connections."""
        if self.ubwa:
            self.ubwa.stop_manager_with_all_streams()
            self.ubwa = None
            self.streams.clear()

class BinanceWebSocket:
    """Backwards-compatible wrapper for tests and old scripts."""
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.manager = BinanceWebSocketManager()
        self.ubwa = None

    async def connect(self):
        self.manager.subscribe(self.symbols, "1m")
        self.ubwa = self.manager.ubwa
        return self

    async def recv(self) -> dict:
        while True:
            data = self.manager.pop_data()
            if data:
                return data
            await asyncio.sleep(0.01)

    async def close(self):
        self.manager.close()

    def subscribe(self, symbols: list):
        self.manager.subscribe(symbols, "1m")
        for s in symbols:
            if s not in self.symbols:
                self.symbols.append(s)

    def unsubscribe(self, symbols: list):
        self.manager.unsubscribe(symbols, "1m")
        for s in symbols:
            if s in self.symbols:
                self.symbols.remove(s)

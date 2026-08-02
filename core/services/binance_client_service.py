import os
import logging
from typing import Dict, List, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class BinanceClientService:
    """
    Service class encapsulating Binance Futures REST API connection and operations.
    Supports a mock Dry-Run mode if API keys are not supplied.
    """
    def __init__(self):
        self.api_key = os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self._client: Optional[Client] = None
        self.live_mode = False

        if self.api_key and self.api_secret:
            try:
                self._client = Client(self.api_key, self.api_secret)
                # Verify connection / futures access
                self._client.futures_ping()
                self.live_mode = True
                logger.info("[BinanceClientService] Successfully connected to Binance Futures REST API (Live Mode).")
            except Exception as exc:
                logger.error(f"[BinanceClientService] Failed to authenticate with Binance Futures: {exc}. Falling back to Dry-Run Mode.")
                self._client = None
        else:
            logger.info("[BinanceClientService] No API keys found in environment. Initialised in Dry-Run (Mock) Mode.")

    def is_live(self) -> bool:
        """Returns True if connected to actual Binance Futures account."""
        return self.live_mode

    def get_futures_balance(self) -> float:
        """
        Fetch available USDT balance in the Futures account.
        In Dry-Run mode, returns 100,000.0.
        """
        if not self.live_mode or not self._client:
            return 100000.0

        try:
            balances = self._client.futures_account_balance()
            for bal in balances:
                if bal.get("asset") == "USDT":
                    # return available balance or wallet balance
                    return float(bal.get("withdrawAvailable", bal.get("balance", 0.0)))
            return 0.0
        except BinanceAPIException as exc:
            logger.error(f"[BinanceClientService] API Error fetching balance: {exc}")
            return 0.0
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error fetching balance: {exc}")
            return 0.0

    def get_open_positions(self) -> Dict[str, float]:
        """
        Fetch active open positions.
        Returns a dictionary of symbol -> quantity (signed: positive for long, negative for short).
        """
        if not self.live_mode or not self._client:
            return {}

        positions = {}
        try:
            raw_pos = self._client.futures_position_information()
            for pos in raw_pos:
                amt = float(pos.get("positionAmt", 0.0))
                if abs(amt) > 1e-8:
                    positions[pos["symbol"]] = amt
        except BinanceAPIException as exc:
            logger.error(f"[BinanceClientService] API Error fetching positions: {exc}")
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error fetching positions: {exc}")
        return positions

    def get_order_book(self, symbol: str, limit: int = 5) -> dict:
        """Fetch current order book for a symbol to get Bid1/Ask1."""
        if not self.live_mode or not self._client:
            # Mock book for dry run
            return {"bids": [["99900.0", "1.0"]], "asks": [["99901.0", "1.0"]]}
        try:
            return self._client.futures_order_book(symbol=symbol, limit=limit)
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error fetching order book: {exc}")
            return {"bids": [], "asks": []}

    def get_order_status(self, symbol: str, order_id: int) -> dict:
        """Fetch the status of a specific order."""
        if not self.live_mode or not self._client:
            # In mock mode, assume filled immediately
            return {"status": "FILLED", "executedQty": "1.0"}
        try:
            return self._client.futures_get_order(symbol=symbol, orderId=order_id)
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error fetching order status: {exc}")
            return {"status": "NEW"}

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel a specific order by ID."""
        if not self.live_mode or not self._client:
            return {"status": "CANCELED"}
        try:
            return self._client.futures_cancel_order(symbol=symbol, orderId=order_id)
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error canceling order: {exc}")
            return {}

    def place_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: float,
        order_type: str = "MARKET",
        price: float = 0.0,
        time_in_force: str = "GTC",
        sl_price: float = 0.0,
        tp_price: float = 0.0,
    ) -> dict:
        """
        Sends orders to Binance Futures.
        Updates leverage first, executes the specified order (MARKET or LIMIT),
        and attaches Stop-Loss and Take-Profit orders if specified.
        """
        if not self.live_mode or not self._client:
            logger.info(
                f"[BinanceClientService][MOCK ORDER] {side} {quantity} {symbol} "
                f"type={order_type} price={price} leverage={leverage}x | SL={sl_price} TP={tp_price}"
            )
            return {"status": "FILLED", "orderId": 999999, "mock": True}

        try:
            # 1. Update leverage
            self._client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

            # 2. Main execution order
            kwargs = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity
            }
            if order_type.upper() == "LIMIT":
                kwargs["price"] = price
                kwargs["timeInForce"] = time_in_force

            main_order = self._client.futures_create_order(**kwargs)
            logger.info(f"[BinanceClientService] Main order placed: {main_order}")

            # Exit side is opposite of entry
            exit_side = "SELL" if side.upper() == "BUY" else "BUY"

            # 3. Stop Loss Order
            if sl_price > 0:
                sl_order = self._client.futures_create_order(
                    symbol=symbol,
                    side=exit_side,
                    type="STOP_MARKET",
                    stopPrice=sl_price,
                    closePosition=True
                )
                logger.info(f"[BinanceClientService] Stop-Loss order placed: {sl_order}")

            # 4. Take Profit Order
            if tp_price > 0:
                tp_order = self._client.futures_create_order(
                    symbol=symbol,
                    side=exit_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=tp_price,
                    closePosition=True
                )
                logger.info(f"[BinanceClientService] Take-Profit order placed: {tp_order}")

            return main_order

        except BinanceAPIException as exc:
            logger.error(f"[BinanceClientService] Binance API Exception placing order: {exc}")
            raise exc
        except Exception as exc:
            logger.error(f"[BinanceClientService] Exception placing order: {exc}")
            raise exc

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol (e.g. SL/TP on exit)."""
        if not self.live_mode or not self._client:
            return
        try:
            self._client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception as exc:
            logger.error(f"[BinanceClientService] Error cancelling orders for {symbol}: {exc}")

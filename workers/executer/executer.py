import time
import socket
import json
import os
from workers.base_worker import BaseWorker
from core.exceptions.custom_exceptions import IPCError, SHMError
from core.services.binance_client_service import BinanceClientService
from core.shm import shm_service
from core.packets.m_packet import MPacket

class Executer(BaseWorker):
    """
    Child process: Executer.
    Listens on UDS socket `/tmp/executer.sock` for orders from Analyser.
    Performs margin checks, executes orders via Binance Futures client (live or mock),
    and updates the split shared memory state.
    """
    UDS_SOCKET_PATH = "/tmp/executer.sock"
    HEARTBEAT_PERIOD = 2
    MAIN_LOOP_SLEEP = 0.1
    INITIAL_CASH_DEFAULT = 100000.0
    BUFFER_SIZE = 4096
    SOCKET_BACKLOG = 5

    def _handle_sell_all_broadcast(self, packet: MPacket) -> None:
        """
        Registered for the "SELL_ALL" message type (see BaseWorker's
        generic dispatch). Broadcast SELL_ALL (from Manager.shutdown_sequence
        going out over broadcast_channel) and the same action arriving
        directly over the UDS order socket both end up liquidating - this
        is the broadcast side; the UDS side is handled separately in
        work()'s connection loop until that channel migrates too.
        """
        self.liquidate_all_positions()
        self.stopped = True

    def _handle_order(self, packet: MPacket) -> None:
        """
        Registered for "BUY", "SELL", and "SMART_EXIT" (see work()). All
        three share the same margin-check / execute / SHM-update structure
        below - this is the same single if/elif block that used to be
        inlined in the connection-read loop, moved behind dispatch()
        unchanged (side_action still distinguishes SMART_EXIT's actual
        closing direction from the entry actions).
        """
        action = packet.type
        command = packet.payload

        if self.shutdown:
            self.send_log(f"Executer is shutting down. Ignoring {action} order.")
            return

        symbol = command.get("symbol")
        qty = float(command.get("quantity", 0.0))
        price = float(command.get("price", 0.0))
        leverage = float(command.get("leverage", 1.0))
        sl_price = float(command.get("sl_price", 0.0))
        tp_price = float(command.get("tp_price", 0.0))
        atr = float(command.get("atr", 0.0))

        side_action = command.get("side") if action == "SMART_EXIT" else action

        if qty <= 0:
            self.send_log(f"Invalid order parameters: qty={qty}", errnum=1)
            return

        # ── Split-Mutex SHM reads ─────────────────────
        portfolio = shm_service.read_portfolio()
        positions_info = shm_service.read_positions_info()

        cash = portfolio.get("cash", self.INITIAL_CASH_DEFAULT)
        positions = portfolio.setdefault("positions", {})

        # Margin calculation
        required_margin = (qty * price) / leverage

        # If this order is an exit (reduces/flips position), we release margin
        old_qty = positions.get(symbol, 0.0)
        is_exit = False
        if side_action == "BUY" and old_qty < 0:
            is_exit = True
        elif side_action == "SELL" and old_qty > 0:
            is_exit = True

        # Check available funds
        if not is_exit and cash < required_margin:
            self.send_log(
                f"EXECUTION REJECTED: Insufficient funds for {side_action} {qty} {symbol}. "
                f"Required Margin: {required_margin:.2f}, Available Cash: {cash:.2f}",
                errnum=1
            )
            return

        # Execute trade via BinanceClientService
        try:
            if action == "SMART_EXIT":
                self.binance_svc.cancel_all_orders(symbol)
                self._execute_smart_exit(symbol, side_action, qty)
            elif is_exit:
                # Cancel open SL/TP orders on exit
                self.binance_svc.cancel_all_orders(symbol)
                # Use Market Order for Exits to guarantee closure
                self.binance_svc.place_futures_order(
                    symbol=symbol,
                    side=side_action,
                    quantity=qty,
                    leverage=leverage,
                    sl_price=sl_price,
                    tp_price=tp_price
                )
            else:
                # Use Soft Chase (Maker Pegging) for Entries
                success = self._execute_soft_chase(
                    symbol=symbol,
                    side=side_action,
                    quantity=qty,
                    leverage=leverage,
                    signal_price=price,
                    sl_price=sl_price,
                    tp_price=tp_price
                )
                if not success:
                    self.send_log(f"EXECUTION ABORTED: Soft Chase failed or walked away for {symbol}.")
                    return
        except Exception as exc:
            self.send_log(f"Binance order execution failed: {exc}", errnum=1)
            return

        # Update SHM State
        if side_action == "BUY":
            new_qty = old_qty + qty
            cost = qty * price
            if is_exit:
                # Closed short position: release margin/profit
                portfolio["cash"] += cost  # simplification for mock cash tracking
                if abs(new_qty) < 1e-8:
                    positions.pop(symbol, None)
                    positions_info.pop(symbol, None)
            else:
                # Entered long position
                portfolio["cash"] -= cost
                positions[symbol] = new_qty

                positions_info[symbol] = {
                    "entry_price": price,
                    "entry_atr": atr,
                    "entry_time": time.time(),
                    "highest_price": price,
                    "lowest_price": 0.0,
                    "candles_in_trade": 0
                }
        else:  # SELL
            new_qty = old_qty - qty
            revenue = qty * price
            if is_exit:
                # Closed long position: release cash
                portfolio["cash"] += revenue
                if abs(new_qty) < 1e-8:
                    positions.pop(symbol, None)
                    positions_info.pop(symbol, None)
            else:
                # Entered short position
                portfolio["cash"] -= revenue  # margin cost
                positions[symbol] = new_qty

                positions_info[symbol] = {
                    "entry_price": price,
                    "entry_atr": atr,
                    "entry_time": time.time(),
                    "highest_price": 0.0,
                    "lowest_price": price,
                    "candles_in_trade": 0
                }

        # ── Split-Mutex SHM writes ────────────────────
        shm_service.write_portfolio(portfolio)
        shm_service.write_positions_info(positions_info)

        self.send_log(
            f"EXECUTION SUCCESS: {action} {qty} {symbol} at {price}. "
            f"Portfolio Cash: {portfolio['cash']:.2f}, Positions: {positions}"
        )

    def liquidate_all_positions(self):
        self.send_log("CRITICAL: Received SELL_ALL signal. Liquidating all positions!")
        try:
            portfolio = shm_service.read_portfolio()

            # Initialize client
            binance_svc = BinanceClientService()

            for symbol, qty in list(portfolio.get("positions", {}).items()):
                if abs(qty) > 1e-8:
                    exit_side = "SELL" if qty > 0 else "BUY"
                    abs_qty = abs(qty)

                    if binance_svc.is_live():
                        try:
                            # Cancel outstanding SL/TP orders
                            binance_svc.cancel_all_orders(symbol)
                            # Place market close
                            binance_svc.place_futures_order(
                                symbol=symbol,
                                side=exit_side,
                                quantity=abs_qty,
                                leverage=1.0 # arbitrary
                            )
                        except Exception as e:
                            self.send_log(f"Binance liquidation failed for {symbol}: {e}", errnum=1)
                    else:
                        self.send_log(f"[MOCK LIQUIDATION] Closed position for {symbol} of qty {qty}")

                    # Reset portfolio position
                    portfolio["positions"][symbol] = 0.0

            # Clean cash/positions info
            portfolio["positions"] = {}

            # Write split states back
            shm_service.write_portfolio(portfolio)
            shm_service.write_positions_info({}) # Clear position metadata
            self.send_log(f"SELL_ALL completed. Portfolio fully liquidated: {portfolio}")
        except Exception as e:
            self.send_log(f"Failed during liquidate_all_positions: {e}", errnum=1)

    def work(self):
        socket_path = self.UDS_SOCKET_PATH
        self.stopped = False
        self.register_handler("SELL_ALL", self._handle_sell_all_broadcast)
        self.register_handler("BUY", self._handle_order)
        self.register_handler("SELL", self._handle_order)
        self.register_handler("SMART_EXIT", self._handle_order)

        # Initialize Binance Client Service
        self.binance_svc = BinanceClientService()
        self.send_log(f"Executer initialised. Live Mode: {self.binance_svc.is_live()}")

        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass

        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(socket_path)
        server_socket.listen(self.SOCKET_BACKLOG)
        server_socket.setblocking(False)

        connections = []
        last_heartbeat = 0

        self.send_log(f"Executer initialized UDS server on {socket_path}.")

        try:
            while not self.stopped:
                self.check_broadcast()
                now = time.time()
                # Heartbeat check
                if now - last_heartbeat >= self.HEARTBEAT_PERIOD:
                    self.send_heartbeat()
                    last_heartbeat = now

                # Accept new connections (from Analyser or Manager)
                try:
                    conn, addr = server_socket.accept()
                    conn.setblocking(False)
                    connections.append(conn)
                    self.send_log("Executer accepted new UDS connection.")
                except BlockingIOError:
                    pass

                # Read from active connections
                active_connections = []
                for conn in connections:
                    try:
                        data = conn.recv(self.BUFFER_SIZE)
                        if data:
                            active_connections.append(conn)
                            raw_msgs = data.decode('utf-8').strip().split('\n')
                            for msg in raw_msgs:
                                if not msg:
                                    continue
                                try:
                                    packet = MPacket.from_json(msg)
                                except json.JSONDecodeError:
                                    self.send_log(f"Executer failed to parse command: {msg}", errnum=1)
                                    continue
                                self.dispatch(packet)
                        else:
                            # Connection closed
                            conn.close()
                    except BlockingIOError:
                        active_connections.append(conn)
                    except socket.error as e:
                        self.send_log(f"Executer connection read error: {e}", errnum=1)
                        conn.close()

                connections = active_connections
                time.sleep(self.MAIN_LOOP_SLEEP)
        finally:
            server_socket.close()
            try:
                os.unlink(socket_path)
            except OSError:
                pass

    def _execute_soft_chase(self, symbol: str, side: str, quantity: float, leverage: float, signal_price: float, sl_price: float, tp_price: float) -> bool:
        """
        Executes a 'Soft Chase' Limit Pegging strategy for Maker entries.
        1. Set max slippage cap (e.g. 10 basis points from signal price).
        2. Place LIMIT order at Bid1/Ask1.
        3. Loop: check status every 500ms.
        4. If price moves away, cancel and replace up to max slippage.
        """
        max_slippage_bps = 10.0 # 0.1% max slippage
        if side.upper() == "BUY":
            max_price = signal_price * (1.0 + max_slippage_bps / 10000.0)
        else:
            min_price = signal_price * (1.0 - max_slippage_bps / 10000.0)

        # Helper to get current target limit price
        def get_peg_price():
            book = self.binance_svc.get_order_book(symbol, limit=5)
            if side.upper() == "BUY":
                return float(book["bids"][0][0]) if book.get("bids") else signal_price
            else:
                return float(book["asks"][0][0]) if book.get("asks") else signal_price

        current_peg = get_peg_price()
        
        # Check max slippage before even starting
        if side.upper() == "BUY" and current_peg > max_price:
            self.send_log(f"Soft Chase Aborted: Initial Bid {current_peg} > Max {max_price}")
            return False
        if side.upper() == "SELL" and current_peg < min_price:
            self.send_log(f"Soft Chase Aborted: Initial Ask {current_peg} < Min {min_price}")
            return False

        # 1. Place initial LIMIT GTX order
        # Ensure we round price properly for Binance tick size. For simplicity here, use round to 2 decimals or let Binance handle it.
        # We will assume BinanceClientService just passes the float as a string accurately.
        order_resp = self.binance_svc.place_futures_order(
            symbol=symbol, side=side, quantity=quantity, leverage=leverage,
            order_type="LIMIT", price=round(current_peg, 4), time_in_force="GTX",
            sl_price=sl_price, tp_price=tp_price
        )
        if order_resp.get("mock"):
            return True # Dry run

        order_id = order_resp.get("orderId")
        if not order_id:
            return False

        # Soft Chase Loop
        import time
        max_chase_time = 60.0 # Chase for up to 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_chase_time:
            time.sleep(0.5)
            status_resp = self.binance_svc.get_order_status(symbol, order_id)
            status = status_resp.get("status")

            if status in ["FILLED", "REJECTED"]:
                return status == "FILLED"
            
            if status == "CANCELED":
                # Maybe we got GTX rejected (market crossed spread). We should try to place again.
                pass
            
            # If NEW or PARTIALLY_FILLED, check if we need to move the peg
            new_peg = get_peg_price()
            if abs(new_peg - current_peg) > 1e-8: # Price moved
                # Check bounds
                if side.upper() == "BUY" and new_peg > max_price:
                    self.send_log("Soft Chase: Price breached Max Slippage Cap. Walking away.")
                    self.binance_svc.cancel_order(symbol, order_id)
                    return False
                if side.upper() == "SELL" and new_peg < min_price:
                    self.send_log("Soft Chase: Price breached Max Slippage Cap. Walking away.")
                    self.binance_svc.cancel_order(symbol, order_id)
                    return False
                
                # Cancel and replace
                self.binance_svc.cancel_order(symbol, order_id)
                current_peg = new_peg
                new_order = self.binance_svc.place_futures_order(
                    symbol=symbol, side=side, quantity=quantity, leverage=leverage,
                    order_type="LIMIT", price=round(current_peg, 4), time_in_force="GTX"
                    # Omit SL/TP on replace to avoid placing them multiple times, they were placed on the first call
                )
                order_id = new_order.get("orderId")
                if not order_id:
                    return False

        # Timeout reached
        self.send_log(f"Soft Chase Timeout for {symbol}. Canceling order.")
        self.binance_svc.cancel_order(symbol, order_id)
        return False

    def _execute_smart_exit(self, symbol: str, side: str, quantity: float) -> bool:
        import time
        self.send_log(f"[SMART_EXIT] Initiating Best Price Exit for {symbol} ({side} {quantity})")
        
        # Step 1: Passive Post (Maker Attempt at Mid-Price)
        def get_mid_price():
            book = self.binance_svc.get_order_book(symbol, limit=5)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids or not asks:
                return 0.0
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return (best_bid + best_ask) / 2.0

        mid_price = get_mid_price()
        if mid_price == 0.0:
            self.send_log("[SMART_EXIT] Empty orderbook, skipping passive post.")
            order_id = None
        else:
            order_resp = self.binance_svc.place_futures_order(
                symbol=symbol, side=side, quantity=quantity, leverage=1.0,
                order_type="LIMIT", price=round(mid_price, 4), time_in_force="GTX"
            )
            order_id = order_resp.get("orderId") if not order_resp.get("mock") else None

        # Step 2: Patient Chase (30 seconds)
        if order_id:
            start_time = time.time()
            filled = False
            while time.time() - start_time < 30.0:
                time.sleep(1.0)
                status_resp = self.binance_svc.get_order_status(symbol, order_id)
                status = status_resp.get("status")
                if status in ["FILLED", "REJECTED"]:
                    filled = (status == "FILLED")
                    break
                
                # Check if mid price moved
                new_mid = get_mid_price()
                if abs(new_mid - mid_price) > 1e-8:
                    self.binance_svc.cancel_order(symbol, order_id)
                    mid_price = new_mid
                    new_resp = self.binance_svc.place_futures_order(
                        symbol=symbol, side=side, quantity=quantity, leverage=1.0,
                        order_type="LIMIT", price=round(mid_price, 4), time_in_force="GTX"
                    )
                    order_id = new_resp.get("orderId")
                    if not order_id:
                        break
            
            if filled:
                self.send_log(f"[SMART_EXIT] Successfully filled {quantity} via Limit Pegging.")
                return True
                
            if order_id:
                self.binance_svc.cancel_order(symbol, order_id)

        # Step 3 & 4: Order Book Volume Check & Dynamic Slicing
        self.send_log(f"[SMART_EXIT] Patient Chase failed. Evaluating Order Book thickness.")
        
        book = self.binance_svc.get_order_book(symbol, limit=20)
        levels = book.get("bids", []) if side.upper() == "SELL" else book.get("asks", [])
        if not levels:
            self.send_log(f"[SMART_EXIT] No liquidity! Forcing Market Order.")
            self.binance_svc.place_futures_order(symbol=symbol, side=side, quantity=quantity, leverage=1.0)
            return True
            
        top_price = float(levels[0][0])
        threshold_price = top_price * 0.9995 if side.upper() == "SELL" else top_price * 1.0005
        
        resting_liquidity = 0.0
        for price_str, qty_str in levels:
            p = float(price_str)
            if (side.upper() == "SELL" and p >= threshold_price) or (side.upper() == "BUY" and p <= threshold_price):
                resting_liquidity += float(qty_str)
        
        self.send_log(f"[SMART_EXIT] Resting Liquidity within 0.05%: {resting_liquidity:.2f}. Required: {quantity:.2f}")
        
        if resting_liquidity >= quantity:
            self.send_log(f"[SMART_EXIT] Thick book detected. Firing 100% Market Order.")
            self.binance_svc.place_futures_order(symbol=symbol, side=side, quantity=quantity, leverage=1.0)
            return True
            
        # Thin book -> Dynamic Slicing
        num_slices = 4
        slice_qty = quantity / num_slices
        self.send_log(f"[SMART_EXIT] Thin book detected. Slicing order into {num_slices} parts of {slice_qty:.4f}.")
        
        for i in range(num_slices):
            self.send_log(f"[SMART_EXIT] Firing Slice {i+1}/{num_slices}")
            self.binance_svc.place_futures_order(symbol=symbol, side=side, quantity=slice_qty, leverage=1.0)
            
            if i == num_slices - 1:
                break # Done
                
            # Wait for liquidity to replenish (Dynamic Slicing)
            wait_start = time.time()
            max_wait = 20.0
            replenished = False
            while time.time() - wait_start < max_wait:
                time.sleep(1.0)
                current_book = self.binance_svc.get_order_book(symbol, limit=20)
                curr_levels = current_book.get("bids", []) if side.upper() == "SELL" else current_book.get("asks", [])
                
                curr_liq = 0.0
                if curr_levels:
                    curr_top = float(curr_levels[0][0])
                    curr_thresh = curr_top * 0.9995 if side.upper() == "SELL" else curr_top * 1.0005
                    for p_str, q_str in curr_levels:
                        p = float(p_str)
                        if (side.upper() == "SELL" and p >= curr_thresh) or (side.upper() == "BUY" and p <= curr_thresh):
                            curr_liq += float(q_str)
                            
                if curr_liq >= resting_liquidity * 0.9: # 90% replenished
                    self.send_log(f"[SMART_EXIT] Book replenished ({curr_liq:.2f}) after {time.time() - wait_start:.1f}s. Firing next slice early.")
                    replenished = True
                    break
                    
            if not replenished:
                self.send_log(f"[SMART_EXIT] Waited {max_wait}s. Book still wounded. Forcing next slice anyway.")
                
        return True

import asyncio
import json
from typing import Any, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from kalshi_api.utils import WS_URL, create_headers
from src.queue_handler import QueueMessage


class KalshiWSClient:
    def __init__(self, out_q: asyncio.Queue):
        self.out_q = out_q
        self.ws = None

        self.commands: Dict[int, List[Dict[str, Any]]] = {}
        self.orderbook_snapshots: Dict[str, Dict[str, Any]] = {}
        self.next_id = 1

    async def connect(self):
        headers = create_headers("GET", "/trade-api/ws/v2")
        self.ws = await websockets.connect(
            WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        )
        if self.ws:
            print("Connected to Kalshi Websocket!")
        else:
            print("Not Connected to Kalshi Websocket!")

    async def close(self, code: int = 1000, reason: str = "client shutdown"):
        if self.ws is None:
            return
        try:
            await self.ws.close(code=code, reason=reason)
        finally:
            self.ws = None

    async def _send_json(self, payload: Dict[str, Any]) -> bool:
        if self.ws is None:
            print("websocket is not connected")
            return False

        try:
            await self.ws.send(json.dumps(payload))
            return True
        except ConnectionClosed as e:
            print(f"websocket send failed: {e}")
            return False
        except Exception as e:
            print(f"unexpected websocket send failure: {e}")
            return False

    async def subscribe(self, params: Dict[str, Any]) -> bool:
        sub = {
            "id": self.next_id,
            "cmd": "subscribe",
            "params": params,
        }
        self.next_id += 1
        return await self._send_json(sub)

    async def unsubscribe(self, sids: List[int]) -> bool:
        unsub = {
            "id": self.next_id,
            "cmd": "unsubscribe",
            "params": {
                "sids": sids
            }
        }
        self.next_id += 1
        return await self._send_json(unsub)

    async def update(self, sid: int, tickers: List[str]) -> bool:
        update = {
            "id": self.next_id,
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": tickers
            }
        }
        self.next_id += 1
        return await self._send_json(update)

    def clear_snapshots_for(self, tickers: List[str]) -> None:
        for ticker in tickers:
            self.orderbook_snapshots.pop(ticker, None)

    async def wait_for_snapshots(
        self,
        tickers: List[str],
        timeout: float = 5.0,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        remaining = set(tickers)

        while loop.time() < deadline and remaining:
            done = {ticker for ticker in remaining if ticker in self.orderbook_snapshots}
            remaining -= done
            if not remaining:
                break
            await asyncio.sleep(0.05)

        return {ticker: self.orderbook_snapshots.get(ticker) for ticker in tickers}

    async def run(self):
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")

        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type in {"subscribed", "unsubscribed", "ok", "error"} and "id" in data:
                    self.commands.setdefault(data["id"], []).append(data)

                elif msg_type == "orderbook_snapshot":
                    msg = data.get("msg", {}) or {}
                    market_ticker = msg.get("market_ticker")
                    if market_ticker:
                        self.orderbook_snapshots[market_ticker] = data
                        await self.out_q.put(
                            QueueMessage(
                                market_ticker,
                                message,
                            )
                        )

                elif msg_type in {"orderbook_delta", "trade"}:
                    msg = data.get("msg", {}) or {}
                    market_ticker = msg.get("market_ticker")
                    if market_ticker:
                        await self.out_q.put(
                            QueueMessage(
                                market_ticker,
                                message,
                            )
                        )

                elif msg_type in {"market_lifecycle_v2", "event_lifecycle"}:
                    print(message)

                else:
                    print(f"Unknown Data type: {msg_type}")

        except ConnectionClosed as e:
            print(f"websocket closed: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"websocket reader failed: {e}")
import asyncio
import websockets
import json
from typing import Any, Dict, List
from kalshi_api.utils import WS_URL, create_headers
from src.queue_handler import QueueMessage

class KalshiWSClient:
    def __init__(self, out_q: asyncio.Queue):
        self.out_q = out_q
        self.ws = None
        self.commands: Dict[int, Any] = {}
        self.next_id = 1

    async def connect(self):
        headers = create_headers("GET", "/trade-api/ws/v2")
        self.ws = await websockets.connect(WS_URL, additional_headers=headers)
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

    async def subscribe(self, params: Dict[str, Any]):
        sub = {
            "id": self.next_id,
            "cmd": "subscribe",
            "params": params
        }
        self.next_id += 1
        await self.ws.send(json.dumps(sub))

    async def unsubscribe(self, sids: List[int]):
        unsub = {
            "id": self.next_id,
            "cmd": "unsubscribe",
            "params": {
                "sids": sids
            }
        }
        self.next_id += 1
        await self.ws.send(json.dumps(unsub))
        
    async def update(self, sid: int, tickers: List[str]):
        update = {
            "id": self.next_id,
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": tickers
            }
        }
        self.next_id += 1
        await self.ws.send(json.dumps(update))
    
    async def run(self):
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")
        async for message in self.ws:
            data = json.loads(message)
            msg_type = data.get("type")

            # Treat these as command acknowledgements
            if msg_type in {"subscribed", "unsubscribed", "ok", "error"} and "id" in data:
                self.commands.setdefault(data["id"], []).append(data)
                print(message)
            elif msg_type in {"market_lifecycle_v2", "event_lifecycle"}:
                print(message)
            elif msg_type in {"orderbook_snapshot", "orderbook_delta", "trade"}:
                await self.out_q.put(QueueMessage(data.get('msg').get('market_ticker'), message))
            else:
                print(f"Unknown Data type: {msg_type}")
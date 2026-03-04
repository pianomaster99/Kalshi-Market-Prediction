import asyncio
import websockets
import json
from typing import Any, Dict, List
from kalshi_api.utils import WS_URL, create_headers

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

    async def subscribe_to_tickers(self, tickers: List[str]):
        sub = {
            "id": self.next_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "trade"],
                "market_tickers": tickers
                }
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
        async for message in self.ws:
            data = json.loads(message)
            msg_type = data.get("type")

            # Treat these as command acknowledgements
            if msg_type in {"subscribed", "unsubscribed", "ok", "error"} and "id" in data:
                self.commands[data["id"]] = data
                continue
            else:
                await self.out_q.put(data)
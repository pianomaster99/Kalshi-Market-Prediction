import asyncio
from kalshi_api.KalshiWSClient import KalshiWSClient

async def main():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    await client.connect()

    ws_task = asyncio.create_task(client.run())

    tickers = [
        "KXNBAGAME-26MAR09"
    ]
    await client.subscribe({
        "channels": ["orderbook_delta", "trade"],
        "market_tickers": tickers,
    })

    await asyncio.sleep(3)
    print(client.commands)

    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited cleanly.")
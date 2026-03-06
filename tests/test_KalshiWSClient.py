import asyncio
from kalshi_api.KalshiWSClient import KalshiWSClient
from file_handler import file_write, SENTINEL  # export SENTINEL from file_handler

async def main():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    await client.connect()

    # start writer first (so it is ready)
    writer_task = asyncio.create_task(file_write(q, "data/0305.ndjson.gz"))

    # then start websocket reader loop
    ws_task = asyncio.create_task(client.run())

    tickers = ["KXNBAMENTION-26MAR06LALDEN-AIRB", "KXNBAMENTION-26MAR06LALDEN-ALLE"]
    params = {
        "channels": ["orderbook_delta", "trade"],
        "market_tickers": tickers,
    }
    await client.subscribe(params)

    params = {
        "channels": ["market_lifecycle_v2"]
    }

    await client.subscribe(params)

    try:
        # keep running forever
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Stopping...")

    # ---- graceful shutdown ----
    # if your KalshiWSClient has a close/disconnect method, call it
    try:
        await client.close()  # or client.disconnect()
    except Exception:
        pass

    # stop ws task
    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass

    # let writer drain remaining items
    await q.put(SENTINEL)
    await writer_task

if __name__ == "__main__":
    asyncio.run(main())
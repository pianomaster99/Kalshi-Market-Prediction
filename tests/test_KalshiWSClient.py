import asyncio
from kalshi_api.KalshiWSClient import KalshiWSClient
from file_handler import file_write, SENTINEL

async def main():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    await client.connect()

    writer_task = asyncio.create_task(
        file_write(q, "data/26MAR06SYDWAR.ndjson.gz")
    )
    ws_task = asyncio.create_task(client.run())

    tickers = [
        "KXRUGBYNRLMATCH-26MAR06SYDWAR-WAR",
        "KXRUGBYNRLMATCH-26MAR06SYDWAR-TIE",
        "KXRUGBYNRLMATCH-26MAR06SYDWAR-SYD",
    ]
    await client.subscribe({
        "channels": ["orderbook_delta", "trade"],
        "market_tickers": tickers,
    })

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print("Stopping...")
        raise
    finally:
        try:
            await client.close()
        except Exception:
            pass

        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass

        await q.put(SENTINEL)
        await q.join()
        await writer_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited cleanly.")
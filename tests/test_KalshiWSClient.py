import asyncio
from kalshi_api.KalshiWSClient import KalshiWSClient

async def main():
    test_queue = asyncio.Queue()

    testClient = KalshiWSClient(test_queue)
    await testClient.connect()
    asyncio.create_task(testClient.run())

    tickers = ["KXNCAABMENTION-26MAR03NEBUCLA-AIRB"]

    await testClient.subscribe_to_tickers(tickers)

    while True:
        msg = await test_queue.get()
        print(msg)

asyncio.run(main())
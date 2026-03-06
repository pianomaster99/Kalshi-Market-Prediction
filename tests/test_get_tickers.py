import asyncio

from kalshi_api.get_tickers import get_series_list, get_markets_list
from kalshi_api.RequestLimiter import RequestLimiter 

async def main():
    limiter = RequestLimiter(limit_per_second=20)
    await limiter.start()
    try:
        series_list = await get_series_list(limiter, tags="Sports", status="open")
        #Watch out. When you request from too many different series, you get a 429 error. 
        markets_list = []

        for series in series_list:
            series_ticker = series["ticker"]

            markets = await get_markets_list(limiter, series_ticker, status="open")
            markets_list.extend(markets)

        print(markets_list)

    finally:
        await limiter.close()

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import time
from kalshi_api.get_tickers import get_series_list, get_markets_list
from kalshi_api.RequestLimiter import RequestLimiter 


async def main():
    limiter = RequestLimiter(limit_per_second=5)
    await limiter.start()

    try:
        series_list = await get_series_list(limiter, tags="Sports", status="open")

        markets_list = []
        now_ts = int(time.time())
        min_close_ts = now_ts + 3 * 60 * 60

        '''
        for series in series_list:
            series_ticker = series["ticker"]

            markets = await get_markets_list(
                limiter,
                series_ticker,
                status="open"
            )

            markets_list.extend(markets)
        '''

        markets_list = await get_markets_list(
            limiter,
            "KXRUGBYNRLMATCH",
            status="open",
            max_pages=20
        )

        print(markets_list)

        team1 = "Sydney Roosters"
        team2 = "New Zealand Warriors"

        matches = [
            m for m in markets_list
            if team1.lower() in m.get("title", "").lower()
            and team2.lower() in m.get("title", "").lower()
        ]

        print(f"Found {len(matches)} matches")
        for m in matches:
            print("ticker:", m.get("ticker"))
            print("title:", m.get("title"))
            print("yes_sub_title:", m.get("yes_sub_title"))
            print("no_sub_title:", m.get("no_sub_title"))
            print()
        
        # ---- top 5 volumes ----
        '''
        top_markets = sorted(
            markets_list,
            key=lambda m: m.get("volume", 0),
            reverse=True
        )[:10]

        print("\nTop 5 markets by volume:\n")

        for m in top_markets:
            print(m)
            '''

    finally:
        await limiter.close()


if __name__ == "__main__":
    asyncio.run(main())
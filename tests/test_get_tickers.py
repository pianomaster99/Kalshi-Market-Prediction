from kalshi_api.get_tickers import get_series_list, get_events_list, get_markets_list

series_list = get_series_list(status="open")

series_ticker = series_list[3]['ticker']

print(series_ticker)

markets_list = get_markets_list(series_ticker)

print(markets_list)
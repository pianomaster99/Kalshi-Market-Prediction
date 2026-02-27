from kalshi_api.get_tickers import get_series_list, get_events_list, get_markets_list

series_list = get_series_list(tags="Sports")

series = get_series_list()

series_ticker = series[0]['ticker']

events_list = get_events_list(series_ticker)

markets_list = get_markets_list(series_ticker)

print(markets_list)
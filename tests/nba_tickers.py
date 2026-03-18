def generate_tickers(date_str, pairs):
    tickers = {}

    for a, b in pairs:
        matchup = f"{a}{b}"

        ticker_a = f"KXNBAGAME-{date_str}{matchup}-{a}"
        ticker_b = f"KXNBAGAME-{date_str}{matchup}-{b}"

        tickers[ticker_a] = ticker_a
        tickers[ticker_b] = ticker_b

    return tickers


date = "26MAR18"

teams = [
    ["GSW", "BOS"],
    ["POR", "IND"],
    ["OKC", "BKN"],
    ["UTA", "MIN"],
    ["TOR", "CHI"],
    ["LAC", "NOP"],
    ["DEN", "MEM"],
    ["ATL", "DAL"],
    ["LAL", "HOU"]
]

tickers = generate_tickers(date, teams)

print(" ".join(tickers.values()))
import asyncio
from typing import Dict, List, Set, Tuple

from kalshi_api.KalshiWSClient import KalshiWSClient
from src.queue_handler import QueueHandler, SENTINEL


CHANNELS = ["orderbook_delta", "trade"]


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


async def do_write(handler: QueueHandler, tickers: List[str]):
    for ticker in tickers:
        await handler.add_id(ticker)
        print(f"writing enabled for {ticker}")


async def do_delete(handler: QueueHandler, tickers: List[str]):
    for ticker in tickers:
        await handler.remove_id(ticker)
        print(f"writing disabled for {ticker}")


async def do_subscribe_one(
    client: KalshiWSClient,
    ticker_to_sids: Dict[str, Set[int]],
    ticker: str,
    timeout: float = 5.0,
) -> bool:
    client.clear_snapshots_for([ticker])

    req_id = client.next_id
    await client.subscribe(
        {
            "channels": CHANNELS,
            "market_tickers": [ticker],
        }
    )

    snapshots = await client.wait_for_snapshots([ticker], timeout=timeout)
    snapshot = snapshots.get(ticker)

    if snapshot is None:
        print(f"subscribe failed for {ticker}: no orderbook_snapshot received")
        client.commands.pop(req_id, None)
        return False

    msg = snapshot.get("msg", {}) or {}
    snapshot_ticker = msg.get("market_ticker")
    market_id = msg.get("market_id")

    if not market_id:
        print(f"ticker does not exist or is invalid: {ticker}")
        print(
            f"snapshot returned market_ticker={snapshot_ticker!r}, "
            f"market_id={market_id!r}"
        )
        client.commands.pop(req_id, None)
        return False

    sids = set()
    for ack in client.commands.pop(req_id, []):
        ack_msg = ack.get("msg", {}) or {}
        sid = ack_msg.get("sid")
        if sid is None:
            sid = ack.get("sid")
        if sid is not None:
            sids.add(sid)

    ticker_to_sids[ticker] = sids
    print(f"subscribed {ticker} (sids={sorted(sids)})")
    return True


async def do_subscribe(
    client: KalshiWSClient,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    for ticker in tickers:
        await do_subscribe_one(client, ticker_to_sids, ticker)


async def unsubscribe_sid_group(
    client: KalshiWSClient,
    sids: List[int],
    timeout: float = 5.0,
) -> bool:
    if not sids:
        return True

    req_id = client.next_id
    await client.unsubscribe(sids)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        acks = client.commands.get(req_id, [])
        # We may receive fewer than len(sids) explicit responses,
        # so break as soon as something arrives and no more are coming quickly.
        if acks:
            await asyncio.sleep(0.1)
            break
        await asyncio.sleep(0.05)

    acks = client.commands.pop(req_id, [])

    bad = [ack for ack in acks if ack.get("type") not in {"unsubscribed", "ok"}]
    if bad:
        print(f"unsubscribe failed for shared sids={sids}: {bad}")
        return False

    if not acks:
        print(f"unsubscribe timed out for shared sids={sids}")
        return False

    return True


def group_tickers_by_sids(
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
) -> Tuple[Dict[tuple[int, ...], List[str]], List[str]]:
    sid_group_to_tickers: Dict[tuple[int, ...], List[str]] = {}
    missing: List[str] = []

    for ticker in tickers:
        sids = ticker_to_sids.get(ticker)
        if sids is None:
            missing.append(ticker)
            continue

        key = tuple(sorted(sids))
        sid_group_to_tickers.setdefault(key, []).append(ticker)

    return sid_group_to_tickers, missing


async def do_unsubscribe(
    client: KalshiWSClient,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    sid_group_to_tickers, missing = group_tickers_by_sids(ticker_to_sids, tickers)

    for ticker in missing:
        print(f"not currently tracking sids for {ticker}")

    for sid_key, grouped_tickers in sid_group_to_tickers.items():
        sids = list(sid_key)

        success = await unsubscribe_sid_group(client, sids)
        if not success:
            print(
                f"did not remove local tracking for {grouped_tickers} "
                f"because unsubscribe failed"
            )
            continue

        for ticker in grouped_tickers:
            ticker_to_sids.pop(ticker, None)
            print(f"unsubscribed {ticker} (removed shared sids={sids})")


async def do_subwrite(
    client: KalshiWSClient,
    handler: QueueHandler,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    for ticker in tickers:
        success = await do_subscribe_one(client, ticker_to_sids, ticker)
        if not success:
            print(f"not enabling writing for {ticker} because subscription was unsuccessful")
            continue
        await do_write(handler, [ticker])


async def do_unsubdelete(
    client: KalshiWSClient,
    handler: QueueHandler,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    sid_group_to_tickers, missing = group_tickers_by_sids(ticker_to_sids, tickers)

    for ticker in missing:
        print(f"not currently tracking sids for {ticker}")

    for sid_key, grouped_tickers in sid_group_to_tickers.items():
        sids = list(sid_key)

        success = await unsubscribe_sid_group(client, sids)
        if not success:
            print(
                f"not disabling writing for {grouped_tickers} "
                f"because unsubscribe failed"
            )
            continue

        for ticker in grouped_tickers:
            ticker_to_sids.pop(ticker, None)
            print(f"unsubscribed {ticker} (removed shared sids={sids})")
            await do_delete(handler, [ticker])


async def master():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    handler = QueueHandler(q, output_dir="data", debug_every=5000)

    await client.connect()

    ws_task = asyncio.create_task(client.run())
    writer_task = asyncio.create_task(handler.run())

    ticker_to_sids: Dict[str, Set[int]] = {}

    print("Commands:")
    print("  write TICKER [TICKER ...]")
    print("  delete TICKER [TICKER ...]")
    print("  subscribe TICKER [TICKER ...]")
    print("  unsubscribe TICKER [TICKER ...]")
    print("  subwrite TICKER [TICKER ...]")
    print("  unsubdelete TICKER [TICKER ...]")
    print("  list")
    print("  quit")

    try:
        while True:
            raw = (await ainput("> ")).strip()
            if not raw:
                continue

            parts = raw.split()
            command = parts[0].lower()
            args = parts[1:]

            if command == "quit":
                break

            elif command == "list":
                ids = await handler.list_ids()
                print("enabled write ids:")
                if not ids:
                    print("  (none)")
                else:
                    for x in ids:
                        print(f"  {x}")

                print("subscribed tickers:")
                if not ticker_to_sids:
                    print("  (none)")
                else:
                    for ticker, sids in ticker_to_sids.items():
                        print(f"  {ticker} -> sids={sorted(sids)}")

            elif command == "write":
                if not args:
                    print("usage: write TICKER [TICKER ...]")
                    continue
                await do_write(handler, args)

            elif command == "delete":
                if not args:
                    print("usage: delete TICKER [TICKER ...]")
                    continue
                await do_delete(handler, args)

            elif command == "subscribe":
                if not args:
                    print("usage: subscribe TICKER [TICKER ...]")
                    continue
                await do_subscribe(client, ticker_to_sids, args)

            elif command == "unsubscribe":
                if not args:
                    print("usage: unsubscribe TICKER [TICKER ...]")
                    continue
                await do_unsubscribe(client, ticker_to_sids, args)

            elif command == "subwrite":
                if not args:
                    print("usage: subwrite TICKER [TICKER ...]")
                    continue
                await do_subwrite(client, handler, ticker_to_sids, args)

            elif command == "unsubdelete":
                if not args:
                    print("usage: unsubdelete TICKER [TICKER ...]")
                    continue
                await do_unsubdelete(client, handler, ticker_to_sids, args)

            else:
                print(f"unknown command: {command}")

    finally:
        await client.close()
        await q.put(SENTINEL)
        await q.join()

        writer_task.cancel()
        ws_task.cancel()

        for task in (writer_task, ws_task):
            try:
                await task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(master())
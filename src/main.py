import asyncio
from typing import Dict, List, Set

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


async def wait_for_update_ack(
    client: KalshiWSClient,
    req_id: int,
    timeout: float = 5.0,
):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        acks = client.commands.get(req_id, [])
        if acks:
            return client.commands.pop(req_id, [])
        await asyncio.sleep(0.05)

    return client.commands.pop(req_id, [])


async def create_initial_shared_subscription(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    ticker: str,
    timeout: float = 5.0,
) -> bool:
    """
    First real subscribe on a fresh connection.
    Creates the shared per-channel sids using a real ticker.
    Validity is determined only from orderbook_snapshot.market_id.
    """
    client.clear_snapshots_for([ticker])

    req_id = client.next_id
    await client.subscribe(
        {
            "channels": CHANNELS,
            "market_tickers": [ticker],
        }
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    found: Dict[str, int] = {}

    while loop.time() < deadline:
        acks = client.commands.get(req_id, [])
        for ack in acks:
            msg = ack.get("msg", {}) or {}
            channel = msg.get("channel")
            sid = msg.get("sid")
            if channel in CHANNELS and sid is not None:
                found[channel] = sid

        if len(found) == len(CHANNELS):
            break

        await asyncio.sleep(0.05)

    acks = client.commands.pop(req_id, [])

    for ack in acks:
        msg = ack.get("msg", {}) or {}
        channel = msg.get("channel")
        sid = msg.get("sid")
        if channel in CHANNELS and sid is not None:
            found[channel] = sid

    missing = [ch for ch in CHANNELS if ch not in found]
    if missing:
        print(f"initial subscribe failed for {ticker}: missing channel sids {missing}")
        print(f"acks: {acks}")
        return False

    snapshots = await client.wait_for_snapshots([ticker], timeout=timeout)
    snapshot = snapshots.get(ticker)

    if snapshot is None:
        print(f"subscribe failed for {ticker}: no orderbook_snapshot received")
        return False

    msg = snapshot.get("msg", {}) or {}
    market_id = msg.get("market_id")
    snapshot_ticker = msg.get("market_ticker")

    if not market_id:
        print(f"ticker does not exist or is invalid: {ticker}")
        print(
            f"snapshot returned market_ticker={snapshot_ticker!r}, "
            f"market_id={market_id!r}"
        )
        return False

    channel_to_sid.clear()
    channel_to_sid.update(found)
    print(f"subscribed {ticker} (shared sids={[channel_to_sid[ch] for ch in CHANNELS]})")
    return True


async def reconcile_subscriptions(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    timeout: float = 5.0,
) -> bool:
    """
    Update all existing shared subscriptions so they track exactly active_tickers.
    Assumes shared sids already exist.
    """
    missing_channels = [ch for ch in CHANNELS if ch not in channel_to_sid]
    if missing_channels:
        print(f"cannot reconcile subscriptions: missing shared sids for {missing_channels}")
        return False

    tickers_sorted = sorted(active_tickers)

    for channel in CHANNELS:
        sid = channel_to_sid[channel]
        req_id = client.next_id
        await client.update(sid, tickers_sorted)

        acks = await wait_for_update_ack(client, req_id, timeout=timeout)
        bad = [ack for ack in acks if ack.get("type") not in {"ok", "subscribed", "updated"}]

        if bad:
            print(f"update_subscription failed for channel={channel}, sid={sid}: {bad}")
            return False

        if not acks:
            print(f"update_subscription timed out for channel={channel}, sid={sid}")
            return False

    return True


async def validate_ticker_via_snapshot(
    client: KalshiWSClient,
    ticker: str,
    timeout: float = 5.0,
) -> bool:
    """
    A ticker is valid iff we receive an orderbook_snapshot with non-empty market_id.
    """
    client.clear_snapshots_for([ticker])

    snapshots = await client.wait_for_snapshots([ticker], timeout=timeout)
    snapshot = snapshots.get(ticker)

    if snapshot is None:
        print(f"subscribe failed for {ticker}: no orderbook_snapshot received")
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
        return False

    return True


async def add_ticker_subscription(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    ticker: str,
    timeout: float = 5.0,
) -> bool:
    if ticker in active_tickers:
        print(f"already subscribed {ticker}")
        return True

    # First ticker on a fresh connection: real subscribe, not empty bootstrap
    if not channel_to_sid:
        ok = await create_initial_shared_subscription(
            client,
            channel_to_sid,
            ticker,
            timeout=timeout,
        )
        if not ok:
            return False

        active_tickers.add(ticker)
        return True

    proposed = set(active_tickers)
    proposed.add(ticker)

    ok = await reconcile_subscriptions(
        client,
        channel_to_sid,
        proposed,
        timeout=timeout,
    )
    if not ok:
        print(f"failed to add {ticker}: could not update shared subscriptions")
        return False

    valid = await validate_ticker_via_snapshot(client, ticker, timeout=timeout)
    if not valid:
        rollback_ok = await reconcile_subscriptions(
            client,
            channel_to_sid,
            active_tickers,
            timeout=timeout,
        )
        if not rollback_ok:
            print(f"warning: rollback after invalid ticker {ticker} may have failed")
        return False

    active_tickers.add(ticker)
    print(f"subscribed {ticker} (shared sids={[channel_to_sid[ch] for ch in CHANNELS]})")
    return True


async def remove_ticker_subscription(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    ticker: str,
    timeout: float = 5.0,
) -> bool:
    if ticker not in active_tickers:
        print(f"not currently subscribed: {ticker}")
        return False

    proposed = set(active_tickers)
    proposed.discard(ticker)

    ok = await reconcile_subscriptions(
        client,
        channel_to_sid,
        proposed,
        timeout=timeout,
    )
    if not ok:
        print(f"failed to remove {ticker}: could not update shared subscriptions")
        return False

    active_tickers.discard(ticker)
    print(f"unsubscribed {ticker}")
    return True


async def do_subscribe(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    tickers: List[str],
):
    for ticker in tickers:
        await add_ticker_subscription(client, channel_to_sid, active_tickers, ticker)


async def do_unsubscribe(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    tickers: List[str],
):
    for ticker in tickers:
        await remove_ticker_subscription(client, channel_to_sid, active_tickers, ticker)


async def do_subwrite(
    client: KalshiWSClient,
    handler: QueueHandler,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    tickers: List[str],
):
    for ticker in tickers:
        success = await add_ticker_subscription(client, channel_to_sid, active_tickers, ticker)
        if not success:
            print(f"not enabling writing for {ticker} because subscription was unsuccessful")
            continue
        await do_write(handler, [ticker])


async def do_unsubdelete(
    client: KalshiWSClient,
    handler: QueueHandler,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    tickers: List[str],
):
    for ticker in tickers:
        success = await remove_ticker_subscription(client, channel_to_sid, active_tickers, ticker)
        if not success:
            print(f"not disabling writing for {ticker} because unsubscribe was unsuccessful")
            continue
        await do_delete(handler, [ticker])


async def master():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    handler = QueueHandler(q, output_dir="data", debug_every=5000)

    await client.connect()

    ws_task = asyncio.create_task(client.run())
    writer_task = asyncio.create_task(handler.run())

    # Shared subscriptions across the whole websocket connection.
    channel_to_sid: Dict[str, int] = {}

    # Tickers currently included in the shared websocket subscriptions.
    active_tickers: Set[str] = set()

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
                if not active_tickers:
                    print("  (none)")
                else:
                    for ticker in sorted(active_tickers):
                        print(f"  {ticker}")

                print("shared channel sids:")
                if not channel_to_sid:
                    print("  (none)")
                else:
                    for channel in CHANNELS:
                        sid = channel_to_sid.get(channel)
                        print(f"  {channel} -> sid={sid}")

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
                await do_subscribe(client, channel_to_sid, active_tickers, args)

            elif command == "unsubscribe":
                if not args:
                    print("usage: unsubscribe TICKER [TICKER ...]")
                    continue
                await do_unsubscribe(client, channel_to_sid, active_tickers, args)

            elif command == "subwrite":
                if not args:
                    print("usage: subwrite TICKER [TICKER ...]")
                    continue
                await do_subwrite(client, handler, channel_to_sid, active_tickers, args)

            elif command == "unsubdelete":
                if not args:
                    print("usage: unsubdelete TICKER [TICKER ...]")
                    continue
                await do_unsubdelete(client, handler, channel_to_sid, active_tickers, args)

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
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


async def ensure_base_subscription(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    timeout: float = 5.0,
) -> bool:
    """
    Ensure we have one shared subscription per channel on this connection.
    Kalshi appears to use shared sids across all tickers on the connection,
    so we create them once and then update them.
    """
    missing_channels = [ch for ch in CHANNELS if ch not in channel_to_sid]
    if not missing_channels:
        return True

    req_id = client.next_id
    await client.subscribe(
        {
            "channels": CHANNELS,
            # dummy ticker just to create the shared channel subscriptions;
            # the snapshot validation happens later in reconcile step
            "market_tickers": [],
        }
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        acks = client.commands.get(req_id, [])
        sids_found = {}

        for ack in acks:
            msg = ack.get("msg", {}) or {}
            channel = msg.get("channel")
            sid = msg.get("sid")
            if channel in CHANNELS and sid is not None:
                sids_found[channel] = sid

        if len(sids_found) == len(CHANNELS):
            channel_to_sid.update(sids_found)
            client.commands.pop(req_id, None)
            return True

        await asyncio.sleep(0.05)

    acks = client.commands.pop(req_id, [])
    sids_found = {}
    for ack in acks:
        msg = ack.get("msg", {}) or {}
        channel = msg.get("channel")
        sid = msg.get("sid")
        if channel in CHANNELS and sid is not None:
            sids_found[channel] = sid

    channel_to_sid.update(sids_found)

    missing_channels = [ch for ch in CHANNELS if ch not in channel_to_sid]
    if missing_channels:
        print(f"failed to establish base shared subscriptions: missing {missing_channels}")
        print(f"acks: {acks}")
        return False

    return True


async def reconcile_subscriptions(
    client: KalshiWSClient,
    channel_to_sid: Dict[str, int],
    active_tickers: Set[str],
    timeout: float = 5.0,
) -> bool:
    """
    Update all shared subscriptions so that they track exactly active_tickers.
    """
    ok = await ensure_base_subscription(client, channel_to_sid, timeout=timeout)
    if not ok:
        return False

    tickers_sorted = sorted(active_tickers)

    for channel in CHANNELS:
        sid = channel_to_sid[channel]
        req_id = client.next_id
        await client.update(sid, tickers_sorted)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            acks = client.commands.get(req_id, [])
            if acks:
                break
            await asyncio.sleep(0.05)

        acks = client.commands.pop(req_id, [])

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
    """
    Add ticker to the shared subscription set, validate it via snapshot,
    and roll back if invalid.
    """
    if ticker in active_tickers:
        print(f"already subscribed {ticker}")
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
        # Roll back to prior ticker set
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
    """
    Remove ticker from the shared subscription set by updating the shared subscriptions.
    """
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
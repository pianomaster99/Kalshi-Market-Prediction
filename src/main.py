import asyncio
from typing import Dict, List, Set

from kalshi_api.KalshiWSClient import KalshiWSClient
from src.queue_handler import QueueHandler, SENTINEL


CHANNELS = ["orderbook_delta", "trade"]


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


async def wait_for_command_acks(
    client: KalshiWSClient,
    req_id: int,
    expected: int,
    timeout: float = 5.0,
):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    out = []

    while loop.time() < deadline:
        msgs = client.commands.get(req_id, [])
        while msgs:
            out.append(msgs.pop(0))

        if len(out) >= expected:
            client.commands.pop(req_id, None)
            return out

        await asyncio.sleep(0.05)

    # clean up if empty
    if req_id in client.commands and not client.commands[req_id]:
        client.commands.pop(req_id, None)

    return out


async def do_write(handler: QueueHandler, tickers: List[str]):
    for ticker in tickers:
        await handler.add_id(ticker)
        print(f"writing enabled for {ticker}")


async def do_delete(handler: QueueHandler, tickers: List[str]):
    for ticker in tickers:
        await handler.remove_id(ticker)
        print(f"writing disabled for {ticker}")


async def do_subscribe(
    client: KalshiWSClient,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    if not tickers:
        return

    params = {
        "channels": CHANNELS,
        "market_tickers": tickers,
    }

    req_id = client.next_id
    await client.subscribe(params)

    acks = await wait_for_command_acks(client, req_id, expected=len(CHANNELS))

    if len(acks) < len(CHANNELS):
        print(f"subscribe timed out for: {', '.join(tickers)}")
        print(f"partial acks: {acks}")
        return

    bad = [ack for ack in acks if ack.get("type") not in {"subscribed", "ok"}]
    if bad:
        print(f"subscribe failed for {tickers}: {bad}")
        return

    sids = set()
    for ack in acks:
        sid = ack.get("msg", {}).get("sid")
        if sid is None:
            sid = ack.get("sid")
        if sid is not None:
            sids.add(sid)

    for ticker in tickers:
        ticker_to_sids.setdefault(ticker, set()).update(sids)
        print(f"subscribed {ticker} (sids={sorted(ticker_to_sids[ticker])})")


async def do_unsubscribe(
    client: KalshiWSClient,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    if not tickers:
        return

    sids_to_remove = set()
    missing = []

    for ticker in tickers:
        sids = ticker_to_sids.get(ticker)
        if not sids:
            missing.append(ticker)
        else:
            sids_to_remove.update(sids)

    for ticker in missing:
        print(f"not currently tracking sids for {ticker}")

    if not sids_to_remove:
        return

    req_id = client.next_id
    await client.unsubscribe(list(sids_to_remove))

    acks = await wait_for_command_acks(client, req_id, expected=len(sids_to_remove), timeout=5.0)

    if len(acks) < len(sids_to_remove):
        print(f"unsubscribe timed out for sids={sorted(sids_to_remove)}")
        print(f"partial acks: {acks}")
        return

    bad = [ack for ack in acks if ack.get("type") not in {"unsubscribed", "ok"}]
    if bad:
        print(f"unsubscribe failed for {tickers}: {bad}")
        return

    for ticker in tickers:
        if ticker in ticker_to_sids:
            removed = sorted(ticker_to_sids[ticker])
            ticker_to_sids.pop(ticker, None)
            print(f"unsubscribed {ticker} (removed sids={removed})")


async def do_subwrite(
    client: KalshiWSClient,
    handler: QueueHandler,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    if not tickers:
        return

    await do_write(handler, tickers)
    await do_subscribe(client, ticker_to_sids, tickers)


async def do_unsubdelete(
    client: KalshiWSClient,
    handler: QueueHandler,
    ticker_to_sids: Dict[str, Set[int]],
    tickers: List[str],
):
    if not tickers:
        return

    await do_unsubscribe(client, ticker_to_sids, tickers)
    await do_delete(handler, tickers)


async def master():
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)
    handler = QueueHandler(q, output_dir="data", debug_every=5000)

    await client.connect()

    ws_task = asyncio.create_task(client.run())
    writer_task = asyncio.create_task(handler.run())

    # ticker -> set of sids across channels
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
                for x in ids:
                    print(f"  {x}")

                print("subscribed tickers:")
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
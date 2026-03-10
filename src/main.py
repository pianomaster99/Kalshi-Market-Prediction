import asyncio
import os
from typing import Any, Dict, List, Optional, Set

import websockets

from kalshi_api.KalshiWSClient import KalshiWSClient
from src.queue_handler import QueueHandler, SENTINEL


CHANNELS = ["orderbook_delta", "trade"]
ACK_TIMEOUT = 5.0


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


async def wait_for_command_acks(
    client: KalshiWSClient,
    req_id: int,
    expected: int,
    timeout: float = ACK_TIMEOUT,
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

    if req_id in client.commands and not client.commands[req_id]:
        client.commands.pop(req_id, None)

    return out


async def do_write(handler: QueueHandler, tickers: List[str]) -> None:
    for ticker in tickers:
        await handler.add_id(ticker)
        print(f"writing enabled for {ticker}")


async def do_delete(handler: QueueHandler, tickers: List[str]) -> None:
    for ticker in tickers:
        await handler.remove_id(ticker)
        print(f"writing disabled for {ticker}")


class SubscriptionManager:
    def __init__(self, client: KalshiWSClient):
        self.client = client
        self.active_tickers: Set[str] = set()
        self.shared_sids: Set[int] = set()

    async def _safe_subscribe_call(self, params: Dict[str, Any]) -> Optional[int]:
        req_id = self.client.next_id
        ok = await self.client.subscribe(params)
        if not ok:
            return None
        return req_id

    async def _safe_unsubscribe_call(self, sids: List[int]) -> Optional[int]:
        req_id = self.client.next_id
        ok = await self.client.unsubscribe(sids)
        if not ok:
            return None
        return req_id

    async def _subscribe_fresh(self, tickers: List[str], timeout: float = ACK_TIMEOUT) -> bool:
        if not tickers:
            self.active_tickers.clear()
            self.shared_sids.clear()
            return True

        unique_tickers = sorted(set(tickers))
        self.client.clear_snapshots_for(unique_tickers)

        req_id = await self._safe_subscribe_call(
            {
                "channels": CHANNELS,
                "market_tickers": unique_tickers,
            }
        )
        if req_id is None:
            print(f"subscribe failed immediately for: {', '.join(unique_tickers)}")
            return False

        acks = await wait_for_command_acks(
            self.client,
            req_id,
            expected=len(CHANNELS),
            timeout=timeout,
        )

        if len(acks) < len(CHANNELS):
            print(f"subscribe timed out for: {', '.join(unique_tickers)}")
            print(f"partial acks: {acks}")
            return False

        bad = [ack for ack in acks if ack.get("type") not in {"subscribed", "ok"}]
        if bad:
            print(f"subscribe failed for {unique_tickers}: {bad}")
            return False

        sids = set()
        for ack in acks:
            sid = ack.get("sid")
            if sid is None:
                sid = ack.get("msg", {}).get("sid")
            if sid is not None:
                sids.add(sid)

        if len(sids) < len(CHANNELS):
            print(f"subscribe failed for {unique_tickers}: missing sids")
            print(f"acks: {acks}")
            return False

        snapshots = await self.client.wait_for_snapshots(unique_tickers, timeout=timeout)

        invalid = []
        for ticker in unique_tickers:
            snapshot = snapshots.get(ticker)
            if snapshot is None:
                invalid.append((ticker, "no snapshot"))
                continue

            msg = snapshot.get("msg", {}) or {}
            market_id = msg.get("market_id")
            if not market_id:
                invalid.append((ticker, f"invalid market_id={market_id!r}"))

        if invalid:
            print("subscribe validation failed:")
            for ticker, reason in invalid:
                print(f"  {ticker}: {reason}")

            cleanup_req_id = await self._safe_unsubscribe_call(sorted(sids))
            if cleanup_req_id is not None:
                await wait_for_command_acks(
                    self.client,
                    cleanup_req_id,
                    expected=len(sids),
                    timeout=timeout,
                )
            return False

        self.active_tickers = set(unique_tickers)
        self.shared_sids = sids
        print(f"subscribed {', '.join(unique_tickers)} (sids={sorted(self.shared_sids)})")
        return True

    async def _unsubscribe_all(self, timeout: float = ACK_TIMEOUT) -> bool:
        if not self.shared_sids:
            self.active_tickers.clear()
            return True

        sids = sorted(self.shared_sids)

        req_id = await self._safe_unsubscribe_call(sids)
        if req_id is None:
            print(f"unsubscribe failed immediately for sids={sids}")
            return False

        acks = await wait_for_command_acks(
            self.client,
            req_id,
            expected=len(sids),
            timeout=timeout,
        )

        if len(acks) < len(sids):
            print(f"unsubscribe timed out for sids={sids}")
            print(f"partial acks: {acks}")
            return False

        bad = [ack for ack in acks if ack.get("type") not in {"unsubscribed", "ok"}]
        if bad:
            print(f"unsubscribe failed for sids={sids}: {bad}")
            return False

        self.active_tickers.clear()
        self.shared_sids.clear()
        return True

    async def subscribe(self, ticker: str, timeout: float = ACK_TIMEOUT) -> bool:
        if ticker in self.active_tickers:
            print(f"already subscribed {ticker}")
            return True

        if not self.active_tickers:
            return await self._subscribe_fresh([ticker], timeout=timeout)

        self.client.clear_snapshots_for([ticker])

        req_id = await self._safe_subscribe_call(
            {
                "channels": CHANNELS,
                "market_tickers": [ticker],
            }
        )
        if req_id is None:
            print(f"subscribe failed immediately for: {ticker}")
            return False

        acks = await wait_for_command_acks(
            self.client,
            req_id,
            expected=len(CHANNELS),
            timeout=timeout,
        )

        if len(acks) < len(CHANNELS):
            print(f"subscribe timed out for: {ticker}")
            print(f"partial acks: {acks}")
            return False

        bad = [ack for ack in acks if ack.get("type") not in {"subscribed", "ok"}]
        if bad:
            print(f"subscribe failed for {ticker}: {bad}")
            return False

        sids = set()
        for ack in acks:
            sid = ack.get("sid")
            if sid is None:
                sid = ack.get("msg", {}).get("sid")
            if sid is not None:
                sids.add(sid)

        if not sids:
            print(f"subscribe failed for {ticker}: no sids returned")
            print(f"acks: {acks}")
            return False

        snapshots = await self.client.wait_for_snapshots([ticker], timeout=timeout)
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

        self.active_tickers.add(ticker)
        self.shared_sids.update(sids)
        print(f"subscribed {ticker} (shared sids={sorted(self.shared_sids)})")
        return True

    async def unsubscribe(self, ticker: str, timeout: float = ACK_TIMEOUT) -> bool:
        if ticker not in self.active_tickers:
            print(f"not currently subscribed: {ticker}")
            return False

        remaining = sorted(self.active_tickers - {ticker})

        if not remaining:
            ok = await self._unsubscribe_all(timeout=timeout)
            if ok:
                print(f"unsubscribed {ticker}")
            return ok

        old_active = set(self.active_tickers)
        old_sids = set(self.shared_sids)

        ok = await self._unsubscribe_all(timeout=timeout)
        if not ok:
            print(f"failed to remove {ticker}: could not unsubscribe current shared subscriptions")
            self.active_tickers = old_active
            self.shared_sids = old_sids
            return False

        ok = await self._subscribe_fresh(remaining, timeout=timeout)
        if not ok:
            print(f"failed to rebuild subscriptions after removing {ticker}")
            return False

        print(f"unsubscribed {ticker}")
        return True

    def print_status(self) -> None:
        print("subscribed tickers:")
        if not self.active_tickers:
            print("  (none)")
        else:
            for ticker in sorted(self.active_tickers):
                print(f"  {ticker}")

        print("shared sids:")
        if not self.shared_sids:
            print("  (none)")
        else:
            for sid in sorted(self.shared_sids):
                print(f"  sid={sid}")


async def do_subscribe(subs: SubscriptionManager, tickers: List[str]) -> None:
    for ticker in tickers:
        await subs.subscribe(ticker)


async def do_unsubscribe(subs: SubscriptionManager, tickers: List[str]) -> None:
    for ticker in tickers:
        await subs.unsubscribe(ticker)


async def do_subwrite(
    subs: SubscriptionManager,
    handler: QueueHandler,
    tickers: List[str],
) -> None:
    for ticker in tickers:
        await handler.add_id(ticker)

        success = await subs.subscribe(ticker)
        if not success:
            await handler.remove_id(ticker)
            print(f"subscription failed for {ticker}; writing not enabled")
            continue

        print(f"subscribed {ticker} and writing enabled")


async def do_unsubdelete(
    subs: SubscriptionManager,
    handler: QueueHandler,
    tickers: List[str],
) -> None:
    for ticker in tickers:
        success = await subs.unsubscribe(ticker)
        if not success:
            print(f"unsubscribe failed for {ticker}; disabling writing anyway")

        await handler.remove_id(ticker)
        print(f"writing disabled for {ticker}")


def print_help() -> None:
    print("Commands:")
    print("  write TICKER [TICKER ...]")
    print("  delete TICKER [TICKER ...]")
    print("  subscribe TICKER [TICKER ...]")
    print("  unsubscribe TICKER [TICKER ...]")
    print("  subwrite TICKER [TICKER ...]")
    print("  unsubdelete TICKER [TICKER ...]")
    print("  list")
    print("  help")
    print("  quit")


async def shutdown_tasks(tasks: List[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()

    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"background task failed during shutdown: {e}")


async def master() -> None:
    q = asyncio.Queue(maxsize=50_000)

    client = KalshiWSClient(q)

    output_dir = os.environ.get("KALSHI_OUTPUT_DIR", "data")
    handler = QueueHandler(
        q,
        output_dir=output_dir,
        debug_every=5000,
        flush_every=200,
    )
    subs = SubscriptionManager(client)

    await client.connect()

    ws_task = asyncio.create_task(client.run(), name="kalshi-ws-reader")
    writer_task = asyncio.create_task(handler.run(), name="queue-writer")

    print_help()

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

            elif command == "help":
                print_help()

            elif command == "list":
                ids = await handler.list_ids()
                print("enabled write ids:")
                if not ids:
                    print("  (none)")
                else:
                    for x in ids:
                        print(f"  {x}")
                subs.print_status()

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
                await do_subscribe(subs, args)

            elif command == "unsubscribe":
                if not args:
                    print("usage: unsubscribe TICKER [TICKER ...]")
                    continue
                await do_unsubscribe(subs, args)

            elif command == "subwrite":
                if not args:
                    print("usage: subwrite TICKER [TICKER ...]")
                    continue
                await do_subwrite(subs, handler, args)

            elif command == "unsubdelete":
                if not args:
                    print("usage: unsubdelete TICKER [TICKER ...]")
                    continue
                await do_unsubdelete(subs, handler, args)

            else:
                print(f"unknown command: {command}")
                print("type 'help' to see available commands")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")

    finally:
        try:
            await client.close()
        except Exception as e:
            print(f"client close failed: {e}")

        try:
            await q.put(SENTINEL)
            await q.join()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"queue shutdown failed: {e}")

        await shutdown_tasks([writer_task, ws_task])


if __name__ == "__main__":
    asyncio.run(master())
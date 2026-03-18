import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from data_collection.kalshi_api.KalshiWSClient import KalshiWSClient
from data_collection.src.queue_handler import QueueHandler, SENTINEL


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
    for ticker in sorted(set(tickers)):
        await handler.add_id(ticker)
        print(f"writing enabled for {ticker}")


async def do_delete(handler: QueueHandler, tickers: List[str]) -> None:
    for ticker in sorted(set(tickers)):
        await handler.remove_id(ticker)
        print(f"writing disabled for {ticker}")


@dataclass
class SubscriptionGroup:
    tickers: FrozenSet[str]
    sids: Set[int] = field(default_factory=set)


class SubscriptionManager:
    """
    Immutable group model:

    - Each `subwrite A B C` creates one subscription group for exactly {A, B, C}.
    - `unsubdelete` must match that exact same group.
    - Partial removal is rejected.
    - A ticker may belong to at most one active group.
    """

    def __init__(self, client: KalshiWSClient):
        self.client = client
        self.groups: Dict[FrozenSet[str], SubscriptionGroup] = {}
        self.ticker_to_group: Dict[str, FrozenSet[str]] = {}

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

    async def subscribe_group(
        self,
        tickers: List[str],
        timeout: float = ACK_TIMEOUT,
    ) -> bool:
        unique = tuple(sorted(set(tickers)))
        if not unique:
            print("cannot subscribe empty group")
            return False

        group_key = frozenset(unique)

        if group_key in self.groups:
            print(f"group already subscribed: {', '.join(sorted(group_key))}")
            return True

        conflicts = []
        for ticker in unique:
            existing_group = self.ticker_to_group.get(ticker)
            if existing_group is not None and existing_group != group_key:
                conflicts.append((ticker, existing_group))

        if conflicts:
            print("cannot subwrite because some tickers already belong to another group:")
            for ticker, existing_group in conflicts:
                print(f"  {ticker} is already in group: {', '.join(sorted(existing_group))}")
            return False

        self.client.clear_snapshots_for(list(unique))

        req_id = await self._safe_subscribe_call(
            {
                "channels": CHANNELS,
                "market_tickers": list(unique),
            }
        )
        if req_id is None:
            print(f"subscribe failed immediately for group: {', '.join(unique)}")
            return False

        acks = await wait_for_command_acks(
            self.client,
            req_id,
            expected=len(CHANNELS),
            timeout=timeout,
        )

        if len(acks) < len(CHANNELS):
            print(f"subscribe timed out for group: {', '.join(unique)}")
            print(f"partial acks: {acks}")
            return False

        bad = [ack for ack in acks if ack.get("type") not in {"subscribed", "ok"}]
        if bad:
            print(f"subscribe failed for group {list(unique)}: {bad}")
            return False

        sids = set()
        for ack in acks:
            sid = ack.get("sid")
            if sid is None:
                sid = ack.get("msg", {}).get("sid")
            if sid is not None:
                sids.add(sid)

        if len(sids) < len(CHANNELS):
            print(f"subscribe failed for group {list(unique)}: missing sids")
            print(f"acks: {acks}")
            return False

        snapshots = await self.client.wait_for_snapshots(list(unique), timeout=timeout)

        invalid = []
        for ticker in unique:
            snapshot = snapshots.get(ticker)
            if snapshot is None:
                invalid.append((ticker, "no snapshot"))
                continue

            msg = snapshot.get("msg", {}) or {}
            snapshot_ticker = msg.get("market_ticker")
            market_id = msg.get("market_id")

            if snapshot_ticker != ticker:
                invalid.append(
                    (ticker, f"snapshot_ticker mismatch: got {snapshot_ticker!r}")
                )
                continue

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

        group = SubscriptionGroup(tickers=group_key, sids=sids)
        self.groups[group_key] = group
        for ticker in group_key:
            self.ticker_to_group[ticker] = group_key

        print(
            f"subscribed group [{', '.join(sorted(group_key))}] "
            f"(sids={sorted(sids)})"
        )
        return True

    async def unsubscribe_group_exact(
        self,
        tickers: List[str],
        timeout: float = ACK_TIMEOUT,
    ) -> bool:
        requested = frozenset(set(tickers))

        if not requested:
            print("cannot unsubdelete empty group")
            return False

        group = self.groups.get(requested)
        if group is None:
            matching_groups = []
            for existing_group in self.groups:
                if requested.issubset(existing_group):
                    matching_groups.append(existing_group)

            if matching_groups:
                print("cannot unsubdelete only part of a subwrite group")
                print("you must unsubdelete the exact same tickers that were subwritten together")
                for g in matching_groups:
                    print(f"  full group: {', '.join(sorted(g))}")
            else:
                print(
                    "no exact subscribed subwrite group found for: "
                    f"{', '.join(sorted(requested))}"
                )
            return False

        sids = sorted(group.sids)

        req_id = await self._safe_unsubscribe_call(sids)
        if req_id is None:
            print(
                f"unsubscribe failed immediately for group: "
                f"{', '.join(sorted(requested))}"
            )
            return False

        acks = await wait_for_command_acks(
            self.client,
            req_id,
            expected=len(sids),
            timeout=timeout,
        )

        if len(acks) < len(sids):
            print(f"unsubscribe timed out for group: {', '.join(sorted(requested))}")
            print(f"partial acks: {acks}")
            return False

        bad = [ack for ack in acks if ack.get("type") not in {"unsubscribed", "ok"}]
        if bad:
            print(f"unsubscribe failed for group {sorted(requested)}: {bad}")
            return False

        del self.groups[requested]
        for ticker in requested:
            self.ticker_to_group.pop(ticker, None)

        print(f"unsubscribed group [{', '.join(sorted(requested))}]")
        return True

    def print_status(self) -> None:
        print("subwrite groups:")
        if not self.groups:
            print("  (none)")
            return

        for group_key, group in self.groups.items():
            print(
                f"  group=[{', '.join(sorted(group_key))}] "
                f"sids={sorted(group.sids)}"
            )


async def do_subwrite(
    subs: SubscriptionManager,
    handler: QueueHandler,
    tickers: List[str],
) -> None:
    unique = sorted(set(tickers))
    if not unique:
        print("usage: subwrite TICKER [TICKER ...]")
        return

    enabled_now: List[str] = []

    try:
        # Enable writing first so the initial orderbook_snapshot is not dropped.
        for ticker in unique:
            await handler.add_id(ticker)
            enabled_now.append(ticker)

        success = await subs.subscribe_group(unique)
        if not success:
            for ticker in enabled_now:
                await handler.remove_id(ticker)
            print("subwrite failed; writing not enabled")
            return

        print(f"subwritten together: {', '.join(unique)}")

    except Exception as e:
        for ticker in enabled_now:
            try:
                await handler.remove_id(ticker)
            except Exception as cleanup_error:
                print(f"cleanup failed for {ticker}: {cleanup_error}")
        print(f"subwrite failed with unexpected error: {e}")


async def do_unsubdelete(
    subs: SubscriptionManager,
    handler: QueueHandler,
    tickers: List[str],
) -> None:
    unique = sorted(set(tickers))

    success = await subs.unsubscribe_group_exact(unique)
    if not success:
        print("unsubdelete rejected; writing remains enabled")
        return

    for ticker in unique:
        await handler.remove_id(ticker)

    print(f"writing disabled for group: {', '.join(unique)}")


def print_help() -> None:
    print("Commands:")
    print("  write TICKER [TICKER ...]")
    print("  delete TICKER [TICKER ...]")
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
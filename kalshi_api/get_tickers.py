import requests
import json
from typing import Dict, Optional, Any, List

from kalshi_api.utils import BASE_URL

LIMIT = 100


def get_series_list(
    *,
    status: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    include_product_metadata: bool = False,
    include_volume: bool = False
) -> List[Dict[str, Any]]:
    cursor = None
    series_list = []

    params = {"limit": LIMIT}
    if status:
        params['status'] = status
    if category:
        params['category'] = category
    if tags:
        params['tags'] = tags
    if include_product_metadata:
        params['include_product_metadata'] = include_product_metadata
    if include_volume:
        params['include_volume'] = include_volume
    if cursor:
        params["cursor"] = cursor

    while True:
        r = requests.get(f"{BASE_URL}/series", params=params)
        r.raise_for_status()
        data = r.json()

        series_batch = data.get("series", [])

        for series in series_batch:
            series_list.append(series)

        cursor = data.get("cursor")
        if not cursor or not series_batch:
            break

    return series_list


def get_events_list(
    series_ticker: str,
    *,
    status: Optional[str] = None,
    with_nested_markets: bool = False,
    with_milestones: bool = False,
    min_close_ts: Optional[int] = None,
    min_update_ts: Optional[int] = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cursor = None
    events_list = []
    milestones_list = []

    params = {"limit": LIMIT, "series_ticker": series_ticker}
    if status:
        params['status'] = status
    if with_nested_markets:
        params['with_nested_markets'] = with_nested_markets
    if with_milestones:
        params['with_milestones'] = with_milestones
    if min_close_ts:
        params['min_close_ts'] = min_close_ts
    if min_update_ts:
        params['min_update_ts'] = min_update_ts
    if cursor:
        params["cursor"] = cursor

    while True:
        r = requests.get(f"{BASE_URL}/events", params=params)
        r.raise_for_status()
        data = r.json()

        events_batch = data.get("events", [])
        milestones_batch = data.get("milestones", [])

        for event in events_batch:
            events_list.append(event)

        for milestone in milestones_batch:
            milestones_list.append(milestone)

        cursor = data.get("cursor")
        if not cursor or not events_batch:
            break

    return events_list, milestones_list


def get_markets_list(
    series_ticker: str,
    event_tickers: Optional[List[str]] = None,
    *,
    status: Optional[str] = None,
    min_created_ts: Optional[int] = None,
    max_created_ts: Optional[int] = None,
    min_updated_ts: Optional[int] = None,
    min_close_ts: Optional[int] = None,
    max_close_ts: Optional[int] = None,
    min_settled_ts: Optional[int] = None
) -> List[Dict[str, Any]]:
    cursor = None
    markets_list = []

    params = {"limit": LIMIT, "series_ticker": series_ticker}
    if event_tickers:
        params['event_ticker'] = event_tickers
    if status:
        params['status'] = status
    if min_created_ts:
        params['min_created_ts'] = min_created_ts
    if max_created_ts:
        params['max_created_ts'] = max_created_ts
    if min_updated_ts:
        params['min_updated_ts'] = min_updated_ts
    if min_close_ts:
        params['min_close_ts'] = min_close_ts
    if max_close_ts:
        params['max_close_ts'] = max_close_ts
    if min_settled_ts:
        params['min_settled_ts'] = min_settled_ts
    if cursor:
        params["cursor"] = cursor

    while True:
        r = requests.get(f"{BASE_URL}/markets", params=params)
        r.raise_for_status()
        data = r.json()

        markets_batch = data.get("markets", [])

        for market in markets_batch:
            markets_list.append(market)

        cursor = data.get("cursor")
        if not cursor or not markets_batch:
            break

    return markets_list
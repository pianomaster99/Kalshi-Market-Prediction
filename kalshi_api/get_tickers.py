import requests
import json
from typing import Dict, Optional, Any, List, Tuple
from kalshi_api.RequestLimiter import RestRequest, RequestLimiter

from kalshi_api.utils import BASE_URL

LIMIT = 100

async def get_series_list(
    limiter: RequestLimiter,
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
        if cursor:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        r = await limiter.request(
            RestRequest(
                method="GET",
                path="/series",
                params=params,
            )
        )
        r.raise_for_status()
        data = r.json()

        series_batch = data.get("series") or []
        series_list.extend(series_batch)

        cursor = data.get("cursor")
        if not cursor or not series_batch:
            break

    return series_list

async def get_events_list(
    limiter: RequestLimiter,
    series_ticker: str,
    *,
    status: Optional[str] = None,
    with_nested_markets: bool = False,
    with_milestones: bool = False,
    min_close_ts: Optional[int] = None,
    min_update_ts: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cursor: Optional[str] = None
    events_list: List[Dict[str, Any]] = []
    milestones_list: List[Dict[str, Any]] = []

    params: Dict[str, Any] = {"limit": LIMIT, "series_ticker": series_ticker}
    if status:
        params["status"] = status
    if with_nested_markets:
        params["with_nested_markets"] = True
    if with_milestones:
        params["with_milestones"] = True
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    if min_update_ts is not None:
        params["min_update_ts"] = min_update_ts

    while True:
        if cursor:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        r = await limiter.request(
            RestRequest(method="GET", path="/events", params=params)
        )
        r.raise_for_status()
        data = r.json() or {}

        events_batch = data.get("events") or []
        milestones_batch = data.get("milestones") or []

        events_list.extend(events_batch)
        milestones_list.extend(milestones_batch)

        cursor = data.get("cursor")
        if not cursor or not events_batch:
            break

    return events_list, milestones_list

async def get_markets_list(
    limiter: RequestLimiter,
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
    cursor: Optional[str] = None
    markets_list: List[Dict[str, Any]] = []

    params: Dict[str, Any] = {"limit": LIMIT, "series_ticker": series_ticker}
    if event_tickers:
        params["event_ticker"] = event_tickers
    if status:
        params["status"] = status
    if min_created_ts is not None:
        params["min_created_ts"] = min_created_ts
    if max_created_ts is not None:
        params["max_created_ts"] = max_created_ts
    if min_updated_ts is not None:
        params["min_updated_ts"] = min_updated_ts
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    if max_close_ts is not None:
        params["max_close_ts"] = max_close_ts
    if min_settled_ts is not None:
        params["min_settled_ts"] = min_settled_ts

    while True:
        if cursor:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        r = await limiter.request(
            RestRequest(method="GET", path="/markets", params=params)
        )
        r.raise_for_status()
        data = r.json() or {}

        markets_batch = data.get("markets") or []
        markets_list.extend(markets_batch)

        cursor = data.get("cursor")
        if not cursor or not markets_batch:
            break

    return markets_list
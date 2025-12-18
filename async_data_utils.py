# async_data_utils.py
import asyncio
import httpx
import pandas as pd
from datetime import datetime
from db import engine, candles
from sqlalchemy import insert
import time

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

async def fetch_single_candle_async(client, symbol, interval, limit=100):
    """Fetch candles for one symbol asynchronously."""
    url = f"{BINANCE_API_URL}?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        
        # Numeric conversion
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms")
        
        return symbol, df
    except Exception as e:
        print(f"❌ Async fetch failed for {symbol}: {e}")
        return symbol, None

async def batch_fetch_candles(symbols: list, interval="1m", limit=100):
    """
    Fetch candles for ALL symbols at the same time (Parallel).
    """
    if not symbols:
        return {}
        
    async with httpx.AsyncClient() as client:
        tasks = [fetch_single_candle_async(client, sym, interval, limit) for sym in symbols]
        results = await asyncio.gather(*tasks)
    
    # Filter out failed fetches (None)
    return {sym: df for sym, df in results if df is not None and not df.empty}

async def bulk_save_to_db(data_map, interval):
    """
    Prepare all rows from all symbols and insert in ONE database transaction.
    """
    all_records = []
    for symbol, df in data_map.items():
        # Only take the last few candles to avoid duplicates if needed, or insert all
        # For efficiency, maybe just the last closed candle? 
        # For now, let's insert the batch (ensure your DB handles duplicates or use distinct)
        for _, row in df.iterrows():
            all_records.append({
                "symbol": symbol.lower(),
                "interval": interval,
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            })
    
    if all_records:
        # Run synchronous DB write in a thread to not block the async loop
        await asyncio.to_thread(_execute_bulk_insert, all_records)
        print(f"💾 Bulk saved {len(all_records)} candles to DB.")

def _execute_bulk_insert(records):
    """Helper for threading."""
    try:
        with engine.begin() as conn:
            conn.execute(insert(candles), records)
    except Exception as e:
        print(f"⚠️ Bulk insert error (ignoring duplicates): {e}")
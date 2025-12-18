# fastapi_backend_v2.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List
import data_fetcher
from fastapi import Body
from pydantic import BaseModel
import websocket_fetcher
from paper_trading_engine import ai_trade, portfolio, latest_price
from paper_trading_engine import PaperTradingEngineSingleton
pte = PaperTradingEngineSingleton.get_instance()
from binance_utils import filter_valid_binance_symbols
import asyncio
import threading
import time
from datetime import datetime
import trade_history


# Add these imports at the top
from scheduler_service import activate_auto_trading, stop_auto_trading, TRADING_ACTIVE

from scheduler_service import initialize_scheduler  # <--- Ensure this is imported
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🟢 STARTUP LOGIC
    # This runs when the app starts
    initialize_scheduler()
    print("✅ Scheduler attached to FastAPI event loop.")
    
    yield  # The app runs while yielding
    
    # 🔴 SHUTDOWN LOGIC (Optional)
    # This runs when the app stops
    print("🛑 Shutting down scheduler...")
    # scheduler.shutdown() # Uncomment if you want clean shutdowns later
# ============================================================
# 🔹 PYDANTIC MODELS
# ============================================================

class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    symbol: str
    days: int = 180
    starting_balance: float = 10000


async def process_live_signals(symbols: List[str]):
    """
    Process live trading signals for multiple symbols:
    1. Verify valid Binance pairs
    2. Fetch OHLCV data
    3. Calculate indicators
    4. Get AI predictions
    5. Start paper trades for BUY signals
    """
    try:
        # Filter for valid Binance symbols
        valid_symbols = filter_valid_binance_symbols(symbols)
        if not valid_symbols:
            return {"ok": False, "error": "No valid Binance symbols found"}

        results = []
        for symbol in valid_symbols:
            try:
                # Add to WebSocket feed first
                websocket_fetcher.add_symbol(symbol)

                # Wait briefly for price data
                attempts = 5
                while attempts > 0 and symbol not in websocket_fetcher.latest_price:
                    await asyncio.sleep(1)
                    attempts -= 1
                    print(f"⏳ Waiting for live price update for {symbol}...")

                # Fetch and store OHLCV data
                df = data_fetcher.get_recent_candles(symbol, interval="1m", limit=1000)
                if df is None or df.empty:
                    print(f"⚠️ No data available for {symbol}")
                    continue

                # Run AI analysis
                parsed = await pte.ai_trade(symbol, df)
                
                # Handle different response formats
                if isinstance(parsed, str):
                    result = {
                        "symbol": symbol,
                        "decision": "HOLD",
                        "confidence": 0,
                        "reason": parsed
                    }
                elif isinstance(parsed, dict):
                    # Try to get decision info from different possible structures
                    if "summary" in parsed and isinstance(parsed["summary"], dict):
                        parsed_data = parsed["summary"].get("parsed", {})
                    else:
                        parsed_data = parsed
                    
                    # Extract decision and confidence safely
                    decision = str(parsed_data.get("decision", "HOLD")).upper()
                    try:
                        confidence = float(parsed_data.get("confidence", 0) or 0)
                    except (ValueError, TypeError):
                        confidence = 0
                    
                    result = {
                        "symbol": symbol,
                        "decision": decision,
                        "confidence": confidence,
                        "reason": str(parsed_data.get("reason", "No reason provided"))
                    }
                else:
                    result = {
                        "symbol": symbol,
                        "decision": "HOLD",
                        "confidence": 0,
                        "reason": "Invalid AI response format"
                    }

                # If AI suggests BUY with good confidence, start paper trade
                if result["decision"] == "BUY" and result["confidence"] >= 70:
                    # Check if trade already exists
                    print(f"🔍 Checking for existing trade for {symbol}...")
                    existing_trade = trade_history.get_open_trade(symbol)
                    
                    if not existing_trade:
                        print(f"✅ No existing trade found for {symbol}, attempting to start trade...")
                        trade_result = pte.start_trade_for_symbol(symbol)
                        print(f"🎯 Trade start result for {symbol}: {trade_result}")
                        
                        if isinstance(trade_result, dict):
                            result["trade_started"] = trade_result.get("ok", False)
                            if not trade_result.get("ok"):
                                result["reason"] = trade_result.get("error", "Unknown error starting trade")
                        else:
                            result["trade_started"] = False
                            result["reason"] = "Invalid trade result format"
                            
                        if result.get("trade_started"):
                            print(f"🚀 Successfully auto-started paper trade for {symbol}")
                        else:
                            print(f"❌ Failed to start paper trade for {symbol}: {result.get('reason')}")
                    else:
                        result["trade_started"] = False
                        result["reason"] = f"Trade already open for {symbol}"
                        print(f"⚠️ {result['reason']}")

                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        return {
            "ok": True,
            "results": results,
            "processed_count": len(results)
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

app = FastAPI(lifespan=lifespan)
# app = FastAPI()

# Allow CORS for your HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "*"  # Allow all origins for Railway deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # ✅ ADD THIS STARTUP EVENT
# @app.on_event("startup")
# async def start_scheduler_event():
#     """Start the scheduler when FastAPI boots up."""
#     initialize_scheduler()
#     print("✅ Scheduler attached to FastAPI event loop.")

# Add these new endpoints
@app.post("/start_auto_mode")
async def start_mode():
    return await activate_auto_trading()

@app.post("/stop_auto_mode")
def stop_mode():
    return stop_auto_trading()

@app.get("/status")
def get_status():
    return {"auto_active": TRADING_ACTIVE}







@app.get("/prices")
def get_prices():
    return websocket_fetcher.latest_price

@app.get("/ai_decision")
def get_all_ai_decisions():
    """Return all AI decisions in clean format (for frontend table)."""
    decisions = getattr(pte, "last_ai_decisions", None)
    if not decisions:
        return {"error": "No AI decisions found yet."}

    result = {}
    for sym, sym_data in decisions.items():
        try:
            # Handle string responses
            if isinstance(sym_data, str):
                result[sym] = {
                    "decision": "HOLD",
                    "confidence": 0,
                    "entry": 0,
                    "target": 0,
                    "stop_loss": 0,
                    "reason": sym_data,
                    "timestamp": "N/A"
                }
                continue

            # Handle dict responses
            if not isinstance(sym_data, dict):
                continue

            # Try to get summary, might be nested or direct
            summary = sym_data.get("summary", sym_data)
            if isinstance(summary, str):
                summary = {"parsed": {"reason": summary}}
            elif not isinstance(summary, dict):
                continue

            # Get parsed data, might be nested or direct
            parsed = summary.get("parsed", summary)
            if not isinstance(parsed, dict):
                parsed = {"reason": str(parsed)}

            result[sym] = {
                "decision": str(parsed.get("decision", "HOLD")).upper(),
                "confidence": float(parsed.get("confidence", 0) or 0),
                "entry": float(parsed.get("entry", 0) or 0),
                "target": float(parsed.get("target", 0) or 0),
                "stop_loss": float(parsed.get("stop_loss", 0) or 0),
                "reason": str(parsed.get("reason", "No reason provided.")),
                "timestamp": str(summary.get("timestamp", "N/A"))
            }
        except Exception as e:
            print(f"⚠️ Error processing AI decision for {sym}: {e}")
            result[sym] = {
                "decision": "HOLD",
                "confidence": 0,
                "entry": 0,
                "target": 0,
                "stop_loss": 0,
                "reason": f"Error processing AI data: {str(e)}",
                "timestamp": "N/A"
            }

    if not result:
        return {"error": "AI decisions structure found, but empty."}

    return result

@app.get("/trades")
def get_trades():
    """Return open trades only from the database."""
    from sqlalchemy import select
    from db import engine, trades

    out = []
    try:
        with engine.connect() as conn:
            # Fetch only trades with status='OPEN' AND closed_at IS NULL
            stmt = (
                select(trades)
                .where(
                    (trades.c.status == "OPEN") &
                    (trades.c.closed_at.is_(None))
                )
                .order_by(trades.c.id.desc())
            )
            rows = conn.execute(stmt).mappings().all()

        print(f"🔍 DEBUG /trades: Found {len(rows)} open trades")

        seen_symbols = set()
        for r in rows:
            symbol = r.get("symbol", "").upper()
            
            # Skip duplicates
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)

            entry = float(r.get("entry") or 0)
            size = float(r.get("size") or 0)
            amount = entry * size if entry and size else 0  # Removed * 10 multiplier

            print(f"   Trade: {symbol} | Entry: {entry} | Target: {r.get('target')} | Stop Loss: {r.get('stop_loss')} | Status: {r.get('status')} | closed_at: {r.get('closed_at')}")

            out.append({
                "symbol": symbol,
                "side": r.get("side", "BUY").upper(),
                "entry": entry,
                "target": float(r.get("target") or 0),
                "stop_loss": float(r.get("stop_loss") or 0),
                "amount": amount,
                "price": None,
                "pnl": None,
                "pnl_pct": None,
                "status": "OPEN",
                "opened_at": str(r.get("opened_at") or ""),
                "closed_at": None
            })

    except Exception as e:
        print(f"⚠️ Error in /trades endpoint: {e}")
        import traceback
        traceback.print_exc()

    print(f"✅ Returning {len(out)} trades to frontend")
    return out

@app.get("/balance")
def get_balance():
    return {"balance": pte.portfolio.get("balance",0)}

@app.get("/ai_decisions")
def get_ai_decisions():
    """Return AI decisions in plain, human-readable English sentences."""
    decisions = getattr(pte, "last_ai_decisions", {}) or {}
    if not decisions:
        return "No AI decisions available yet."

    lines = ["AI Market Decisions:\n"]

    for symbol, sym_data in decisions.items():
        try:
            # Handle string responses
            if isinstance(sym_data, str):
                lines.append(f"• {symbol}: {sym_data}")
                continue

            # Handle dict responses
            if not isinstance(sym_data, dict):
                lines.append(f"• {symbol}: Invalid data type")
                continue

            # Try to get summary, might be nested or direct
            summary = sym_data.get("summary", sym_data)
            if isinstance(summary, str):
                lines.append(f"• {symbol}: {summary}")
                continue

            # Get parsed data, might be nested or direct
            parsed = summary.get("parsed", summary) if isinstance(summary, dict) else summary
            if not isinstance(parsed, dict):
                lines.append(f"• {symbol}: {str(parsed)}")
                continue

            # Extract decision info
            decision = str(parsed.get("decision", "N/A")).upper()
            reason = str(parsed.get("reason", "No reason provided."))
            confidence = parsed.get("confidence")
            timestamp = summary.get("timestamp", "N/A") if isinstance(summary, dict) else "N/A"

            line = f"• {symbol}: {decision} — {reason}"
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    if conf_val > 0:
                        line += f" (Confidence: {conf_val:.1f}%)"
                except (ValueError, TypeError):
                    pass

            if timestamp and timestamp != "N/A":
                line += f" [Updated: {timestamp}]"

            lines.append(line)
        except Exception as e:
            lines.append(f"• {symbol}: Unable to parse AI result ({e}).")

    return "\n".join(lines)

@app.get("/ai_decision/{symbol}")
def get_ai_for_symbol(symbol: str):
    sym = symbol.strip().upper()
    decisions = getattr(pte, "last_ai_decisions", {}) or {}

    if sym not in decisions:
        return {"error": f"No AI decision found for {sym}."}

    sym_data = decisions.get(sym, {})
    try:
        # Handle string responses
        if isinstance(sym_data, str):
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reason": sym_data,
                "timestamp": "N/A"
            }

        # Handle dict responses
        if not isinstance(sym_data, dict):
            return {"error": f"{sym}: Invalid data type"}

        # Try to get summary, might be nested or direct
        summary = sym_data.get("summary", sym_data)
        if isinstance(summary, str):
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reason": summary,
                "timestamp": "N/A"
            }

        # Get parsed data, might be nested or direct
        parsed = summary.get("parsed", summary) if isinstance(summary, dict) else summary
        if not isinstance(parsed, dict):
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reason": str(parsed),
                "timestamp": "N/A"
            }

        # Return normalized response
        return {
            "decision": str(parsed.get("decision", "N/A")).upper(),
            "confidence": float(parsed.get("confidence", 0) or 0),
            "reason": str(parsed.get("reason", "No reason provided.")),
            "timestamp": str(summary.get("timestamp", "N/A") if isinstance(summary, dict) else "N/A")
        }
    except Exception as e:
        return {"error": f"{sym}: Unable to parse AI result ({e})."}

@app.post("/subscribe_symbol")
def subscribe_symbol(payload: dict = Body(...)):
    sym = str(payload.get("symbol", "")).strip()
    if not sym:
        return {"ok": False, "error": "symbol required"}
    websocket_fetcher.add_symbol(sym)
    return {"ok": True, "tracked": websocket_fetcher.get_tracked_symbols()}

@app.post("/start_paper_trade")
async def start_paper_trade(payload: dict = Body(...)):
    """
    When user clicks 'Analyze':
    - Fetches recent data
    - Runs AI model (ai_trade)
    - Returns AI decision but does NOT execute trade automatically
    """
    sym = str(payload.get("symbol", "")).strip()
    if not sym:
        return {"ok": False, "error": "symbol required"}

    websocket_fetcher.add_symbol(sym)

    try:
        # Wait briefly for price data
        attempts = 5
        while attempts > 0 and sym.lower() not in websocket_fetcher.latest_price:
            await asyncio.sleep(1)
            attempts -= 1
            print(f"⏳ Waiting for live price update for {sym}...")

        # Fetch latest 1m candles
        df = data_fetcher.get_recent_candles(sym, interval="1m", limit=1000)
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "No candle data available"}

        # Run AI model asynchronously
        parsed = await pte.ai_trade(sym, df)

        # Add to monitoring (no trade yet)
        try:
            pte.start_paper_trading_for(sym)
        except Exception as sub_e:
            print(f"⚠️ start_paper_trading_for failed for {sym}: {sub_e}")

        return {
            "ok": True,
            "symbol": sym.upper(),
            "decision": parsed,
            "message": "AI analysis complete. Ready for manual trade start."
        }

    except Exception as e:
        return {"ok": False, "error": f"start_paper_trade failed: {e}"}

# ============================================================
# 🔹 CONSISTENT GAINERS BACKGROUND TASK
# ============================================================

# Global state for background task
_consistent_gainers_task_running = False
_last_fetch_time = None

# API endpoint for consistent gainers
CONSISTENT_GAINERS_API = "http://57.128.242.183:9100/api/consistent-gainers-all"

async def fetch_consistent_gainer_symbols(timeframe='15m'):
    """
    Fetch coin symbols from the consistent gainers API.
    Returns only the symbols list - processing is done by existing workflow.
    """
    try:
        print(f"🔄 Fetching consistent gainer symbols from API for {timeframe}...")
        
        # Fetch from your consistent gainers API
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Call the API - the deployed endpoint doesn't require timeframe in URL
        url = CONSISTENT_GAINERS_API
        print(f"📡 Calling API: {url}")
        
        response = session.get(url, timeout=(15, 30))
        response.raise_for_status()
        data = response.json()
        
        print(f"📥 API Response received: {len(str(data))} bytes")
        
        # Extract symbols from the response
        symbols = []
        
        # Handle different response formats
        if isinstance(data, dict):
            # Check for 'data' field (array of coin objects)
            if 'data' in data and isinstance(data['data'], list):
                for coin in data['data']:
                    if isinstance(coin, dict) and 'symbol' in coin:
                        symbols.append(coin['symbol'])
                    elif isinstance(coin, str):
                        symbols.append(coin)
            # Check for 'symbols' field (array of strings)
            elif 'symbols' in data and isinstance(data['symbols'], list):
                for item in data['symbols']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(item['symbol'])
                    elif isinstance(item, str):
                        symbols.append(item)
            # Check for 'gainers' field
            elif 'gainers' in data and isinstance(data['gainers'], list):
                for item in data['gainers']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(item['symbol'])
                    elif isinstance(item, str):
                        symbols.append(item)
            # Check for direct list in response
            else:
                for item in data.values():
                    if isinstance(item, list):
                        for coin in item:
                            if isinstance(coin, dict) and 'symbol' in coin:
                                symbols.append(coin['symbol'])
                            elif isinstance(coin, str):
                                symbols.append(coin)
                        break
        elif isinstance(data, list):
            # Direct array of symbols or coin objects
            for item in data:
                if isinstance(item, dict) and 'symbol' in item:
                    symbols.append(item['symbol'])
                elif isinstance(item, str):
                    symbols.append(item)
        
        # Normalize symbols: ensure they are uppercase and end with USDT
        normalized_symbols = []
        for s in symbols:
            if not isinstance(s, str):
                continue
            s = s.upper().strip()
            # If it's a raw symbol like 'BTC', append 'USDT'
            if not s.endswith('USDT'):
                s += 'USDT'
            normalized_symbols.append(s)
        
        symbols = normalized_symbols
        
        print(f"🎯 Found {len(symbols)} consistent gainer symbols from API")
        if symbols:
            print(f"📋 Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        return symbols
        
    except requests.exceptions.Timeout:
        print(f"⏱️ API request timed out")
        return []
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection error to API: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching from API: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return []

async def consistent_gainers_background_task(timeframe='15m'):
    """Background task that runs every 2 hours"""
    global _last_fetch_time, _consistent_gainers_task_running
    
    print(f"🚀 Starting consistent gainers background task for {timeframe}")
    
    while _consistent_gainers_task_running:
        try:
            # Fetch consistent gainer symbols
            symbols = await fetch_consistent_gainer_symbols(timeframe)
            _last_fetch_time = datetime.now()
            
            print(f"📊 Updated consistent gainers: {len(symbols)} symbols at {_last_fetch_time}")
            
            # Process symbols using existing workflow
            if symbols:
                # Take top 20 symbols
                top_symbols = symbols[:20]
                print(f"🔄 Processing top {len(top_symbols)} gainers through existing workflow...")
                print(f"Symbols: {', '.join(top_symbols)}")
                
                # Use existing process_live_signals which handles:
                # 1. Binance validation
                # 2. Multi-timeframe data fetching
                # 3. DB storage
                # 4. Indicator calculation
                # 5. AI model prediction
                # 6. Paper trade execution
                result = await process_live_signals(top_symbols)
                print(f"✅ Processing complete: {result.get('processed_count', 0)} symbols processed")
            
            # Wait 2 hours before next fetch
            print("⏰ Waiting 2 hours before next fetch...")
            await asyncio.sleep(7200)  # 2 hours
            
        except Exception as e:
            print(f"❌ Error in background task: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(300)  # Wait 5 minutes on error

@app.get("/api/consistent-gainers/{timeframe}")
async def get_consistent_gainers(timeframe: str, background_tasks: BackgroundTasks):
    """
    Get coins with consistent gains and process through full workflow:
    1. Fetch symbols from deployed consistent gainers API
    2. Validate with Binance
    3. Fetch multi-timeframe data
    4. Store in DB
    5. Calculate indicators
    6. Run AI model
    7. Execute paper trades if BUY signal
    """
    global _consistent_gainers_task_running, _last_fetch_time
    
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if timeframe not in valid_timeframes:
            return {
                "ok": False,
                "error": f"Invalid timeframe. Valid options: {', '.join(valid_timeframes)}"
            }
        
        # Start background task if not running
        if not _consistent_gainers_task_running:
            _consistent_gainers_task_running = True
            background_tasks.add_task(consistent_gainers_background_task, timeframe)
            print(f"✅ Started consistent gainers background task for {timeframe}")
        
        # Fetch symbols immediately
        print(f"🔍 Fetching consistent gainers for immediate processing...")
        symbols = await fetch_consistent_gainer_symbols(timeframe)
        _last_fetch_time = datetime.now()
        
        if not symbols:
            return {
                "ok": False,
                "error": "No consistent gainers found from external API",
                "timeframe": timeframe,
                "last_updated": _last_fetch_time.isoformat() if _last_fetch_time else None
            }
        
        # Take top 20 symbols
        top_symbols = symbols[:20]
        print(f"🔄 Processing {len(top_symbols)} symbols through full workflow...")
        
        # Process through existing workflow
        result = await process_live_signals(top_symbols)
        
        return {
            "ok": True,
            "timeframe": timeframe,
            "symbols_found": len(symbols),
            "symbols_processed": len(top_symbols),
            "last_updated": _last_fetch_time.isoformat() if _last_fetch_time else None,
            "processing_result": result,
            "symbols": top_symbols
        }
        
    except Exception as e:
        print(f"❌ Error in get_consistent_gainers: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "timeframe": timeframe
        }


@app.post('/analyze_all')
async def analyze_all_symbols(payload: dict = Body(...)):
    """
    Analyze multiple symbols concurrently.
    For each symbol:
      - Fetches recent candles
      - Runs AI (via ai_trade)
      - Automatically starts trade if AI says BUY
      - Skips if trade is already open
    """
    import random
    from data_fetcher import get_recent_candles

    try:
        symbols = payload.get("symbols", [])
        if not isinstance(symbols, list) or not symbols:
            return {"ok": False, "error": "symbols list required"}

        results = {}

        # Single-symbol worker
        async def analyze_one(sym):
            sym = sym.upper()
            print(f"🔹 Starting AI for {sym}")
            
            # Add to WebSocket feed first
            websocket_fetcher.add_symbol(sym)

            # Wait briefly for price data
            attempts = 5
            while attempts > 0 and sym.lower() not in websocket_fetcher.latest_price:
                await asyncio.sleep(1)
                attempts -= 1
                print(f"⏳ Waiting for live price update for {sym}...")

            df = get_recent_candles(sym, interval="1m", limit=1000)
            if df is None or getattr(df, "empty", True):
                print(f"⚠️ No data for {sym}")
                return {"decision": "HOLD", "reason": "no_data"}

            parsed = await pte.ai_trade(sym, df)
            decision = parsed.get("decision", "").upper()

            # Auto-trade logic only for BUY signals
            if decision == "BUY":
                open_trade = trade_history.get_open_trade(sym)
                if open_trade and open_trade.get("closed_at") is None:
                    print(f"⚠️ Skipping {sym} — trade already open.")
                    parsed["trade_started"] = False
                    parsed["reason"] = "already_open"
                else:
                    trade_result = pte.start_trade_for_symbol(sym)
                    parsed["trade_started"] = trade_result.get("ok", False)
                    print(f"🚀 Auto trade started for {sym}")
            else:
                parsed["trade_started"] = False

            return parsed

        # Limit concurrency to 5 and add staggered delays
        sem = asyncio.Semaphore(5)

        async def limited_analyze(sym):
            async with sem:
                await asyncio.sleep(random.uniform(0.5, 1.5))  # small delay to avoid overload
                return sym.upper(), await analyze_one(sym)

        # Run all in parallel
        results_list = await asyncio.gather(*[limited_analyze(sym) for sym in symbols], return_exceptions=True)

        # Collect results
        for res in results_list:
            if isinstance(res, Exception):
                print(f"⚠️ Exception in analyze_all: {res}")
                continue
            sym, parsed = res
            results[sym] = parsed

        print(f"✅ Finished analyzing {len(symbols)} symbols concurrently.")
        return {"ok": True, "results": results}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"ok": False, "error": str(e)}

@app.post("/start_trade")
def start_trade(payload: dict = Body(...)):
    """
    When user clicks 'Start Trade':
    - Uses last AI decision (BUY/SELL)
    - Starts paper trade if AI recommended BUY
    """
    sym = str(payload.get("symbol", "")).strip()
    if not sym:
        return {"ok": False, "error": "symbol required"}

    websocket_fetcher.add_symbol(sym)

    # Fetch the last AI decision stored in memory
    ai_decision = pte.last_ai_decisions.get(sym.upper(), {}).get("summary", {}).get("parsed", {})
    if not ai_decision:
        return {"ok": False, "error": "No AI decision found for this symbol. Run analysis first."}

    decision = ai_decision.get("decision", "HOLD").upper()

    if decision == "BUY":
        result = pte.start_trade_for_symbol(sym)
        if isinstance(result, dict) and result.get("ok"):
            try:
                pte.start_paper_trading_for(sym)
            except Exception as e:
                print(f"⚠️ start_paper_trading_for failed: {e}")
            return {"ok": True, "trade_started": True, "ai_decision": ai_decision}
        else:
            return {"ok": False, "error": "Failed to start trade"}

    return {"ok": False, "error": f"Trade not started because AI decision = {decision}"}

@app.get('/trade_history')
def get_trade_history(limit: int = None):
    try:
        items = trade_history.get_history(limit=limit)
        return {'ok': True, 'count': len(items), 'history': items}
    except Exception as e:
        return {'ok': False, 'error': str(e)}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# # ============================================================
# # 🔹 BACKTESTING ENDPOINTS
# # ============================================================

# @app.post('/backtest/run')
# async def run_backtest(request: BacktestRequest):
#     """
#     Run backtesting on historical data for a specific symbol.
    
#     Args:
#         request: BacktestRequest with symbol, days, starting_balance
    
#     Returns:
#         Backtest results with P&L metrics and trades
#     """
#     try:
#         symbol = request.symbol
#         days = request.days
#         starting_balance = request.starting_balance
        
#         # Validate inputs
#         if not symbol or not isinstance(symbol, str):
#             return {"ok": False, "error": "Invalid symbol"}
        
#         symbol = symbol.upper()
        
#         if days < 1 or days > 365:
#             return {"ok": False, "error": "Days must be between 1 and 365"}
        
#         if starting_balance < 100:
#             return {"ok": False, "error": "Starting balance must be at least $100"}
        
#         print(f"🚀 Starting backtest for {symbol} ({days} days, ${starting_balance:,.2f})")
        
# #         # Import backtesting engine
# #         from backtesting_engine import run_backtest_for_symbol
        
# #         # Run the backtest
# #         result = run_backtest_for_symbol(
# #             symbol=symbol,
# #             days=days,
# #             starting_balance=starting_balance
# #         )
        
# #         if result and result.get("status") == "✅ OK":
# #             return {
# #                 "ok": True,
# #                 "symbol": symbol,
# #                 "days": days,
# #                 "starting_balance": starting_balance,
# #                 "final_balance": result.get("final_balance", 0),
# #                 "total_pnl": result.get("metrics", {}).get("total_pnl", 0),
# #                 "pnl_percent": result.get("metrics", {}).get("total_pnl_percent", 0),
# #                 "total_trades": result.get("metrics", {}).get("total_trades", 0),
# #                 "winning_trades": result.get("metrics", {}).get("winning_trades", 0),
# #                 "losing_trades": result.get("metrics", {}).get("losing_trades", 0),
# #                 "win_rate": result.get("metrics", {}).get("win_rate_percent", 0),
# #                 "profit_factor": result.get("metrics", {}).get("profit_factor", 0),
# #                 "avg_win": result.get("metrics", {}).get("avg_win", 0),
# #                 "avg_loss": result.get("metrics", {}).get("avg_loss", 0),
# #                 "max_drawdown": result.get("metrics", {}).get("max_drawdown_percent", 0),
# #                 "sharpe_ratio": result.get("metrics", {}).get("sharpe_ratio", 0),
# #                 "trades": result.get("trades", [])
# #             }
# #         else:
# #             return {
# #                 "ok": False,
# #                 "error": f"Backtest failed for {symbol}",
# #                 "details": str(result)
# #             }
    
# #     except Exception as e:
# #         print(f"❌ Backtest error: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return {
# #             "ok": False,
# #             "error": f"Backtest failed: {str(e)}"
# #         }


# # # @app.get('/backtest/status')
# # # def backtest_status():
# # #     """Check if backtesting is available"""
# # #     try:
# # #         from backtesting_engine import BacktestingEngine
# # #         return {
# # #             "ok": True,
# # #             "available": True,
# # #             "message": "Backtesting engine is ready"
# # #         }
# # #     except Exception as e:
# # #         return {
# # #             "ok": False,
# # #             "available": False,
# # #             "error": str(e)
# # #         }







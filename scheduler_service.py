# scheduler_service.py
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import httpx

# Import your existing modules
from binance_utils import filter_valid_binance_symbols
from paper_trading_engine import ai_trade, start_trade_for_symbol
from trade_history import get_open_trade
from async_data_utils import batch_fetch_candles, bulk_save_to_db

# --- STATE MANAGEMENT ---
active_trade_list = set() 
scheduler = AsyncIOScheduler()
TRADING_ACTIVE = False

# External API for "Consistent Gainers"
EXTERNAL_SOURCE_API = "http://57.128.242.183:9100/api/consistent-gainers-all"

# ==========================================

# 🐢 TRACK A: Heavy Analysis (Every 2 Hours)
# ==========================================
async def track_a_heavy_analysis():
    global active_trade_list
    print(f"\n[TRACK A] 🐢 Starting Heavy Analysis: {datetime.now()}")

    # 1. Fetch Candidates from External API
    raw_symbols = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(EXTERNAL_SOURCE_API, timeout=60)
            data = resp.json()
            
            # Handle different API response formats
            if isinstance(data, dict):
                # Try multiple possible field names
                if 'symbols' in data and isinstance(data['symbols'], list):
                    raw_symbols = data['symbols']
                elif 'data' in data and isinstance(data['data'], list):
                    # Extract symbols from list of dicts
                    for item in data['data']:
                        if isinstance(item, dict) and 'symbol' in item:
                            raw_symbols.append(item['symbol'])
                        elif isinstance(item, str):
                            raw_symbols.append(item)
                elif 'gainers' in data and isinstance(data['gainers'], list):
                    for item in data['gainers']:
                        if isinstance(item, dict) and 'symbol' in item:
                            raw_symbols.append(item['symbol'])
                        elif isinstance(item, str):
                            raw_symbols.append(item)
            elif isinstance(data, list):
                # Direct list of symbols or objects
                for item in data:
                    if isinstance(item, dict) and 'symbol' in item:
                        raw_symbols.append(item['symbol'])
                    elif isinstance(item, str):
                        raw_symbols.append(item)
            
            # Fallback handling
            if not raw_symbols and isinstance(data, dict):
                for key in ['symbols', 'data', 'gainers']:
                    if key in data and isinstance(data[key], list):
                        raw_symbols = data[key]
                        break

    except Exception as e:
        print(f"⚠️ Track A External API failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if not raw_symbols:
        print("⚠️ No symbols found from external API.")
        return

    # 2. Normalize and Validate against Binance
    # Ensure USDT suffix and handle both dict and string symbols
    normalized = []
    for s in raw_symbols:
        if isinstance(s, dict) and 'symbol' in s:
            s = s['symbol']
        if isinstance(s, str):
            s = s.upper().strip()
            if not s.endswith('USDT'):
                s += 'USDT'
            normalized.append(s)
    
    valid_symbols = filter_valid_binance_symbols(normalized)
    
    # 3. Update the Active List
    active_trade_list.update(valid_symbols)
    
    print(f"[TRACK A] ✅ Updated Watchlist: {len(active_trade_list)} coins.")
    print(f"👀 Watching: {list(active_trade_list)[:10]}...")


# ==========================================
# 🐇 TRACK B: Fast Stream (Every 1 Minute)
# ==========================================
async def track_b_fast_stream():
    global active_trade_list
    if not active_trade_list:
        print("[TRACK B] 💤 No coins to monitor.")
        return

    print(f"[TRACK B] 🐇 Scanning {len(active_trade_list)} coins...")

    # 1. Batch Fetch 1m Data
    candles_map = await batch_fetch_candles(list(active_trade_list), interval="1m", limit=50)

    # 2. Bulk Save
    await bulk_save_to_db(candles_map, interval="1m")

    # 3. Concurrent Analysis
    tasks = []
    for sym, df in candles_map.items():
        tasks.append(process_fast_signal(sym, df))
    
    await asyncio.gather(*tasks)

async def process_fast_signal(symbol, df):
    """Run AI check and execute trade if needed."""
    # 1. Check open trades
    trade = get_open_trade(symbol)
    if trade and trade.get("status") == "OPEN":
        return 

    # 2. Run AI
    decision_data = await ai_trade(symbol, df)
    
    # 3. Execute
    decision = decision_data.get("decision", "HOLD")
    confidence = decision_data.get("confidence", 0)
    
    if decision == "BUY" and confidence >= 75:
        print(f"🚀 [TRACK B] BUY SIGNAL: {symbol} ({confidence}%)")
        start_trade_for_symbol(symbol)


# ==========================================
# 🎮 CONTROLS
# ==========================================
def initialize_scheduler():
    """Initialize scheduler (jobs added later when event loop is ready)"""
    try:
        if not scheduler.running:
            # DON'T start yet - wait for event loop
            print("⏳ Scheduler initialized (jobs will start on first request).")
    except Exception as e:
        print(f"⚠️ Scheduler init warning: {e}")

async def activate_auto_trading():
    global TRADING_ACTIVE
    if TRADING_ACTIVE:
        return {"status": "active", "message": "Already running."}

    TRADING_ACTIVE = True
    print("🚀 MANUAL TRIGGER: Starting Auto-Trading Sequence...")

    # Start scheduler if not already running
    if not scheduler.running:
        scheduler.start()
        print("✅ Scheduler engine started!")

    # Run Track A immediately (Await it so user sees results)
    await track_a_heavy_analysis()

    # Schedule future jobs
    if not scheduler.get_job('track_a'):
        scheduler.add_job(track_a_heavy_analysis, 'interval', hours=2, id='track_a')
    
    if not scheduler.get_job('track_b'):
        scheduler.add_job(track_b_fast_stream, 'interval', minutes=1, id='track_b')
    
    return {"status": "success", "message": "Analysis done. Auto-pilot activated. Next fetch in 2 hours."}

def stop_auto_trading():
    global TRADING_ACTIVE
    scheduler.remove_all_jobs()
    TRADING_ACTIVE = False
    return {"status": "stopped"}
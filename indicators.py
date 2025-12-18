# # indicators.py

# indicators.py
import pandas as pd
import numpy as np
from utils import timer


def get_default_indicators():
    """Return default/neutral indicators."""
    return {
        'ema_9': 0,
        'ema_21': 0,
        'ema_slope': 0,
        'rsi': 50.0,
        'macd': 0,
        'macd_signal': 0,
        'macd_hist': 0,
        'macd_slope': 0,
        'bb_upper': 0,
        'bb_middle': 0,
        'bb_lower': 0,
        'atr': 0,
        'atr_trend': 0,
        'vwap': 0,
        'stochastic': 50.0,
        'cci': 0
    }


@timer
def calculate_indicators(df, latest_price=None):
    """Calculate technical indicators with TradingView-accurate formulas."""

    # Safety check - ensure we have minimum data
    if df is None or df.empty or 'close' not in df.columns:
        return get_default_indicators()

    # Append latest price (fake candle removed completely)
    if latest_price is not None:
        df = df.copy()
        df.loc[len(df)] = [
            latest_price, latest_price, latest_price,
            latest_price, 0
        ]

    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    indicators = {}

    if len(df) < 2:
        return get_default_indicators()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    try:

        # --------------------------------
        # EMA (TradingView exact)
        # --------------------------------
        indicators["ema_9"] = close.ewm(span=9, adjust=False).mean().iloc[-1]
        indicators["ema_21"] = close.ewm(span=21, adjust=False).mean().iloc[-1]
        indicators["ema_slope"] = indicators["ema_9"] - indicators["ema_21"]

        # --------------------------------
        # RSI (TradingView / Wilder)
        # --------------------------------
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()

        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi"] = float(rsi.iloc[-1])

        # --------------------------------
        # MACD (TradingView standard)
        # --------------------------------
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        indicators["macd"] = float(macd_line.iloc[-1])
        indicators["macd_signal"] = float(macd_signal.iloc[-1])
        indicators["macd_hist"] = float(macd_hist.iloc[-1])
        indicators["macd_slope"] = float(macd_line.iloc[-1] - macd_line.iloc[-2])

        # --------------------------------
        # Bollinger Bands (20, 2)
        # --------------------------------
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()

        indicators["bb_middle"] = float(sma20.iloc[-1])
        indicators["bb_upper"] = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
        indicators["bb_lower"] = float(sma20.iloc[-1] - 2 * std20.iloc[-1])

        # --------------------------------
        # ATR (14) - True TradingView logic
        # --------------------------------
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()    # Wilders
        indicators["atr"] = float(atr.iloc[-1])
        indicators["atr_trend"] = float(atr.iloc[-1] - atr.iloc[-2])

        # --------------------------------
        # VWAP (TradingView exact)
        # --------------------------------
        tp = (high + low + close) / 3
        vwap = (tp * volume).cumsum() / volume.cumsum()
        indicators["vwap"] = float(vwap.iloc[-1])

        # --------------------------------
        # Stochastic %K (14) TradingView
        # --------------------------------
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()

        stoch_k = (close - low_14) / (high_14 - low_14) * 100
        indicators["stochastic"] = float(stoch_k.iloc[-1])

        # --------------------------------
        # CCI (20) TradingView
        # --------------------------------
        typical = (high + low + close) / 3
        sma_tp = typical.rolling(20).mean()
        mean_dev = (typical - sma_tp).abs().rolling(20).mean()

        cci = (typical - sma_tp) / (0.015 * mean_dev)
        indicators["cci"] = float(cci.iloc[-1])

    except Exception as e:
        print(f"[ERROR] Indicator calc failed: {e}")
        return get_default_indicators()

    return indicators


def evaluate_trend(df, indicators=None):
    """Trend model (unchanged)."""
    try:
        if indicators is None:
            indicators = calculate_indicators(df)
    except Exception:
        return {'trend': 'NEUTRAL', 'score': 0.0, 'votes': {}, 'reason': 'indicators_unavailable'}

    votes = {}
    score = 0.0

    # EMA cross
    ema9 = float(indicators.get('ema_9') or 0)
    ema21 = float(indicators.get('ema_21') or 0)
    if ema9 > ema21:
        votes['ema_cross_up'] = True
        score += 1.0
    elif ema9 < ema21:
        votes['ema_cross_down'] = True
        score -= 1.0

    # EMA slope
    ema_slope = float(indicators.get('ema_slope') or 0)
    if ema_slope > 0:
        votes['ema_slope_up'] = True
        score += 0.6
    elif ema_slope < 0:
        votes['ema_slope_down'] = True
        score -= 0.6

    # MACD momentum
    macd_hist = float(indicators.get('macd_hist') or 0)
    macd_slope = float(indicators.get('macd_slope') or 0)
    if macd_hist > 0 and macd_slope > 0:
        votes['macd_momentum_up'] = True
        score += 0.8
    elif macd_hist < 0 and macd_slope < 0:
        votes['macd_momentum_down'] = True
        score -= 0.8

    # ATR trend
    atr_trend = float(indicators.get('atr_trend') or 0)
    if atr_trend > 0:
        votes['atr_expanding'] = True
        score += 0.4
    elif atr_trend < 0:
        votes['atr_contracting'] = True
        score -= 0.2

    # RSI levels
    rsi = float(indicators.get('rsi') or 50)
    if rsi < 35:
        votes['rsi_oversold'] = True
        score += 0.5
    elif rsi > 65:
        votes['rsi_overbought'] = True
        score -= 0.5

    # VWAP bias
    vwap = float(indicators.get('vwap') or 0)
    price = float(df['close'].iloc[-1])
    if price > vwap:
        votes['price_above_vwap'] = True
        score += 0.4
    elif price < vwap:
        votes['price_below_vwap'] = True
        score -= 0.4

    # CCI bias
    cci = float(indicators.get('cci') or 0)
    if cci > 100:
        votes['cci_pos'] = True
        score += 0.3
    elif cci < -100:
        votes['cci_neg'] = True
        score -= 0.3

    # Final decision
    trend = "NEUTRAL"
    if score >= 1.8:
        trend = "UP"
    elif score <= -1.8:
        trend = "DOWN"

    reason = ", ".join([k.replace("_", " ") for k, v in votes.items() if v]) or "no clear bias"

    return {"trend": trend, "score": round(score, 3), "votes": votes, "reason": reason}


def get_multi_timeframe_indicators(symbol: str):
    """Fetch candles and compute indicators (unchanged)."""
    from data_fetcher import get_recent_candles

    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    results = {}

    for tf in timeframes:
        try:
            df = get_recent_candles(symbol, tf, limit=1000)
            if df is None or df.empty:
                print(f"[SKIP] {symbol} {tf}: no candles")
                continue

            indicators = calculate_indicators(df)
            trend_info = evaluate_trend(df, indicators)
            results[tf] = trend_info
        except Exception as e:
            print(f"[ERROR] {symbol} {tf}: {e}")

    return results






# import pandas as pd
# import numpy as np


# def get_default_indicators():
#     """Return default/neutral indicators."""
#     return {
#         'ema_9': 0,
#         'ema_21': 0,
#         'ema_slope': 0,
#         'rsi': 50.0,
#         'macd': 0,
#         'macd_signal': 0,
#         'macd_hist': 0,
#         'macd_slope': 0,
#         'bb_upper': 0,
#         'bb_middle': 0,
#         'bb_lower': 0,
#         'atr': 0,
#         'atr_trend': 0,
#         'vwap': 0,
#         'stochastic': 50.0,
#         'cci': 0
#     }


# def calculate_indicators(df, latest_price=None):
#     """Calculate technical indicators for trading analysis."""
    
#     # Safety check - ensure we have the minimum required data
#     if df is None or df.empty or 'close' not in df.columns:
#         return get_default_indicators()
    
#     # Append latest price as last candle if provided
#     if latest_price is not None:
#         new_row = pd.DataFrame([{
#             'open': latest_price,
#             'high': latest_price,
#             'low': latest_price,
#             'close': latest_price,
#             'volume': 0
#         }])
#         df = pd.concat([df, new_row], ignore_index=True)

#     # Ensure numeric columns - check which ones exist
#     for col in ['open', 'high', 'low', 'close', 'volume']:
#         if col in df.columns:
#             try:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#             except:
#                 pass

#     indicators = {}
    
#     if len(df) < 2:
#         return get_default_indicators()

#     try:
#         close_series = pd.Series(df['close'].values)


#         # --- EMA ---
#         if len(df) >= 9:
#             ema_9 = close_series.ewm(span=9, adjust=False).mean().iloc[-1]
#             indicators['ema_9'] = float(ema_9)
#         else:
#             indicators['ema_9'] = float(df['close'].iloc[-1])
        
#         if len(df) >= 21:
#             ema_21 = close_series.ewm(span=21, adjust=False).mean().iloc[-1]
#             indicators['ema_21'] = float(ema_21)
#         else:
#             indicators['ema_21'] = float(df['close'].iloc[-1])
            
#         indicators['ema_slope'] = indicators['ema_9'] - indicators['ema_21']

#         # --- RSI ---
#         if len(df) >= 14:
#             delta = close_series.diff()
#             gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#             rs = gain / loss
#             rsi = 100 - (100 / (1 + rs))
#             indicators['rsi'] = float(rsi.iloc[-1])
#         else:
#             indicators['rsi'] = 50.0

#         # --- MACD ---
#         if len(df) >= 26:
#             ema_12 = close_series.ewm(span=12, adjust=False).mean()
#             ema_26 = close_series.ewm(span=26, adjust=False).mean()
#             macd_line = ema_12 - ema_26
#             signal_line = macd_line.ewm(span=9, adjust=False).mean()
#             macd_hist = macd_line - signal_line
            
#             indicators['macd'] = float(macd_line.iloc[-1])
#             indicators['macd_signal'] = float(signal_line.iloc[-1])
#             indicators['macd_hist'] = float(macd_hist.iloc[-1])
#             if len(macd_line) > 5:
#                 indicators['macd_slope'] = float(macd_line.iloc[-1] - macd_line.iloc[-5])
#             else:
#                 indicators['macd_slope'] = 0.0
#         else:
#             indicators['macd'] = 0.0
#             indicators['macd_signal'] = 0.0
#             indicators['macd_hist'] = 0.0
#             indicators['macd_slope'] = 0.0

#         # --- Bollinger Bands ---
#         if len(df) >= 20:
#             sma_20 = close_series.rolling(20).mean()
#             std_20 = close_series.rolling(20).std()
#             indicators['bb_upper'] = float(sma_20.iloc[-1] + 2 * std_20.iloc[-1])
#             indicators['bb_middle'] = float(sma_20.iloc[-1])
#             indicators['bb_lower'] = float(sma_20.iloc[-1] - 2 * std_20.iloc[-1])
#         else:
#             indicators['bb_upper'] = float(df['close'].max())
#             indicators['bb_middle'] = float(df['close'].mean())
#             indicators['bb_lower'] = float(df['close'].min())

#         # --- ATR ---
#         if len(df) >= 14 and 'high' in df.columns and 'low' in df.columns:
#             high = pd.Series(df['high'].values)
#             low = pd.Series(df['low'].values)
#             close_prev = close_series.shift(1)
            
#             tr1 = high - low
#             tr2 = (high - close_prev).abs()
#             tr3 = (low - close_prev).abs()
            
#             tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#             atr = tr.rolling(14).mean()
#             indicators['atr'] = float(atr.iloc[-1])
#             indicators['atr_trend'] = 0.0
#         else:
#             indicators['atr'] = 0.0
#             indicators['atr_trend'] = 0.0

#         # --- VWAP ---
#         if 'volume' in df.columns and len(df) > 0:
#             cumulative_volume = pd.Series(df['volume'].values).cumsum()
#             cumulative_vwap = (close_series * df['volume'].values).cumsum()
#             vwap = cumulative_vwap / cumulative_volume
#             indicators['vwap'] = float(vwap.iloc[-1])
#         else:
#             indicators['vwap'] = float(df['close'].iloc[-1])

#         # --- Stochastic Oscillator ---
#         if len(df) >= 14 and 'high' in df.columns and 'low' in df.columns:
#             high_14 = pd.Series(df['high'].values).rolling(14).max()
#             low_14 = pd.Series(df['low'].values).rolling(14).min()
            
#             denominator = high_14.iloc[-1] - low_14.iloc[-1]
#             if denominator != 0 and not np.isnan(denominator):
#                 stoch = ((df['close'].iloc[-1] - low_14.iloc[-1]) / denominator) * 100
#                 indicators['stochastic'] = float(stoch)
#             else:
#                 indicators['stochastic'] = 50.0
#         else:
#             indicators['stochastic'] = 50.0

#         # --- CCI ---
#         if len(df) >= 20 and 'high' in df.columns and 'low' in df.columns:
#             typical_price = (df['high'] + df['low'] + df['close']) / 3
#             tp_sma = typical_price.rolling(20).mean()
#             mean_deviation = (typical_price - tp_sma).abs().rolling(20).mean()
            
#             denom = 0.015 * mean_deviation.iloc[-1]
#             if denom != 0 and not np.isnan(denom):
#                 cci = (typical_price.iloc[-1] - tp_sma.iloc[-1]) / denom
#                 indicators['cci'] = float(cci)
#             else:
#                 indicators['cci'] = 0.0
#         else:
#             indicators['cci'] = 0.0
        
#     except Exception as e:
#         print(f"[ERROR] Indicator calculation failed: {str(e)[:100]}")
#         return get_default_indicators()

#     return indicators


# def evaluate_trend(df, indicators=None):
#     """Enhanced trend evaluation using multiple factors."""
#     try:
#         if indicators is None:
#             indicators = calculate_indicators(df)
#     except Exception:
#         return {'trend': 'NEUTRAL', 'score': 0.0, 'votes': {}, 'reason': 'indicators_unavailable'}

#     votes = {}
#     score = 0.0

#     # EMA cross
#     ema9 = float(indicators.get('ema_9') or 0)
#     ema21 = float(indicators.get('ema_21') or 0)
#     if ema9 > ema21:
#         votes['ema_cross_up'] = True
#         score += 1.0
#     elif ema9 < ema21:
#         votes['ema_cross_down'] = True
#         score -= 1.0

#     # EMA slope
#     ema_slope = float(indicators.get('ema_slope') or 0)
#     if ema_slope > 0:
#         votes['ema_slope_up'] = True
#         score += 0.6
#     elif ema_slope < 0:
#         votes['ema_slope_down'] = True
#         score -= 0.6

#     # MACD histogram & slope
#     macd_hist = float(indicators.get('macd_hist') or 0)
#     macd_slope = float(indicators.get('macd_slope') or 0)
#     if macd_hist > 0 and macd_slope > 0:
#         votes['macd_momentum_up'] = True
#         score += 0.8
#     elif macd_hist < 0 and macd_slope < 0:
#         votes['macd_momentum_down'] = True
#         score -= 0.8

#     # ATR trend
#     atr_trend = float(indicators.get('atr_trend') or 0)
#     if atr_trend > 0:
#         votes['atr_expanding'] = True
#         score += 0.4
#     elif atr_trend < 0:
#         votes['atr_contracting'] = True
#         score -= 0.2

#     # RSI levels
#     rsi = float(indicators.get('rsi') or 50)
#     if rsi < 35:
#         votes['rsi_oversold'] = True
#         score += 0.5
#     elif rsi > 65:
#         votes['rsi_overbought'] = True
#         score -= 0.5

#     # Price vs VWAP
#     vwap = float(indicators.get('vwap') or 0)
#     price = float(df['close'].iloc[-1])
#     if vwap and price > vwap:
#         votes['price_above_vwap'] = True
#         score += 0.4
#     elif vwap and price < vwap:
#         votes['price_below_vwap'] = True
#         score -= 0.4

#     # CCI bias
#     cci = float(indicators.get('cci') or 0)
#     if cci > 100:
#         votes['cci_pos'] = True
#         score += 0.3
#     elif cci < -100:
#         votes['cci_neg'] = True
#         score -= 0.3

#     # Final decision
#     trend = 'NEUTRAL'
#     if score >= 1.8:
#         trend = 'UP'
#     elif score <= -1.8:
#         trend = 'DOWN'

#     reason = ', '.join([k.replace('_', ' ') for k, v in votes.items() if v]) or 'no clear bias'

#     return {'trend': trend, 'score': round(score, 3), 'votes': votes, 'reason': reason}


# def get_multi_timeframe_indicators(symbol: str):
#     """Fetch candles for multiple timeframes and compute indicators."""
#     from data_fetcher import get_recent_candles

#     timeframes = ["1m", "5m", "15m", "1h", "4h"]
#     results = {}

#     for tf in timeframes:
#         try:
#             df = get_recent_candles(symbol, tf, limit=1000)
#             if df is None or df.empty:
#                 print(f"[SKIP] {symbol} {tf}: no candles")
#                 continue

#             indicators = calculate_indicators(df)
#             trend_info = evaluate_trend(df, indicators)
#             results[tf] = trend_info
#         except Exception as e:
#             print(f"[ERROR] {symbol} {tf}: {e}")

#     return results


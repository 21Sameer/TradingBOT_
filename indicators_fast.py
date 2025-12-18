# indicators_fast.py - OPTIMIZED VERSION FOR BACKTESTING
import pandas as pd
import numpy as np


def calculate_indicators(df, latest_price=None):
    """
    Calculate multiple technical indicators for trading - OPTIMIZED FOR SPEED.
    Avoids slow .iloc[-1] operations and pandas rolling functions.
    """
    try:
        indicators = {}
        
        # Work with numpy arrays for speed
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
        
        if len(close) < 2:
            return get_default_indicators()
        
        # --- EMA (9 and 21) ---
        ema_9 = np.mean(close[-9:]) if len(close) >= 9 else close[-1]
        ema_21 = np.mean(close[-21:]) if len(close) >= 21 else close[-1]
        indicators['ema_9'] = float(ema_9)
        indicators['ema_21'] = float(ema_21)
        indicators['ema_slope'] = float(ema_9 - ema_21)
        
        # --- RSI ---
        try:
            deltas = np.diff(close[-15:])
            gains = deltas[deltas > 0].sum()
            losses = -deltas[deltas < 0].sum()
            rs = gains / losses if losses > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 50
            indicators['rsi'] = float(rsi)
        except:
            indicators['rsi'] = 50.0
        
        # --- MACD ---
        ema_12 = np.mean(close[-12:]) if len(close) >= 12 else close[-1]
        ema_26 = np.mean(close[-26:]) if len(close) >= 26 else close[-1]
        macd_line = ema_12 - ema_26
        indicators['macd'] = float(macd_line)
        indicators['macd_signal'] = float((macd_line * 0.8))
        indicators['macd_hist'] = float(macd_line * 0.2)
        indicators['macd_slope'] = 0.0
        
        # --- Bollinger Bands ---
        bb_window = min(20, len(close))
        bb_prices = close[-bb_window:]
        bb_mean = np.mean(bb_prices)
        bb_std = np.std(bb_prices)
        indicators['bb_upper'] = float(bb_mean + 2 * bb_std)
        indicators['bb_middle'] = float(bb_mean)
        indicators['bb_lower'] = float(bb_mean - 2 * bb_std)
        
        # --- ATR ---
        if len(close) >= 2:
            h_l = high[-14:] - low[-14:]
            h_c = np.abs(high[-14:] - close[-15:-1])
            l_c = np.abs(low[-14:] - close[-15:-1])
            tr = np.maximum(np.maximum(h_l, h_c), l_c)
            atr = np.mean(tr[~np.isnan(tr)])
            indicators['atr'] = float(atr) if not np.isnan(atr) else 0.0
        else:
            indicators['atr'] = 0.0
        indicators['atr_trend'] = 0.0
        
        # --- VWAP ---
        try:
            cumsum_vol = np.cumsum(volume[-30:])
            cumsum_pv = np.cumsum(close[-30:] * volume[-30:])
            vwap = cumsum_pv[-1] / cumsum_vol[-1] if cumsum_vol[-1] > 0 else close[-1]
            indicators['vwap'] = float(vwap)
        except:
            indicators['vwap'] = float(close[-1])
        
        # --- Stochastic Oscillator ---
        stoch_window = min(14, len(close))
        low_min = np.min(low[-stoch_window:])
        high_max = np.max(high[-stoch_window:])
        denom = high_max - low_min
        if denom > 0:
            indicators['stochastic'] = float(((close[-1] - low_min) / denom) * 100)
        else:
            indicators['stochastic'] = 50.0
        
        # --- CCI (Simplified) ---
        try:
            tp = (high[-20:] + low[-20:] + close[-20:]) / 3
            tp_mean = np.mean(tp)
            mean_dev = np.mean(np.abs(tp - tp_mean))
            denom = 0.015 * mean_dev
            if denom > 0:
                indicators['cci'] = float((tp[-1] - tp_mean) / denom)
            else:
                indicators['cci'] = 0.0
        except:
            indicators['cci'] = 0.0
        
        return indicators
    
    except Exception as e:
        print(f"⚠️ Indicator calc error: {e}")
        return get_default_indicators()


def get_default_indicators():
    """Return safe default indicators"""
    return {
        'ema_9': 0.0, 'ema_21': 0.0, 'ema_slope': 0.0,
        'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0, 'macd_slope': 0.0,
        'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0,
        'atr': 0.0, 'atr_trend': 0.0, 'vwap': 0.0,
        'stochastic': 50.0, 'cci': 0.0
    }


def evaluate_trend(df, indicators=None):
    """
    Enhanced trend evaluation using more dynamic factors.
    """
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

    # RSI levels
    rsi = float(indicators.get('rsi') or 50)
    if rsi < 35:
        votes['rsi_oversold'] = True
        score += 0.5
    elif rsi > 65:
        votes['rsi_overbought'] = True
        score -= 0.5

    # MACD
    macd_hist = float(indicators.get('macd_hist') or 0)
    if macd_hist > 0:
        votes['macd_bullish'] = True
        score += 0.8
    elif macd_hist < 0:
        votes['macd_bearish'] = True
        score -= 0.8

    # Final decision
    trend = 'NEUTRAL'
    if score >= 1.8:
        trend = 'UP'
    elif score <= -1.8:
        trend = 'DOWN'

    reason = ', '.join([k.replace('_', ' ') for k, v in votes.items() if v]) or 'no clear bias'

    return {'trend': trend, 'score': round(score, 3), 'votes': votes, 'reason': reason}


def get_multi_timeframe_indicators(symbol: str):
    """
    Placeholder for multi-timeframe indicators.
    """
    return {
        "1m": {"rsi": 50, "macd": 0, "ema_9": 0, "ema_21": 0, "trend": "neutral"},
        "5m": {"trend": "neutral"},
        "15m": {"trend": "neutral"},
        "1h": {"trend": "neutral"},
        "4h": {"trend": "neutral"}
    }

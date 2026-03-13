import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# [FIX 4] PATH & ENV — Absolute path เสมอ
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))  # ✅ Absolute path

# ─────────────────────────────────────────────
# [FIX 3] ENTERPRISE LOGGING — Console + File
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(                              # ✅ Absolute path
            os.path.join(LOG_DIR, "stock_sync.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.Sync")

from config import get_supabase, SET100_SYMBOLS  # type: ignore

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
supabase       = get_supabase()
THAI_TZ        = ZoneInfo("Asia/Bangkok")     # [FIX 1] Named timezone
FETCH_PERIOD   = "60d"
BATCH_SIZE     = 10
MAX_RETRY      = 3
BATCH_SLEEP    = 0.8

# ─────────────────────────────────────────────
# [FIX 2] RETRY HELPER — Exponential Backoff
# ─────────────────────────────────────────────
def retry_call(fn, *args, max_retries: int = MAX_RETRY, label: str = "call", **kwargs):
    """Generic retry wrapper พร้อม exponential backoff"""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            wait = 2 ** attempt
            if attempt == max_retries:
                log.error("❌ [%s] ล้มเหลวทุก %d retry: %s", label, max_retries, exc)
                raise
            log.warning("⚠️ [%s] attempt %d/%d: %s — retry in %ds",
                        label, attempt, max_retries, exc, wait)
            time.sleep(wait)

# ─────────────────────────────────────────────
# MARKET STATUS DETECTION
# ─────────────────────────────────────────────
def get_market_status() -> str:
    """
    ตรวจสถานะตลาดหุ้นไทย (SET) จากเวลาปัจจุบัน (THAI_TZ)
    [FIX 1] ใช้ ZoneInfo("Asia/Bangkok") แทน hardcode offset
    """
    now  = datetime.now(THAI_TZ)   # [FIX 1]
    wday = now.weekday()           # 0=จันทร์, 5=เสาร์, 6=อาทิตย์
    hm   = now.hour * 100 + now.minute

    if wday >= 5:              return "CLOSED"
    if hm < 930:               return "CLOSED"
    if 930  <= hm < 1000:      return "PRE_OPEN"
    if 1000 <= hm < 1230:      return "OPEN"
    if 1230 <= hm < 1400:      return "LUNCH"
    if 1400 <= hm < 1700:      return "OPEN"
    return "CLOSED"

# ─────────────────────────────────────────────
# INDICATOR CALCULATIONS (Quant Standard)
# ─────────────────────────────────────────────
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI มาตรฐาน Wilder's Smoothing — ตรงกับ TradingView"""
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def calc_macd(prices: pd.Series, fast=12, slow=26,
              signal=9) -> tuple[pd.Series, pd.Series]:
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ─────────────────────────────────────────────
# SAFE VALUE HELPERS
# ─────────────────────────────────────────────
def safe_float(val, default=None) -> Optional[float]:
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default

def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except Exception:
        return default

# ─────────────────────────────────────────────
# YFINANCE BATCH DOWNLOADER (with retry)
# ─────────────────────────────────────────────
def download_batch(symbols: list[str],
                   retries: int = MAX_RETRY) -> dict[str, pd.DataFrame]:
    """ดาวน์โหลดราคาหลายหุ้นพร้อมกัน — [FIX 2] exponential backoff"""
    tickers = [f"{s}.BK" for s in symbols]
    raw     = None

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                period=FETCH_PERIOD,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            wait = 2 ** attempt
            log.warning("⚠️ yf.download attempt %d/%d: %s — retry in %ds",
                        attempt, retries, e, wait)
            if attempt == retries:
                log.error("❌ yf.download ล้มเหลวทุก retry")
                return {}
            time.sleep(wait)

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = raw.copy() if len(symbols) == 1 else raw[f"{sym}.BK"].copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 30:
                result[sym] = df
            else:
                log.warning("⚠️ %s: ข้อมูลไม่เพียงพอ (%d rows)", sym, len(df))
        except KeyError:
            log.warning("⚠️ %s: ไม่พบข้อมูลใน yf response", sym)
    return result

# ─────────────────────────────────────────────
# PROCESS SINGLE SYMBOL
# ─────────────────────────────────────────────
def process_symbol(symbol: str, df: pd.DataFrame,
                   market_status: str) -> Optional[dict]:
    """รับ DataFrame OHLCV → คำนวณ Indicators → return payload dict"""
    prices              = df["Close"].astype(float)
    rsi_series          = calc_rsi(prices, 14)
    macd_line, sig_line = calc_macd(prices)

    last        = df.iloc[-1]
    prev        = df.iloc[-2] if len(df) >= 2 else last
    last_price  = safe_float(last["Close"])
    prev_close  = safe_float(prev["Close"])
    pct_change  = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0

    return {
        "symbol":         symbol,
        "last_price":     last_price,
        "open_price":     safe_float(last.get("Open")),
        "high_price":     safe_float(last.get("High")),
        "low_price":      safe_float(last.get("Low")),
        "volume":         safe_int(last.get("Volume")),
        "rsi_14":         safe_float(rsi_series.iloc[-1]),
        "macd_val":       safe_float(macd_line.iloc[-1]),
        "macd_signal":    safe_float(sig_line.iloc[-1]),
        "percent_change": round(pct_change, 4),
        "market_status":  market_status,
        # [FIX 1] updated_at เป็น UTC มาตรฐาน ✅
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }

# ─────────────────────────────────────────────
# MAIN SYNC
# ─────────────────────────────────────────────
def sync_realtime_data():
    start_time    = time.time()
    market_status = get_market_status()

    log.info("=" * 60)
    log.info("🚀 Trinity Realtime Sync | %s | Status: %s",
             datetime.now(THAI_TZ).strftime("%H:%M:%S"), market_status)
    log.info("🔖 Batch Process Started | symbols=%d", len(SET100_SYMBOLS))
    log.info("=" * 60)

    total_success = 0
    total_fail    = 0

    batches = [SET100_SYMBOLS[i:i+BATCH_SIZE]
               for i in range(0, len(SET100_SYMBOLS), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, 1):
        log.info("📦 Batch %d/%d: %s", batch_idx, len(batches), ', '.join(batch))
        data_map = download_batch(batch)

        batch_payloads = []
        for symbol in batch:
            if symbol not in data_map:
                log.warning("⚠️ %s | No Data", symbol)
                total_fail += 1
                continue
            try:
                payload = process_symbol(symbol, data_map[symbol], market_status)
                if payload:
                    batch_payloads.append(payload)
                    rsi_str = f"{payload['rsi_14']:.2f}" if payload["rsi_14"] else "N/A"
                    log.info("✅ Symbol %s Updated | ฿%8.2f | %+6.2f%% | RSI: %s",
                             symbol, payload["last_price"],
                             payload["percent_change"], rsi_str)
                    total_success += 1
            except Exception as e:
                log.error("❌ %s | Process error: %s", symbol, e, exc_info=True)
                total_fail += 1

        # [FIX 2] Bulk upsert พร้อม retry
        if batch_payloads:
            try:
                retry_call(
                    lambda p=batch_payloads: supabase.table("stock_realtime")
                        .upsert(p, on_conflict="symbol")
                        .execute(),
                    label=f"upsert_realtime_batch_{batch_idx}",
                )
                log.info("💾 Batch %d upsert: %d rows saved",
                         batch_idx, len(batch_payloads))
            except Exception as e:
                log.error("❌ Batch %d upsert failed, trying fallback: %s",
                          batch_idx, e)
                for p in batch_payloads:
                    try:
                        supabase.table("stock_realtime")\
                            .upsert(p, on_conflict="symbol").execute()
                    except Exception as e2:
                        log.error("❌ Fallback upsert %s: %s", p["symbol"], e2)

        if batch_idx < len(batches):
            time.sleep(BATCH_SLEEP)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("🏁 Batch Process Completed in %.1fs | ✅ %d | ❌ %d",
             elapsed, total_success, total_fail)
    log.info("=" * 60)


# ─────────────────────────────────────────────
# SCHEDULER SUPPORT
# ─────────────────────────────────────────────
def run_with_schedule():
    """รัน sync ทุก 1 นาทีในช่วงตลาดเปิด"""
    import schedule

    def job():
        status = get_market_status()
        if status in ("OPEN", "PRE_OPEN"):
            sync_realtime_data()
        else:
            log.info("⏸️  ตลาดปิด (%s) — รอรอบถัดไป", status)

    schedule.every(1).minutes.do(job)
    log.info("⏰ Scheduler เริ่มทำงาน (sync ทุก 1 นาที)")
    sync_realtime_data()
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true",
                        help="รันแบบ loop ต่อเนื่อง")
    args = parser.parse_args()

    if args.schedule:
        run_with_schedule()
    else:
        sync_realtime_data()
"""
historical.py - Trinity EOD Archive Engine (Enterprise Edition)
================================================================
Archive ราคาปิด EOD → stock_historical + อัปเดต actual_price/error_percent ใน predictions

[ENTERPRISE FIXES APPLIED]
  1. TIMEZONE  : ใช้ THAI_TZ (Asia/Bangkok) สำหรับ date logic ทุกจุด
  2. RETRY     : supabase upsert + yfinance มี exponential backoff
  3. LOGGING   : FileHandler บันทึกลง logs/historical.log (Absolute Path)
  4. ENV PATH  : BASE_DIR + load_dotenv absolute path
  5. APE       : error_percent = |predicted - actual| / actual × 100 (ไม่มี sign)
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
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
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
_DATE_FMT   = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_DATE_FMT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(                              # ✅ Absolute path log file
            os.path.join(LOG_DIR, "historical.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.Historical")

from config import get_supabase, SET100_SYMBOLS  # type: ignore

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
supabase         = get_supabase()
THAI_TZ          = ZoneInfo("Asia/Bangkok")       # [FIX 1] Timezone ชัดเจน
BATCH_SIZE       = 10
MAX_RETRY        = 3
BATCH_SLEEP      = 1.0
INDICATOR_BUFFER = 60   # วันย้อนหลังสำหรับ warm-up RSI14 / MACD26

# ─────────────────────────────────────────────
# [FIX 2] RETRY HELPER — Exponential Backoff
# ─────────────────────────────────────────────
def retry_call(fn, *args, max_retries: int = MAX_RETRY, label: str = "call", **kwargs):
    """
    Generic retry wrapper พร้อม exponential backoff
    ใช้สำหรับ Supabase และ API calls ทั่วไป
    """
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
# INDICATOR CALCULATIONS (Wilder's EMA — มาตรฐาน)
# ─────────────────────────────────────────────
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def calc_macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ─────────────────────────────────────────────
# SAFE VALUE HELPERS
# ─────────────────────────────────────────────
def safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None

def safe_int(val) -> int:
    try:
        return int(val)
    except Exception:
        return 0

# ─────────────────────────────────────────────
# INCREMENTAL DATE DETECTION
# ─────────────────────────────────────────────
def get_missing_dates_per_symbol(symbols: list[str]) -> dict[str, str]:
    """ดึงวันที่ล่าสุดที่บันทึกไว้ใน stock_historical ของแต่ละหุ้น"""
    log.info("🔍 ตรวจสอบวันที่ล่าสุดใน DB สำหรับแต่ละหุ้น...")
    try:
        res = retry_call(
            lambda: supabase.table("stock_historical")
                .select("symbol, date")
                .in_("symbol", symbols)
                .order("date", desc=True)
                .execute(),
            label="get_missing_dates",
        )
        latest: dict[str, str] = {}
        for row in (res.data or []):
            sym = row["symbol"]
            if sym not in latest:
                latest[sym] = row["date"]
        return latest
    except Exception as e:
        log.error("❌ get_missing_dates_per_symbol error: %s", e)
        return {}


def calc_fetch_start(last_date_str: Optional[str]) -> str:
    """
    คำนวณ start date ที่ต้องดึง:
    - มีข้อมูลอยู่แล้ว → ย้อนหลัง INDICATOR_BUFFER วันเพื่อ warm-up indicator
    - ยังไม่มีข้อมูล  → ดึง 6 เดือนย้อนหลัง

    [FIX 1] ใช้ THAI_TZ สำหรับ "วันนี้" เพื่อป้องกัน off-by-one ข้ามเที่ยงคืน UTC
    """
    if last_date_str:
        last_date = datetime.fromisoformat(last_date_str)
        start     = last_date - timedelta(days=INDICATOR_BUFFER)
    else:
        # [FIX 1] now ใช้ THAI_TZ ไม่ใช่ UTC เพื่อให้ตรงกับปฏิทินตลาดไทย
        start = datetime.now(THAI_TZ) - timedelta(days=180)
    return start.strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# YFINANCE BATCH DOWNLOADER (with retry)
# ─────────────────────────────────────────────
def download_batch(
    symbols: list[str],
    start_date: str,
    retries: int = MAX_RETRY,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV ตั้งแต่ start_date ถึงวันนี้ สำหรับหุ้น batch"""
    tickers = [f"{s}.BK" for s in symbols]
    raw     = None

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start_date,
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
                log.error("❌ yf.download ล้มเหลวทุก retry สำหรับ batch: %s",
                          ', '.join(symbols))
                return {}
            time.sleep(wait)

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = raw[f"{sym}.BK"].copy() if len(symbols) > 1 else raw.copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 5:
                result[sym] = df
        except (KeyError, TypeError):
            log.warning("⚠️ %s: ไม่พบใน yf response", sym)
    return result

# ─────────────────────────────────────────────
# BUILD HISTORICAL RECORDS (Incremental)
# ─────────────────────────────────────────────
def build_historical_records(
    symbol: str,
    df: pd.DataFrame,
    last_date_in_db: Optional[str],
) -> list[dict]:
    """
    คำนวณ indicators ทั้ง DataFrame แล้วกรองเอาเฉพาะแถวที่ยังไม่มีใน DB

    [FIX 1] today_str ใช้ THAI_TZ เพื่อให้ date boundary ตรงกับตลาดไทย
    """
    prices           = df["Close"].astype(float)
    rsi_series       = calc_rsi(prices, 14)
    macd_line, sig_line = calc_macd(prices)

    records  = []
    # [FIX 1] ใช้ THAI_TZ ไม่ใช่ UTC — ป้องกันการข้ามวันบน UTC server
    today_str = datetime.now(THAI_TZ).strftime("%Y-%m-%d")

    for idx, (date_idx, row) in enumerate(df.iterrows()):
        date_str = date_idx.strftime("%Y-%m-%d")
        if last_date_in_db and date_str <= last_date_in_db:
            continue
        if date_str > today_str:
            continue

        records.append({
            "symbol":      symbol,
            "date":        date_str,
            "open_price":  safe_float(row.get("Open")),
            "high_price":  safe_float(row.get("High")),
            "low_price":   safe_float(row.get("Low")),
            "close_price": safe_float(row.get("Close")),
            "volume":      safe_int(row.get("Volume")),
            "rsi_14":      safe_float(rsi_series.iloc[idx]),
            "macd_val":    safe_float(macd_line.iloc[idx]),
            "macd_signal": safe_float(sig_line.iloc[idx]),
            # backup_at บันทึกเป็น UTC (มาตรฐาน) ✅
            "backup_at":   datetime.now(timezone.utc).isoformat(),
        })

    return records

# ─────────────────────────────────────────────
# [FIX 5] UPDATE PREDICTION ACCURACY — APE Formula
# ─────────────────────────────────────────────
def update_prediction_accuracy(actual_map: dict[str, tuple[str, float]]):
    """
    อัปเดต actual_price และ error_percent ใน stock_predictions_v3

    [FIX 5] error_percent = |predicted - actual| / actual × 100
            เป็น Absolute Percentage Error (APE) ไม่มี sign
            ใช้ค่านี้คำนวณ MAPE ในภายหลังได้ทันที
    """
    if not actual_map:
        return

    symbols   = list(actual_map.keys())
    date_list = list({v[0] for v in actual_map.values()})

    try:
        res = retry_call(
            lambda: supabase.table("stock_predictions_v3")
                .select("id, symbol, prediction_date, predicted_price, horizon_type")
                .in_("symbol", symbols)
                .in_("prediction_date", date_list)
                .is_("actual_price", "null")
                .execute(),
            label="fetch_pending_predictions",
        )

        if not res.data:
            return

        updates = []
        for pred in res.data:
            sym  = pred["symbol"]
            date = pred["prediction_date"]
            if sym not in actual_map or actual_map[sym][0] != date:
                continue

            actual    = actual_map[sym][1]
            predicted = float(pred["predicted_price"])

            # [FIX 5] APE — Absolute Percentage Error (ไม่มี sign, ใช้คำนวณ MAPE ได้)
            error_pct = (abs(predicted - actual) / actual * 100) if actual != 0 else 0.0

            updates.append({
                "id":              pred["id"],
                # ✅ ส่ง NOT NULL columns กลับไปด้วยเพื่อผ่าน DB constraint
                "symbol":          sym,
                "prediction_date": date,
                "actual_price":    round(actual, 2),
                "error_percent":   round(error_pct, 4),
            })

        if updates:
            retry_call(
                lambda: supabase.table("stock_predictions_v3")
                    .upsert(updates, on_conflict="id")
                    .execute(),
                label="upsert_prediction_accuracy",
            )
            log.info("🎯 อัปเดต %d prediction records (APE formula)", len(updates))

    except Exception as e:
        log.error("❌ update_prediction_accuracy error: %s", e)

# ─────────────────────────────────────────────
# MAIN EOD ARCHIVE
# ─────────────────────────────────────────────
def sync_eod_to_historical(override_start: Optional[str] = None):
    """
    Args:
        override_start: ถ้า set จะดึงตั้งแต่วันนี้ override ค่าใน DB
                        ใช้สำหรับ --backfill mode
    """
    start_time = time.time()
    log.info("=" * 60)
    log.info("🚀 Trinity EOD Archive | %s",
             datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M:%S"))
    log.info("🔖 Batch Process Started | symbols=%d", len(SET100_SYMBOLS))
    log.info("=" * 60)

    latest_dates = {} if override_start else get_missing_dates_per_symbol(SET100_SYMBOLS)

    total_new_rows = 0
    total_fail     = 0
    actual_map: dict[str, tuple[str, float]] = {}

    batches = [SET100_SYMBOLS[i:i+BATCH_SIZE]
               for i in range(0, len(SET100_SYMBOLS), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, 1):
        log.info("📦 Batch %d/%d: %s", batch_idx, len(batches), ', '.join(batch))

        if override_start:
            earliest_start = override_start
        else:
            start_dates    = [calc_fetch_start(latest_dates.get(sym)) for sym in batch]
            earliest_start = min(start_dates)

        data_map     = download_batch(batch, start_date=earliest_start)
        batch_records: list[dict] = []

        for symbol in batch:
            if symbol not in data_map:
                log.warning("⚠️ %s | No Data from yfinance", symbol)
                total_fail += 1
                continue

            try:
                df         = data_map[symbol]
                last_in_db = None if override_start else latest_dates.get(symbol)
                records    = build_historical_records(symbol, df, last_in_db)

                if not records:
                    log.info("⏭️  %s | ข้อมูลเป็นปัจจุบันแล้ว", symbol)
                    continue

                batch_records.extend(records)
                latest_rec = records[-1]
                actual_map[symbol] = (latest_rec["date"], latest_rec["close_price"])
                total_new_rows    += len(records)
                log.info("✅ Symbol %s Updated | +%d rows | ฿%.2f",
                         symbol, len(records), latest_rec["close_price"])

            except Exception as e:
                log.error("❌ Error at %s: %s", symbol, e, exc_info=True)
                total_fail += 1

        # [FIX 2] Bulk Upsert พร้อม retry
        if batch_records:
            try:
                retry_call(
                    lambda recs=batch_records: supabase.table("stock_historical")
                        .upsert(recs, on_conflict="symbol,date")
                        .execute(),
                    label=f"bulk_upsert_batch_{batch_idx}",
                )
                log.info("💾 Batch %d: upserted %d rows", batch_idx, len(batch_records))
            except Exception as e:
                log.error("❌ Batch %d bulk upsert failed, trying fallback: %s",
                          batch_idx, e)
                for rec in batch_records:
                    try:
                        supabase.table("stock_historical")\
                            .upsert(rec, on_conflict="symbol,date")\
                            .execute()
                    except Exception as e2:
                        log.error("❌ Fallback upsert %s %s: %s",
                                  rec["symbol"], rec["date"], e2)

        if batch_idx < len(batches):
            time.sleep(BATCH_SLEEP)

    # อัปเดต Prediction Accuracy (APE)
    log.info("🔄 อัปเดต Prediction Accuracy สำหรับ %d หุ้น...", len(actual_map))
    update_prediction_accuracy(actual_map)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("🏁 Batch Process Completed in %.1fs", elapsed)
    log.info("   📊 New rows archived : %d", total_new_rows)
    log.info("   ❌ Failed symbols    : %d", total_fail)
    log.info("=" * 60)


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
def run_eod_scheduler():
    """ตั้ง schedule ให้รัน EOD archive เวลา 17:05 น. ทุกวันจันทร์-ศุกร์"""
    import schedule

    def is_weekday():
        return datetime.now(THAI_TZ).weekday() < 5

    def eod_job():
        if is_weekday():
            log.info("⏰ Triggered EOD Archive (17:05 scheduler)")
            sync_eod_to_historical()
        else:
            log.info("⏸️  วันหยุด — ข้าม EOD Archive")

    schedule.every().day.at("17:05").do(eod_job)
    log.info("⏰ EOD Scheduler เริ่มทำงาน — archive ทุกวันที่ 17:05 น.")

    now_thai = datetime.now(THAI_TZ)
    if now_thai.hour >= 17 or now_thai.hour < 9:
        log.info("🔄 ตรวจสอบ backfill ที่อาจขาด...")
        sync_eod_to_historical()

    while True:
        schedule.run_pending()
        time.sleep(30)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trinity EOD Archive")
    parser.add_argument("--schedule", action="store_true",
                        help="รันแบบ scheduler อัตโนมัติ")
    parser.add_argument("--backfill", type=int, default=0, metavar="DAYS",
                        help="Force backfill ย้อนหลัง N วัน")
    args = parser.parse_args()

    if args.backfill > 0:
        # [FIX 4] แก้ backfill mode ที่มีแค่ pass placeholder เดิม
        log.info("🔁 Force Backfill mode: %d วัน", args.backfill)
        # [FIX 1] คำนวณ start_override จาก THAI_TZ
        start_override = (
            datetime.now(THAI_TZ) - timedelta(days=args.backfill)
        ).strftime("%Y-%m-%d")
        log.info("📅 Backfill start date: %s", start_override)
        sync_eod_to_historical(override_start=start_override)
    elif args.schedule:
        run_eod_scheduler()
    else:
        sync_eod_to_historical()

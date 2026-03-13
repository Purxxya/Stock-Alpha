import gc
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from dotenv import load_dotenv
from huggingface_hub import login
from pandas.tseries.offsets import CustomBusinessDay

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# [FIX 4] PATH & ENV — Absolute path เสมอ
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ─────────────────────────────────────────────
# [FIX 3] ENTERPRISE LOGGING — Console + File (Absolute Path)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            # [FIX 3] เดิมใช้ "forecast_v3.log" (relative) — แก้เป็น absolute path
            os.path.join(LOG_DIR, "forecast_v3.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("trinity.forecast")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
try:
    from config import get_supabase, HF_TOKEN, SET100_SYMBOLS
except ImportError as exc:
    log.critical("❌ config.py not found or incomplete: %s", exc)
    sys.exit(1)

THAI_TZ          = ZoneInfo("Asia/Bangkok")   # [FIX 1] Named timezone
MODEL_ID         = "amazon/chronos-t5-base"
MAX_HORIZON      = 30
HORIZONS         = [7, 15, 30]
CONTEXT_LENGTH   = 128
MIN_DATA_ROWS    = CONTEXT_LENGTH
NUM_SAMPLES      = 20
UPSERT_BATCH_SZ  = 50
SLEEP_BETWEEN    = 0.1
MAX_RETRY        = 3

# ─────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────
def detect_device() -> str:
    if torch.backends.mps.is_available():
        log.info("🍎 Device: Apple MPS")
        return "mps"
    if torch.cuda.is_available():
        log.info("⚡ Device: CUDA (%s)", torch.cuda.get_device_name(0))
        return "cuda"
    log.warning("🐌 Device: CPU — inference จะช้ามาก")
    return "cpu"


# ─────────────────────────────────────────────
# 1. DATE MANAGEMENT
# ─────────────────────────────────────────────
def get_thai_forecast_dates(
    forecast_length: int = MAX_HORIZON,
) -> pd.DatetimeIndex:
    now_thai = datetime.now(THAI_TZ)
    th_holidays = holidays.Thailand(
        years=[now_thai.year, now_thai.year + 1]
    )
    thai_bday = CustomBusinessDay(holidays=th_holidays)
    
    # [จุดที่ต้องแก้] ตัดเวลาออกให้เหลือแต่วันที่
    today = now_thai.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    
    # [จุดที่ต้องแก้] เปลี่ยน start จาก (today + thai_bday) เป็น (today) 
    # เพื่อให้พยากรณ์เริ่มตั้งแต่วันนี้ กราฟจะเชื่อมกับราคาปัจจุบันทันที
    dates = pd.date_range(
        start=today, 
        periods=forecast_length,
        freq=thai_bday,
    )
    
    if len(dates) < forecast_length:
        raise RuntimeError(
            f"ได้วันทำการแค่ {len(dates)} วัน แต่ต้องการ {forecast_length}"
        )
    return dates


# ─────────────────────────────────────────────
# 2. SUPABASE HELPER (retry — existing, path label improved)
# ─────────────────────────────────────────────
def supabase_upsert_with_retry(
    supabase,
    table: str,
    records: list[dict],
    conflict_cols: str,
    max_retries: int = MAX_RETRY,
) -> bool:
    """Upsert พร้อม exponential-backoff retry"""
    for start in range(0, len(records), UPSERT_BATCH_SZ):
        batch = records[start: start + UPSERT_BATCH_SZ]
        for attempt in range(1, max_retries + 1):
            try:
                supabase.table(table).upsert(
                    batch, on_conflict=conflict_cols
                ).execute()
                break
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(
                    "Supabase upsert attempt %d/%d failed: %s — retry in %ds",
                    attempt, max_retries, exc, wait,
                )
                if attempt == max_retries:
                    log.error("❌ Batch upsert ล้มเหลวถาวร: %s", exc)
                    return False
                time.sleep(wait)
    return True


# ─────────────────────────────────────────────
# 3. EVALUATION — SYNC ACTUAL PRICES
# ─────────────────────────────────────────────
def sync_actual_prices_and_evaluate_v3(supabase) -> None:
    log.info("🔄 [Step 1] Syncing actual prices & evaluating accuracy...")

    # [FIX 1] ใช้ THAI_TZ สำหรับ today_str — ป้องกัน off-by-one บน UTC server
    today_str = datetime.now(THAI_TZ).strftime("%Y-%m-%d")   # ✅

    res = (
        supabase.table("stock_predictions_v3")
        .select("id, symbol, prediction_date, predicted_price")
        .is_("actual_price", "null")
        .lte("prediction_date", today_str)
        .execute()
    )
    if not res.data:
        log.info("✅ All predictions are already evaluated.")
        return

    from collections import defaultdict
    by_symbol: dict[str, list] = defaultdict(list)
    for item in res.data:
        by_symbol[item["symbol"]].append(item)

    updated_total = 0
    for symbol, items in by_symbol.items():
        dates_needed = [i["prediction_date"] for i in items]
        hist_res = (
            supabase.table("stock_historical")
            .select("date, close_price")
            .eq("symbol", symbol)
            .in_("date", dates_needed)
            .execute()
        )
        actual_map = {
            h["date"]: float(h["close_price"])
            for h in hist_res.data
        }

        updates = []
        for item in items:
            actual = actual_map.get(item["prediction_date"])
            if actual is None:
                continue
            predicted = float(item["predicted_price"])
            mape = (
                round(abs(actual - predicted) / actual * 100, 4)
                if actual != 0 else None
            )
            updates.append({
                "id":           item["id"],
                "actual_price": actual,
                "error_percent": mape,
            })

        for upd in updates:
            try:
                supabase.table("stock_predictions_v3").update({
                    "actual_price":  upd["actual_price"],
                    "error_percent": upd["error_percent"],
                }).eq("id", upd["id"]).execute()
                updated_total += 1
            except Exception as exc:
                log.warning("⚠️ Update id=%s failed: %s", upd["id"], exc)

    log.info("✅ Evaluation complete — updated %d rows.", updated_total)


# ─────────────────────────────────────────────
# 4. CLEANUP OLD PREDICTIONS
# ─────────────────────────────────────────────
def cleanup_stale_predictions(
    supabase, days_to_keep: int = 45
) -> None:
    from datetime import timedelta
    # [FIX 1] ใช้ THAI_TZ สำหรับ cutoff date
    cutoff = (
        datetime.now(THAI_TZ) - timedelta(days=days_to_keep)
    ).strftime("%Y-%m-%d")
    try:
        res = (
            supabase.table("stock_predictions_v3")
            .delete()
            .lt("prediction_date", cutoff)
            .execute()
        )
        count = len(res.data) if res.data else "unknown"
        log.info("🧹 Cleanup: removed %s stale rows (before %s).", count, cutoff)
    except Exception as exc:
        log.warning("⚠️ Cleanup failed: %s", exc)


# ─────────────────────────────────────────────
# 5. CORE FORECASTING
# ─────────────────────────────────────────────
def prepare_and_forecast_v3(
    symbol: str,
    pipeline: "ChronosPipeline",
    forecast_dates: pd.DatetimeIndex,
    device: str,
) -> Optional[list[dict]]:
    """ดึงข้อมูล stock_historical → Chronos inference → สร้าง records"""
    res = (
        supabase.table("stock_historical")
        .select("date, close_price")
        .eq("symbol", symbol)
        .order("date", desc=True)
        .limit(250)
        .execute()
    )
    if not res.data or len(res.data) < MIN_DATA_ROWS:
        log.warning("⚠️ %s: ข้อมูลไม่พอ (%d rows, ต้องการ %d)",
                    symbol, len(res.data or []), MIN_DATA_ROWS)
        return None

    df = (
        pd.DataFrame(res.data)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    prices = df["close_price"].values.astype(np.float32)

    context_prices = prices[-CONTEXT_LENGTH:]
    if np.any(np.isnan(context_prices)) or np.any(context_prices <= 0):
        nan_count = int(np.sum(np.isnan(context_prices)))
        log.warning("⚠️ %s: พบ NaN/ค่าติดลบ %d ค่า — ข้ามหุ้นนี้",
                    symbol, nan_count)
        return None

    context_tensor = torch.tensor(
        context_prices, dtype=torch.float32
    ).unsqueeze(0)

    with torch.inference_mode():
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=MAX_HORIZON,
            num_samples=NUM_SAMPLES,
        )

    forecast_np = forecast[0].cpu().numpy()
    low, median, high = np.quantile(forecast_np, [0.10, 0.50, 0.90], axis=0)

    # [FIX 1] created_at ใช้ UTC มาตรฐาน ✅ (timestamp บันทึกเป็น UTC)
    now_iso = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for h in HORIZONS:
        if h > len(forecast_dates):
            log.warning("forecast_dates มีแค่ %d วัน ไม่พอสำหรับ %dD",
                        len(forecast_dates), h)
            continue
        for i in range(h):
            records.append({
                "symbol":          symbol,
                "prediction_date": forecast_dates[i].strftime("%Y-%m-%d"),
                "predicted_price": round(float(median[i]), 2),
                "lower_bound":     round(float(low[i]), 2),
                "upper_bound":     round(float(high[i]), 2),
                "horizon_type":    f"{h}D",
                "day_index":       i + 1,
                "model_name":      MODEL_ID,
                "created_at":      now_iso,
            })

    return records


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
def main(symbols: list[str] = None, dry_run: bool = False) -> None:
    log.info("═" * 60)
    # [FIX 1] datetime.now() → datetime.now(THAI_TZ)
    log.info("🚀 Trinity Forecasting Engine V3 | %s",
             datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M"))
    if dry_run:
        log.warning("⚠️  DRY-RUN MODE — ไม่มีการเขียนลง DB")
    log.info("🔖 Batch Process Started | symbols=%d",
             len(symbols or SET100_SYMBOLS))
    log.info("═" * 60)

    # Auth
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        log.info("🔑 HuggingFace authenticated")
    except Exception as exc:
        log.critical("❌ HF login failed: %s", exc)
        sys.exit(1)

    # Supabase
    global supabase
    try:
        supabase = get_supabase()
    except Exception as exc:
        log.critical("❌ Supabase connection failed: %s", exc)
        sys.exit(1)

    # Step 1: Evaluate pending predictions
    sync_actual_prices_and_evaluate_v3(supabase)

    # Step 2: Cleanup stale rows
    cleanup_stale_predictions(supabase, days_to_keep=45)

    # Step 3: Forecast dates
    forecast_dates = get_thai_forecast_dates(MAX_HORIZON)
    log.info("🗓️  Forecast window: %s → %s",
             forecast_dates[0].date(), forecast_dates[-1].date())

    # Step 4: Load model
    device = detect_device()
    log.info("📦 Loading model: %s ...", MODEL_ID)
    try:
        pipeline = ChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    except Exception as exc:
        log.critical("❌ Model load failed: %s", exc)
        sys.exit(1)

    # Step 5: Forecast loop
    target_symbols = symbols or SET100_SYMBOLS
    total = len(target_symbols)
    success, skipped, failed = 0, 0, 0

    for idx, symbol in enumerate(target_symbols, 1):
        log.info("[%d/%d] 🔮 %s", idx, total, symbol)
        try:
            records = prepare_and_forecast_v3(
                symbol, pipeline, forecast_dates, device)

            if not records:
                skipped += 1
                continue

            if not dry_run:
                ok = supabase_upsert_with_retry(
                    supabase,
                    table="stock_predictions_v3",
                    records=records,
                    conflict_cols="symbol, prediction_date, horizon_type, model_name",
                )
                if not ok:
                    failed += 1
                    continue

            log.info("    ✅ %s → %d records (7D/15D/30D) %s",
                     symbol, len(records),
                     "[DRY RUN]" if dry_run else "saved")
            success += 1

        except Exception as exc:
            log.exception("    ❌ %s: unexpected error — %s", symbol, exc)
            failed += 1

        finally:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        time.sleep(SLEEP_BETWEEN)

    log.info("═" * 60)
    log.info(
        "🏁 Batch Process Completed │ ✅ %d │ ⏭️ %d │ ❌ %d │ Total: %d",
        success, skipped, failed, total,
    )
    log.info("═" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trinity Forecast Engine V3")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="รันเฉพาะหุ้นที่ระบุ เช่น --symbols PTT KBANK SCB")
    parser.add_argument("--dry-run", action="store_true",
                        help="คำนวณแต่ไม่เขียนลง Database")
    args = parser.parse_args()
    main(symbols=args.symbols, dry_run=args.dry_run)

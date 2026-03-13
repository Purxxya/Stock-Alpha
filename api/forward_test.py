import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# [FIX 4] PATH & ENV — Absolute path เสมอ
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ─────────────────────────────────────────────
# [FIX 3] ENTERPRISE LOGGING — Console + File
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "forward_test.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.ForwardTest")

from config import get_supabase, SET100_SYMBOLS  # type: ignore

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
supabase  = get_supabase()
THAI_TZ   = ZoneInfo("Asia/Bangkok")
MAX_RETRY = 3
HORIZONS  = ["7D", "15D", "30D"]

# [FIX 1] TODAY_STR ใช้ THAI_TZ — ป้องกัน off-by-one เมื่อรันบน UTC server
# ตัวอย่าง: เที่ยงคืนไทย = 17:00 UTC วันก่อน → UTC.date() จะได้วันที่ผิด
TODAY_STR = datetime.now(THAI_TZ).date().isoformat()   # ✅ Bangkok date

# ─────────────────────────────────────────────
# [FIX 2] RETRY HELPER
# ─────────────────────────────────────────────
def retry_call(fn, *args, max_retries: int = MAX_RETRY,
               label: str = "call", **kwargs):
    """Generic retry wrapper พร้อม exponential backoff"""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            wait = 2 ** attempt
            if attempt == max_retries:
                log.error("❌ [%s] ล้มเหลวทุก %d retry: %s",
                          label, max_retries, exc)
                raise
            log.warning("⚠️ [%s] attempt %d/%d: %s — retry in %ds",
                        label, attempt, max_retries, exc, wait)
            time.sleep(wait)


# ─────────────────────────────────────────────
# STEP 1: ดึง Predictions ที่ถึงวันกำหนดแล้ว
# ─────────────────────────────────────────────
def fetch_pending_predictions(symbols: list[str]) -> list[dict]:
    """ดึง prediction rows ที่ prediction_date ≤ วันนี้ และ actual_price IS NULL"""
    try:
        res = retry_call(
            lambda: supabase.table("stock_predictions_v3")
                .select("id,symbol,prediction_date,predicted_price,horizon_type")
                .in_("symbol", symbols)
                .lte("prediction_date", TODAY_STR)
                .is_("actual_price", "null")
                .execute(),
            label="fetch_pending_predictions",
        )
        rows = res.data or []
        log.info("📋 พบ %d prediction rows ที่รอ actual_price", len(rows))
        return rows
    except Exception as e:
        log.error("❌ fetch_pending_predictions: %s", e)
        return []


# ─────────────────────────────────────────────
# STEP 2: ดึงราคาจริงจาก stock_historical
# ─────────────────────────────────────────────
def fetch_actual_prices(
    symbols: list[str],
    dates: list[str],
) -> dict[tuple[str, str], float]:
    
    if not symbols or not dates:
        return {}
    try:
        res = retry_call(
            lambda: supabase.table("stock_historical")
                .select("symbol,date,close_price")
                .in_("symbol", symbols)
                .in_("date", dates)
                .execute(),
            label="fetch_actual_prices",
        )
        mapping: dict[tuple[str, str], float] = {}
        for row in res.data or []:
            if row.get("close_price"):
                mapping[(row["symbol"], row["date"])] = float(row["close_price"])
        log.info("📊 พบราคาจริง %d คู่ (symbol, date)", len(mapping))
        return mapping
    except Exception as e:
        log.error("❌ fetch_actual_prices: %s", e)
        return {}


# ─────────────────────────────────────────────
# STEP 3: คำนวณ APE และ upsert กลับ
# ─────────────────────────────────────────────
def update_actuals(
    pending: list[dict],
    actual_map: dict[tuple[str, str], float],
) -> int:
    total_saved = 0
    skipped = 0

    log.info("🎯 เริ่มต้นคำนวณ Error (%) รายตัว...")

    for pred in pending:
        symbol = pred["symbol"]
        pred_date = pred["prediction_date"]
        pred_id = pred["id"]
        
        key    = (symbol, pred_date)
        actual = actual_map.get(key)

        if actual is None or actual <= 0:
            skipped += 1
            continue

        try:
            predicted = float(pred["predicted_price"])
            
            # [CALCULATION FIX] คำนวณ Error เป็นเปอร์เซ็นต์ (APE)
            # สูตร: (|Predicted - Actual| / Actual) * 100
            diff = abs(predicted - actual)
            error_val = (diff / actual) * 100
            
            # ปัดเศษเป็น 2 ตำแหน่งเพื่อให้เป็นมาตรฐาน % (เช่น 3.64)
            error_pct = round(error_val, 2)

            # แสดง Log ให้เห็นชัดเจนว่า Pred vs Real คือเท่าไหร่ และ Error กี่ %
            log.info(f"📊 {symbol: <7} | Date: {pred_date} | Pred: {predicted:8.2f} | Real: {actual:8.2f} | Error: {error_pct:6.2f}%")

            retry_call(
                lambda: supabase.table("stock_predictions_v3")
                    .update({
                        "actual_price": round(actual, 4),
                        "error_percent": error_pct  # บันทึกค่าที่ปัดเศษแล้วลง DB
                    })
                    .eq("id", pred_id)
                    .execute(),
                label=f"update_{symbol}_{pred_id}"
            )
            total_saved += 1
            
        except Exception as e:
            log.error(f"  ❌ {symbol} (ID: {pred_id}) เกิดข้อผิดพลาด: {e}")

    log.info("✅ อัปเดตสำเร็จทั้งหมด %d rows (ข้าม %d รายการ)", total_saved, skipped)
    return total_saved

# ─────────────────────────────────────────────
# STEP 4: สร้าง Accuracy Report
# ─────────────────────────────────────────────
def generate_accuracy_report(symbols: list[str]) -> pd.DataFrame:
    try:
        res = retry_call(
            lambda: supabase.table("stock_predictions_v3")
                .select("symbol,horizon_type,error_percent")
                .in_("symbol", symbols)
                .not_.is_("actual_price", "null")
                .not_.is_("error_percent", "null")
                .execute(),
            label="fetch_accuracy_report",
        )
        rows = res.data or []
    except Exception as e:
        log.error("❌ generate_accuracy_report: %s", e)
        return pd.DataFrame()

    if not rows:
        log.warning("⚠️ ยังไม่มีข้อมูล actual_price (ต้องรอให้ historical.py รันก่อน)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # error_percent คือ APE แล้ว → ใช้ mean() ได้เลยเพื่อ MAPE
    df["abs_error"] = df["error_percent"].abs()

    report = (
        df.groupby(["symbol", "horizon_type"])["abs_error"]
        .agg(
            n_samples="count",
            mape_mean="mean",
            mape_median="median",
            mape_min="min",
            mape_max="max",
        )
        .reset_index()
        .sort_values(["horizon_type", "mape_mean"])
    )
    report.columns = ["Symbol", "Horizon", "Samples",
                      "MAPE Mean%", "MAPE Median%", "Min%", "Max%"]
    return report.round(2)


def print_report(df: pd.DataFrame):
    if df.empty:
        return

    log.info("=" * 80)
    log.info("📊  CHRONOS WALK-FORWARD VALIDATION REPORT")
    log.info("    วันที่รายงาน: %s", TODAY_STR)
    log.info("=" * 80)

    for horizon in HORIZONS:
        chunk = df[df["Horizon"] == horizon]
        if chunk.empty:
            continue
        log.info("🎯  Horizon: %s  (%d หุ้น มีข้อมูล)", horizon, len(chunk))
        # ใช้ log แทน print ทั้งหมด
        for line in chunk.to_string(index=False).split("\n"):
            log.info("    %s", line)

    overall = (
        df.groupby("Symbol")["MAPE Mean%"]
        .mean()
        .reset_index()
        .sort_values("MAPE Mean%")
        .head(10)
    )
    log.info("🏆  Top 10 หุ้นที่โมเดลแม่นสุด (MAPE เฉลี่ยทุก Horizon):")
    for line in overall.to_string(index=False).split("\n"):
        log.info("    %s", line)

    summary = df.groupby("Horizon")["MAPE Mean%"].mean().round(2)
    log.info("📈  สรุป MAPE เฉลี่ยต่อ Horizon:")
    for h, v in summary.items():
        bar = "█" * int(v / 2) if v > 0 else "" # ป้องกัน error ถ้า v เป็น 0
    # เพิ่มเครื่องหมาย % เข้าไปใน f-string
        log.info(f"    {h:>5}: {v:>6.2f}%  {bar}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_forward_test(symbols: list[str], report_only: bool = False):
    log.info("=" * 60)
    log.info("🚀 Trinity Forward Test | %s", TODAY_STR)
    log.info("🔖 Batch Process Started | symbols=%d", len(symbols))
    log.info("=" * 60)

    if not report_only:
        # Phase 1: อัปเดต actual_price
        pending = fetch_pending_predictions(symbols)

        if pending:
            unique_dates   = list({p["prediction_date"] for p in pending})
            unique_symbols = list({p["symbol"] for p in pending})
            actual_map     = fetch_actual_prices(unique_symbols, unique_dates)
            update_actuals(pending, actual_map)
        else:
            log.info("⏭️  ไม่มี prediction rows ที่รอ actual_price วันนี้")

    # Phase 2: Accuracy Report
    log.info("📊 กำลังสร้าง Accuracy Report...")
    report_df = generate_accuracy_report(symbols)
    print_report(report_df)

    log.info("✨ Forward Test เสร็จสมบูรณ์")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trinity Walk-Forward Validation")
    parser.add_argument("--report-only", action="store_true",
                        help="แสดงรายงานอย่างเดียว ไม่อัปเดต DB")
    parser.add_argument("--symbol", type=str, default=None,
                        help="รันเฉพาะหุ้นนี้ เช่น --symbol PTT")
    args = parser.parse_args()

    target_symbols = [args.symbol.upper()] if args.symbol else SET100_SYMBOLS
    run_forward_test(symbols=target_symbols, report_only=args.report_only)

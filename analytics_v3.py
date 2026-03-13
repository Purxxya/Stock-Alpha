import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

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
            os.path.join(LOG_DIR, "analytics.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.Analytics")

from config import get_supabase, OPENAI_API_KEY, SET100_SYMBOLS  # type: ignore

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
supabase      = get_supabase()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL     = "gpt-4o"
THAI_TZ       = ZoneInfo("Asia/Bangkok")   # [FIX 1]
MAX_RETRY     = 3

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
# INDICATOR CALCULATIONS (Quant Standard)
# ─────────────────────────────────────────────
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI มาตรฐาน Wilder Smoothing"""
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_macd(prices: pd.Series, fast: int = 12,
              slow: int = 26, signal: int = 9):
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_crossover(macd: pd.Series, signal: pd.Series) -> str:
    if len(macd) < 2:
        return "Neutral"
    prev_diff = macd.iloc[-2] - signal.iloc[-2]
    curr_diff = macd.iloc[-1] - signal.iloc[-1]
    if   prev_diff <= 0 and curr_diff > 0:  return "Golden Cross (Buy Signal)"
    elif prev_diff >= 0 and curr_diff < 0:  return "Death Cross (Sell Signal)"
    elif curr_diff > 0:                     return "Bullish (MACD above Signal)"
    else:                                   return "Bearish (MACD below Signal)"


def calc_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period, min_periods=period).mean()

# ─────────────────────────────────────────────
# DATA FETCHING (Batch)
# ─────────────────────────────────────────────
def fetch_historical_batch(
    symbols: list[str], days: int = 120
) -> dict[str, pd.DataFrame]:
    log.info("📡 Fetching historical data for %d symbols (%d days)...",
             len(symbols), days)
    # [FIX 1] ใช้ THAI_TZ สำหรับ cutoff date
    cutoff = (datetime.now(THAI_TZ) - timedelta(days=days)).date().isoformat()

    try:
        res = retry_call(
            lambda: supabase.table("stock_historical")
                .select("symbol,date,close_price,volume,rsi_14,macd_val,macd_signal")
                .in_("symbol", symbols)
                .gte("date", cutoff)
                .order("date", desc=False)
                .execute(),
            label="fetch_historical_batch",
        )
        if not res.data:
            log.warning("⚠️ ไม่พบข้อมูล historical")
            return {}

        df_all = pd.DataFrame(res.data)
        df_all["date"]        = pd.to_datetime(df_all["date"])
        df_all["close_price"] = pd.to_numeric(df_all["close_price"], errors="coerce")

        result = {}
        for sym, df_sym in df_all.groupby("symbol"):
            result[sym] = df_sym.sort_values("date").reset_index(drop=True)

        log.info("✅ Loaded %d symbols from DB", len(result))
        return result

    except Exception as e:
        log.error("❌ fetch_historical_batch error: %s", e)
        return {}


def fetch_predictions_batch(
    symbols: list[str],
) -> dict[str, dict]:
    """ดึงผลพยากรณ์ Chronos ล่าสุด (7D, 15D, 30D)"""
    log.info("🔮 Fetching Chronos predictions for %d symbols...", len(symbols))
    try:
        # [FIX 1] ใช้ THAI_TZ สำหรับ tomorrow
        tomorrow = (datetime.now(THAI_TZ) + timedelta(days=1)).date().isoformat()
        res = retry_call(
            lambda: supabase.table("stock_predictions_v3")
                .select("symbol,horizon_type,predicted_price,"
                        "lower_bound,upper_bound,prediction_date")
                .in_("symbol", symbols)
                .gte("prediction_date", tomorrow)
                .order("prediction_date", desc=False)
                .execute(),
            label="fetch_predictions_batch",
        )
        if not res.data:
            return {}

        result: dict[str, dict] = {}
        for row in res.data:
            sym = row["symbol"]
            hz  = row["horizon_type"]
            if sym not in result:
                result[sym] = {}
            if hz not in result[sym]:
                result[sym][hz] = row
        return result

    except Exception as e:
        log.error("❌ fetch_predictions_batch error: %s", e)
        return {}

# ─────────────────────────────────────────────
# MAPE EVALUATION
# ─────────────────────────────────────────────
def fetch_evaluation_data() -> pd.DataFrame | None:
    log.info("📊 Fetching completed predictions for MAPE evaluation...")
    try:
        res = retry_call(
            lambda: supabase.table("stock_predictions_v3")
                .select("symbol,horizon_type,day_index,"
                        "predicted_price,actual_price,error_percent")
                .not_("actual_price", "null")
                .execute(),
            label="fetch_evaluation_data",
        )
        if not res.data:
            log.warning("⚠️ ยังไม่มีข้อมูล actual_price สำหรับประเมิน")
            return None

        df = pd.DataFrame(res.data)
        df["predicted_price"] = pd.to_numeric(
            df["predicted_price"], errors="coerce")
        df["actual_price"]    = pd.to_numeric(
            df["actual_price"], errors="coerce")
        return df.dropna(subset=["predicted_price","actual_price"])

    except Exception as e:
        log.error("❌ fetch_evaluation_data error: %s", e)
        return None


def calculate_mape_stats(df: pd.DataFrame) -> dict:
    """คำนวณ MAPE — ใช้ error_percent (APE) ที่บันทึกไว้แล้วจาก DB"""
    df = df.copy()
    # error_percent ใน DB คือ APE แล้ว (absolute) → ใช้ mean() ได้เลย
    df["abs_error_pct"] = (
        (df["actual_price"] - df["predicted_price"]).abs()
        / df["actual_price"]
    ) * 100

    horizon_summary  = df.groupby("horizon_type")["abs_error_pct"]\
        .mean().reset_index()
    horizon_summary.columns = ["Horizon", "MAPE (%)"]
    horizon_summary["MAPE (%)"] = horizon_summary["MAPE (%)"].round(2)

    decay_summary    = df.groupby("day_index")["abs_error_pct"].mean().reset_index()
    decay_summary.columns = ["Day Index", "MAPE (%)"]

    symbol_perf      = df.groupby("symbol")["abs_error_pct"].mean().sort_values()

    return {
        "horizon":       horizon_summary,
        "decay":         decay_summary,
        "best_stocks":   symbol_perf.head(5),
        "worst_stocks":  symbol_perf.tail(5),
        "overall_mape":  round(df["abs_error_pct"].mean(), 2),
        "total_samples": len(df),
    }

# ─────────────────────────────────────────────
# PER-SYMBOL INDICATOR ANALYSIS
# ─────────────────────────────────────────────
def analyze_symbol(
    symbol: str,
    df_hist: pd.DataFrame,
    predictions: dict,
) -> dict | None:
    if df_hist is None or len(df_hist) < 30:
        return None

    prices     = df_hist["close_price"]
    last_price = float(prices.iloc[-1])

    rsi    = calc_rsi(prices, 14)
    sma20  = calc_sma(prices, 20)
    sma50  = calc_sma(prices, 50)
    macd_line, signal_line, _ = calc_macd(prices)
    macd_signal_str = detect_crossover(macd_line, signal_line)

    sma_signal = "Neutral"
    if pd.notna(sma20.iloc[-1]) and pd.notna(sma50.iloc[-1]):
        if len(sma20) >= 2 and pd.notna(sma20.iloc[-2]) and pd.notna(sma50.iloc[-2]):
            if sma20.iloc[-2] <= sma50.iloc[-2] and sma20.iloc[-1] > sma50.iloc[-1]:
                sma_signal = "SMA Golden Cross (Bull)"
            elif sma20.iloc[-2] >= sma50.iloc[-2] and sma20.iloc[-1] < sma50.iloc[-1]:
                sma_signal = "SMA Death Cross (Bear)"
        sma_signal = (
            f"SMA20={sma20.iloc[-1]:.2f} vs SMA50={sma50.iloc[-1]:.2f} "
            f"| {sma_signal}"
        )

    fc_ctx = {}
    sym_preds = predictions.get(symbol, {})
    for hz in ["7D", "15D", "30D"]:
        if hz in sym_preds:
            p = sym_preds[hz]
            change_pct = (
                (float(p["predicted_price"]) - last_price) / last_price * 100
            )
            fc_ctx[hz] = {
                "target":     float(p["predicted_price"]),
                "low":        float(p["lower_bound"]),
                "high":       float(p["upper_bound"]),
                "change_pct": round(change_pct, 2),
            }

    return {
        "symbol":      symbol,
        "last_price":  round(last_price, 2),
        "rsi_14":      round(float(rsi.iloc[-1]), 2),
        "macd_signal": macd_signal_str,
        "sma_signal":  sma_signal,
        "forecast":    fc_ctx,
    }

# ─────────────────────────────────────────────
# AI STRATEGIC INSIGHT
# ─────────────────────────────────────────────
def generate_model_insight(stats: dict, top_analyses: list[dict]) -> str:
    top_summary = "\n".join([
        f"- {a['symbol']}: RSI={a['rsi_14']}, "
        f"{a['macd_signal']}, "
        f"Forecast7D={a['forecast'].get('7D',{}).get('change_pct','N/A')}%"
        for a in top_analyses[:10]
    ])

    prompt = f"""คุณคือ "Senior Quantitative Analyst" ผู้เชี่ยวชาญตลาดหุ้นไทย

[ผลการประเมินโมเดล Chronos AI – Forward Test]
MAPE รวม: {stats['overall_mape']}% (จาก {stats['total_samples']} จุดข้อมูล)

MAPE ตาม Horizon:
{stats['horizon'].to_string(index=False)}

หุ้นที่โมเดลแม่นที่สุด (Top 5):
{stats['best_stocks'].to_string()}

หุ้นที่โมเดลคลาดเคลื่อนสูงที่สุด (Worst 5):
{stats['worst_stocks'].to_string()}

[สรุป Technical + Chronos Forecast สำหรับ Top Stocks]
{top_summary}

[คำสั่งวิเคราะห์]
1. ประเมิน "ความเสถียรของโมเดล" เมื่อระยะเวลาพยากรณ์ยาวขึ้น (Model Decay Analysis)
2. แนะนำว่าควรใช้ Horizon ไหน (7D / 15D / 30D) ในการเทรดจริง พร้อมเหตุผลเชิง Quant
3. วิเคราะห์สาเหตุที่หุ้น Worst 5 มีความคลาดเคลื่อนสูง
4. ให้ Consensus View ว่า Technical + Chronos สอดคล้องหรือขัดแย้งกัน

สรุปเป็นข้อๆ กระชับ เป็นภาษาไทยมืออาชีพ พร้อม Actionable Recommendation"""

    try:
        # [FIX 2] retry สำหรับ OpenAI call
        resp = retry_call(
            lambda: openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800,
            ),
            label="openai_model_insight",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("❌ OpenAI insight error: %s", e)
        return f"ไม่สามารถสร้าง AI Insight ได้ในขณะนี้: {e}"

# ─────────────────────────────────────────────
# SAVE INSIGHT
# ─────────────────────────────────────────────
def save_insight(insight_text: str, stats: dict):
    try:
        retry_call(
            lambda: supabase.table("model_insights").insert({
                "insight_text":  insight_text,
                "overall_mape":  stats["overall_mape"],
                "total_samples": stats["total_samples"],
                # [FIX 1] created_at เป็น UTC มาตรฐาน ✅
                "created_at":    datetime.now(timezone.utc).isoformat(),
            }).execute(),
            label="save_insight",
        )
        log.info("✅ Saved AI Insight to model_insights table")
    except Exception as e:
        try:
            supabase.table("model_insights").insert({
                "insight_text": insight_text
            }).execute()
            log.info("✅ Saved AI Insight (basic mode)")
        except Exception as e2:
            log.error("❌ ไม่สามารถบันทึก insight: %s", e2)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_analytics():
    log.info("=" * 60)
    # [FIX 1] datetime.now() → datetime.now(THAI_TZ)
    log.info("🏁 Trinity Analytics V3 | %s",
             datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M:%S"))
    log.info("🔖 Batch Process Started")
    log.info("=" * 60)

    # Step 1: MAPE Evaluation
    df_eval = fetch_evaluation_data()

    if df_eval is None or df_eval.empty:
        log.warning("⚠️ ยังไม่มีข้อมูล actual_price เพียงพอ — ข้าม MAPE evaluation")
        stats = {
            "horizon":       pd.DataFrame({
                "Horizon": ["7D","15D","30D"],
                "MAPE (%)": ["N/A","N/A","N/A"],
            }),
            "best_stocks":   pd.Series(dtype=float),
            "worst_stocks":  pd.Series(dtype=float),
            "overall_mape":  0.0,
            "total_samples": 0,
        }
    else:
        stats = calculate_mape_stats(df_eval)
        log.info("📈 MAPE Summary:\n%s", stats["horizon"].to_string(index=False))
        log.info("🏆 Best 5:\n%s",  stats["best_stocks"].to_string())
        log.info("⚠️ Worst 5:\n%s", stats["worst_stocks"].to_string())

    # Step 2: Batch Fetch Historical
    hist_data   = fetch_historical_batch(SET100_SYMBOLS, days=120)
    predictions = fetch_predictions_batch(SET100_SYMBOLS)

    # Step 3: Per-Symbol Technical Analysis
    analyses = []
    for symbol in SET100_SYMBOLS:
        try:
            result = analyze_symbol(
                symbol, hist_data.get(symbol), predictions)
            if result:
                analyses.append(result)
        except Exception as e:
            log.warning("⚠️ analyze_symbol(%s) error: %s", symbol, e)

    log.info("✅ วิเคราะห์ Technical ครบ %d หุ้น", len(analyses))

    # Step 4: AI Strategic Insight
    log.info("🤖 Generating AI Strategic Insight...")
    insight = generate_model_insight(stats, analyses)
    log.info("=" * 60)
    log.info("💡 AI STRATEGIC INSIGHT")
    log.info("=" * 60)
    log.info(insight)
    log.info("=" * 60)

    # Step 5: Save to DB
    save_insight(insight, stats)

    log.info("🏁 Analytics V3 เสร็จสมบูรณ์!")


if __name__ == "__main__":
    run_analytics()
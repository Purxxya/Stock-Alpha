import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# [FIX 4] PATH & ENV — Absolute path เสมอ
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# [FIX 4] Script paths เป็น Absolute เพื่อให้รัน Cron ได้จากทุก directory
SCRIPTS = {
    "stock_sync":   os.path.join(BASE_DIR, "stock_sync.py"),
    "historical":   os.path.join(BASE_DIR, "historical.py"),
    "forward_test": os.path.join(BASE_DIR, "forward_test.py"),
    "forecast":     os.path.join(BASE_DIR, "model_forecast_v3.py"),
    "news":         os.path.join(BASE_DIR, "news.py"),
    "analytics":    os.path.join(BASE_DIR, "analytics_v3.py"),
}

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
            os.path.join(LOG_DIR, "pipeline.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.Orchestrator")

THAI_TZ = ZoneInfo("Asia/Bangkok")

# ─────────────────────────────────────────────
# MARKET STATUS
# ─────────────────────────────────────────────
def get_market_status() -> str:
    now  = datetime.now(THAI_TZ)
    wday = now.weekday()
    hm   = now.hour * 100 + now.minute
    if wday >= 5:              return "CLOSED"
    if hm < 930:               return "CLOSED"
    if 930  <= hm < 1000:      return "PRE_OPEN"
    if 1000 <= hm < 1230:      return "OPEN"
    if 1230 <= hm < 1400:      return "LUNCH"
    if 1400 <= hm < 1700:      return "OPEN"
    if 1700 <= hm < 1800:      return "POST_CLOSE"
    return "CLOSED"


# ─────────────────────────────────────────────
# SCRIPT RUNNER
# ─────────────────────────────────────────────
def run_script(script_key: str, args: list[str] = [],
               timeout: int = 1800) -> bool:
    """
    รัน script แล้ว return True ถ้าสำเร็จ
    [FIX 4] ใช้ script path จาก SCRIPTS dict (Absolute) ทั้งหมด
    """
    script_path = SCRIPTS.get(script_key, script_key)
    cmd   = [sys.executable, script_path] + args
    label = f"{os.path.basename(script_path)} {' '.join(args)}".strip()

    log.info("▶️  Running: %s", label)
    start = time.time()
    try:
        subprocess.run(cmd, timeout=timeout, check=True)
        elapsed = time.time() - start
        log.info("✅ %s เสร็จใน %.1fs", label, elapsed)
        return True
    except subprocess.TimeoutExpired:
        log.error("⏱️  %s timeout (%ds)", label, timeout)
        return False
    except subprocess.CalledProcessError as e:
        log.error("❌ %s failed (exit=%d)", label, e.returncode)
        return False
    except FileNotFoundError:
        log.error("❌ ไม่พบไฟล์: %s", script_path)
        return False


# ─────────────────────────────────────────────
# EOD SEQUENCE
# ─────────────────────────────────────────────
def run_eod_sequence():
    ts = datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 60)
    log.info("🌙 Trinity EOD Pipeline เริ่มต้น | %s", ts)
    log.info("🔖 Batch Process Started — EOD Sequence")
    log.info("=" * 60)

    results: dict[str, bool] = {}

    results["stock_sync"]   = run_script("stock_sync")
    results["historical"]   = run_script("historical")
    results["forward_test"] = run_script("forward_test")
    results["forecast"]     = run_script("forecast")
    results["news"]         = run_script("news", ["--mode", "full"])

    log.info("=" * 60)
    log.info("📋 EOD Pipeline Summary:")
    for script, ok in results.items():
        log.info("  %s %s", "✅" if ok else "❌", script)
    success_count = sum(results.values())
    log.info("🏁 Batch Process Completed — %d/%d scripts OK",
             success_count, len(results))
    log.info("=" * 60)


# ─────────────────────────────────────────────
# INTRADAY LOOP
# ─────────────────────────────────────────────
def run_intraday_loop():
    log.info("⏰ Intraday Loop เริ่มทำงาน (sync ทุก 60 วินาที)")
    while True:
        status = get_market_status()
        if status in ("OPEN", "PRE_OPEN"):
            run_script("stock_sync")
        elif status == "POST_CLOSE":
            log.info("🌙 ตลาดปิดแล้ว — สั่งรัน EOD sequence...")
            run_eod_sequence()
            time.sleep(3600)
            continue
        else:
            log.info("⏸️  ตลาด %s — รอรอบถัดไป", status)
        time.sleep(60)


# ─────────────────────────────────────────────
# FULL SCHEDULER (24/7)
# ─────────────────────────────────────────────
def run_full_schedule():
    """
    Auto schedule:
    - ช่วงตลาดเปิด: sync ทุก 1 นาที
    - 17:05 น.    : EOD sequence
    - 08:00 น.    : ดึงข่าวเช้า (pre-market sentiment)
    """
    import schedule as sched

    def intraday_job():
        status = get_market_status()
        if status in ("OPEN", "PRE_OPEN"):
            run_script("stock_sync")
        else:
            log.info("⏸️  Skip intraday (%s)", status)

    def eod_job():
        now = datetime.now(THAI_TZ)
        if now.weekday() < 5:
            run_eod_sequence()
        else:
            log.info("⏭️  Skip EOD — วันหยุด")

    def morning_news_job():
        now = datetime.now(THAI_TZ)
        if now.weekday() < 5:
            log.info("🌅 Morning News Job")
            run_script("news", ["--mode", "quick"])

    sched.every(1).minutes.do(intraday_job)
    sched.every().day.at("17:05").do(eod_job)
    sched.every().day.at("08:00").do(morning_news_job)

    log.info("⏰ Full Scheduler เริ่มทำงาน")
    log.info("  • ทุก 1 นาที  : intraday sync")
    log.info("  • 17:05 น.   : EOD pipeline")
    log.info("  • 08:00 น.   : Morning news")

    status = get_market_status()
    now    = datetime.now(THAI_TZ)
    if status in ("POST_CLOSE", "CLOSED") and now.weekday() < 5:
        log.info("🚀 รัน EOD sequence ทันทีเพราะตลาดปิดแล้ว")
        run_eod_sequence()

    while True:
        sched.run_pending()
        time.sleep(30)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trinity Master Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["eod", "intraday", "schedule", "news", "forecast", "sync"],
        default="eod",
        help=(
            "eod=EOD sequence ครั้งเดียว | "
            "intraday=loop realtime | "
            "schedule=Full auto 24/7 | "
            "news=news only | "
            "forecast=Chronos only | "
            "sync=realtime sync only"
        ),
    )
    args = parser.parse_args()

    MODE_MAP = {
        "eod":      run_eod_sequence, 
        "intraday": run_intraday_loop,
        "schedule": run_full_schedule,
        "news":     lambda: run_script("news", ["--mode", "full"]),
        "forecast": lambda: run_script("forecast"),
        "sync":     lambda: run_script("stock_sync"),
    }
    MODE_MAP[args.mode]()

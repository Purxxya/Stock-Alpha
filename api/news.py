import os
import re
import json
import time
import asyncio
import logging
import calendar
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from apify_client import ApifyClient
import schedule

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
        logging.FileHandler(
            os.path.join(LOG_DIR, "news.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("Trinity.News")

# ─────────────────────────────────────────────
# CONFIG & INIT
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
APIFY_TOKEN    = os.getenv("APIFY_TOKEN", "")

# [FIX 1] ใช้ ZoneInfo ชัดเจน ไม่ hardcode offset
THAI_TZ = ZoneInfo("Asia/Bangkok")

if not all([OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    try:
        from config import OPENAI_API_KEY, get_supabase  # type: ignore
        supabase: Client = get_supabase()
    except ImportError:
        log.critical("❌ ไม่พบ API Keys ใน .env หรือ config.py — ยุติการทำงาน")
        sys.exit(1)
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
apify_client  = ApifyClient(APIFY_TOKEN) if APIFY_TOKEN else None

MAX_RETRY = 3   # จำนวน retry สูงสุดสำหรับ API calls

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CURRENT_MODEL = "gpt-4o-mini"
TODAY_STR     = datetime.now(THAI_TZ).date().isoformat()   # [FIX 1] ใช้ THAI_TZ

SET100_SYMBOLS = [
    "AAV","ADVANC","AEONTS","AMATA","AOT","AP","AURA","AWC","BA","BAM",
    "BANPU","BBL","BCH","BCP","BCPG","BDMS","BEM","BGRIM","BH","BJC","BLA",
    "BTG","BTS","CBG","CCET","CENTEL","CHG","CK","COM7","CPALL","CPF","CPN",
    "CRC","DELTA","DOHOME","EA","EGCO","ERW","GFPT","GLOBAL","GPSC","GULF",
    "GUNKUL","HANA","HMPRO","ICHI","IRPC","IVL","JAS","JMART","JMT","JTS",
    "KBANK","KCE","KKP","KTB","KTC","LH","M","MEGA","MINT","MOSHI","MTC",
    "OR","OSP","PLANB","PR9","PRM","PTG","PTT","PTTEP","PTTGC","QH","RATCH",
    "RCL","SAWAD","SCB","SCC","SCGP","SIRI","SISB","SJWD","SPALI","SPRC",
    "STA","STECON","STGT","TASCO","TCAP","TFG","TIDLOR","TISCO","TLI","TOA",
    "TOP","TRUE","TTB","TU","VGI","WHA",
]

MARKET_KEYWORDS = [
    "SET","หุ้นไทย","ตลาดหุ้น","เศรษฐกิจ","ธปท.","FED","ดอกเบี้ย",
    "เงินบาท","SET50","SET100","เงินเฟ้อ","กนง.","หุ้น",
]

COMPANY_NAMES_TH: dict[str, list[str]] = {
    "AOT":   ["ทอท","ท่าอากาศยานไทย"],
    "PTT":   ["ปตท.","กลุ่มปตท"],
    "KBANK": ["กสิกรไทย","เคแบงก์","KBank"],
    "SCB":   ["ไทยพาณิชย์","SCB X","SCBX"],
    "CPALL": ["ซีพีออลล์","เซเว่นอีเลฟเว่น","CP ALL"],
    "BDMS":  ["กรุงเทพดุสิต","โรงพยาบาลกรุงเทพ","BDMS"],
    "ADVANC":["แอดวานซ์","เอไอเอส","AIS"],
    "TRUE":  ["ทรู","ทรูคอร์ปอเรชั่น"],
    "BBL":   ["แบงก์กรุงเทพ","ธนาคารกรุงเทพ","บัวหลวง"],
    "KTB":   ["กรุงไทย","ธนาคารกรุงไทย"],
    "TTB":   ["ทีทีบี","ทหารไทยธนชาต"],
}

RSS_FEEDS = {
    "Kaohoon":         "https://www.kaohoon.com/feed",
    "Thunhoon":        "https://thunhoon.com/feed",
    "Mitihoon":        "https://www.mitihoon.com/feed",
    "Prachachat":      "https://www.prachachat.net/finance/feed",
    "BangkokBiz":      "https://www.bangkokbiznews.com/rss/feed/business/finance",
    "Manager":         "https://mgronline.com/rss/stockmarket",
    "InfoQuest":       "https://www.infoquest.co.th/feed",
    "MoneyAndBanking": "https://moneyandbanking.co.th/feed",
}

# ─────────────────────────────────────────────
# [FIX 2] RETRY HELPER — Exponential Backoff
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
# RATE LIMITER (Token Bucket)
# ─────────────────────────────────────────────
class RateLimiter:
    """ควบคุมความถี่การเรียก API ไม่เกิน max_calls ต่อ period วินาที"""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period    = period
        self._calls: list[float] = []

    async def acquire(self):
        now = asyncio.get_event_loop().time()
        self._calls = [t for t in self._calls if now - t < self.period]
        if len(self._calls) >= self.max_calls:
            wait = self.period - (now - self._calls[0])
            await asyncio.sleep(wait)
        self._calls.append(asyncio.get_event_loop().time())

openai_limiter = RateLimiter(max_calls=50, period=60)

# ─────────────────────────────────────────────
# SYMBOL MATCHING
# ─────────────────────────────────────────────
def extract_symbols(text: str) -> list[str]:
    matched: set[str] = set()
    text_upper = text.upper()
    for sym in SET100_SYMBOLS:
        if sym in COMPANY_NAMES_TH:
            if any(th in text for th in COMPANY_NAMES_TH[sym]):
                matched.add(sym)
                continue
        if re.search(rf"(?<![A-Z]){re.escape(sym)}(?![A-Z])", text_upper):
            matched.add(sym)
    return list(matched)

# ─────────────────────────────────────────────
# DATE PARSER
# ─────────────────────────────────────────────
def parse_date(raw: str) -> str:
    now_utc = datetime.now(timezone.utc)
    if not raw:
        return now_utc.isoformat()
    raw = str(raw).lower().strip()
    try:
        for unit, kws in [
            ("second", ["วินาที","second","secs"]),
            ("minute", ["นาที","minute","mins"]),
            ("hour",   ["ชั่วโมง","hour"]),
            ("day",    ["วัน","day"]),
        ]:
            if any(k in raw for k in kws):
                n = int(re.findall(r'\d+', raw)[0]) if re.findall(r'\d+', raw) else 0
                return (now_utc - timedelta(**{f"{unit}s": n})).isoformat()
        dt = date_parser.parse(raw, fuzzy=True)
        if dt.tzinfo is None:
            # [FIX 1] สร้าง timezone จาก ZoneInfo ไม่ใช่ hardcode offset
            dt = dt.replace(tzinfo=THAI_TZ)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return now_utc.isoformat()

# ─────────────────────────────────────────────
# ASYNC RSS SCRAPER
# ─────────────────────────────────────────────
async def fetch_rss_async(session: aiohttp.ClientSession,
                          source: str, url: str) -> list[dict]:
    """ดึง RSS feed แบบ async"""
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            content = await resp.text()

        feed    = feedparser.parse(content)
        results = []

        for entry in feed.entries[:50]:
            ps = entry.get("published_parsed") or entry.get("updated_parsed")
            if ps:
                pub_date = datetime.fromtimestamp(
                    calendar.timegm(ps), timezone.utc
                ).isoformat()
            else:
                pub_date = parse_date(
                    entry.get("published") or entry.get("updated") or "")

            title = entry.get("title", "").strip()
            if not title:
                continue

            raw_summary = entry.get("summary", "") or entry.get("description", "")
            snippet     = BeautifulSoup(
                raw_summary, "html.parser"
            ).get_text(strip=True)[:500]
            link        = entry.get("link", "")
            symbols     = extract_symbols(title)

            item_base = {
                "title":        title,
                "snippet":      snippet,
                "source":       source,
                "url":          link,
                "published_at": pub_date,
            }

            if symbols:
                for sym in symbols:
                    results.append({**item_base, "symbol": sym})
            elif any(kw.upper() in title.upper() for kw in MARKET_KEYWORDS):
                results.append({
                    **item_base, "symbol": "SET",
                    "title": f"[MARKET] {title}",
                })

        log.info("✅ %s: %d บทความ", source, len(results))
        return results

    except asyncio.TimeoutError:
        log.warning("⏱️ %s: timeout", source)
        return []
    except Exception as e:
        log.warning("⚠️ %s: %s", source, e)
        return []


async def scrape_rss_all() -> list[dict]:
    """ดึง RSS ทุกแหล่งพร้อมกัน"""
    log.info("📰 [Phase 1] กวาด RSS News แบบ Async...")
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks   = [fetch_rss_async(session, src, url)
                   for src, url in RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks)
    all_news = [item for sublist in results for item in sublist]
    log.info("📰 RSS รวม: %d รายการ", len(all_news))
    return all_news

# ─────────────────────────────────────────────
# APIFY SOCIAL SCRAPER
# ─────────────────────────────────────────────
def scrape_social_apify() -> list[dict]:
    if not apify_client:
        log.info("⏭️ Skipping Apify (no token)")
        return []

    log.info("📱 [Phase 1.5] Apify Social Scraper...")
    queries = [
        "site:facebook.com 'SET100' OR 'หุ้นไทย'",
        "site:x.com 'หุ้นไทย' OR 'SET100'",
    ]
    results = []
    try:
        run = apify_client.actor("apify/google-search-scraper").call(
            run_input={
                "queries":          "\n".join(queries),
                "maxPagesPerQuery": 1,
                "resultsPerPage":   20,
                "timePeriod":       "d",
            }
        )
        for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            for res in item.get("organicResults", []):
                title   = res.get("title", "")
                snippet = res.get("description", "")
                url     = res.get("url", "")
                if not title or not url:
                    continue
                pub_date = parse_date(
                    res.get("date") or res.get("displayedDate") or "")
                symbols  = extract_symbols(title)
                base     = {
                    "title":        title,
                    "snippet":      snippet,
                    "source":       "Social",
                    "url":          url,
                    "published_at": pub_date,
                }
                if symbols:
                    for sym in symbols:
                        results.append({**base, "symbol": sym})
                elif any(kw.upper() in title.upper() for kw in MARKET_KEYWORDS):
                    results.append({
                        **base, "symbol": "SET",
                        "title": f"[SOCIAL] {title}",
                    })
    except Exception as e:
        log.warning("⚠️ Apify error: %s", e)

    log.info("📱 Apify รวม: %d รายการ", len(results))
    return results

# ─────────────────────────────────────────────
# OPENAI: MARKET TONE
# ─────────────────────────────────────────────
def get_market_tone() -> str:
    log.info("📊 [Phase 2] วิเคราะห์สภาวะตลาดรวม...")
    try:
        start_of_today = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
        res = supabase.table("stock_news")\
            .select("title,snippet")\
            .eq("symbol", "SET")\
            .gte("published_at", start_of_today)\
            .execute()

        if not res.data:
            return ("สภาวะตลาดวันนี้ความผันผวนจำกัด "
                    "นักลงทุนเน้นเลือกหุ้นรายบริษัท (Selective Buy)")

        market_content = "\n".join(
            [f"- {n['title']}: {n['snippet']}" for n in res.data[:15]]
        )
        prompt = (
            "คุณคือ 'ผู้จัดการกองทุนอาวุโส' ของไทย\n"
            f"จงสรุปสภาวะตลาดหุ้นไทย (SET) ในวันนี้จากกระแสข่าวด้านล่าง:\n"
            f"{market_content}\n\n"
            "สรุป 2-3 ประโยคภาษาทางการ กระชับ เน้นสาระ ไม่มีคำฟุ่มเฟือย"
        )
        # [FIX 2] retry สำหรับ OpenAI call
        resp = retry_call(
            lambda: openai_client.chat.completions.create(
                model=CURRENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            ),
            label="openai_market_tone",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("⚠️ Market tone error: %s", e)
        return "สภาวะตลาดอยู่ในช่วงเฝ้าระวังปัจจัยใหม่"

# ─────────────────────────────────────────────
# OPENAI: PER-STOCK ANALYSIS
# ─────────────────────────────────────────────
def get_stock_summary(symbol: str,
                      market_tone: str) -> Optional[dict]:
    """
    วิเคราะห์หุ้น 1 ตัว → return dict ที่ตรงกับ schema stock_summary
    [FIX 2] เพิ่ม retry สำหรับ OpenAI API call
    """
    # ดึงข่าวล่าสุด 24 ชั่วโมง
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        news_res = supabase.table("stock_news")\
            .select("title,snippet")\
            .eq("symbol", symbol)\
            .gte("published_at", cutoff)\
            .order("published_at", desc=True)\
            .limit(10)\
            .execute()
        news_items = news_res.data or []
    except Exception:
        news_items = []

    # ดึงข้อมูล Technical
    tech_ctx = ""
    try:
        rt = supabase.table("stock_realtime")\
            .select("last_price,percent_change,rsi_14,macd_val,macd_signal")\
            .eq("symbol", symbol).limit(1).execute()
        if rt.data:
            d = rt.data[0]
            tech_ctx = (
                f"ราคา: {d.get('last_price','N/A')} บาท | "
                f"เปลี่ยนแปลง: {d.get('percent_change','N/A')}% | "
                f"RSI14: {d.get('rsi_14','N/A')} | "
                f"MACD: {d.get('macd_val','N/A')} | "
                f"Signal: {d.get('macd_signal','N/A')}"
            )
    except Exception:
        tech_ctx = "ไม่มีข้อมูล Technical"

    # ดึงผลพยากรณ์ Chronos (7D)
    forecast_ctx = ""
    try:
        fc = supabase.table("stock_predictions_v3")\
            .select("predicted_price,lower_bound,upper_bound,horizon_type")\
            .eq("symbol", symbol).eq("horizon_type","7D")\
            .order("prediction_date", desc=False).limit(1).execute()
        if fc.data:
            f = fc.data[0]
            forecast_ctx = (
                f"AI Forecast 7D: {f['predicted_price']} "
                f"[{f['lower_bound']}–{f['upper_bound']}]"
            )
    except Exception:
        forecast_ctx = ""

    news_text = (
        "\n".join([f"- {n['title']}: {n.get('snippet','')}"
                   for n in news_items])
        if news_items
        else "ไม่มีข่าวใหม่ใน 24 ชั่วโมงที่ผ่านมา"
    )

    prompt = f"""คุณคือ "Senior Fund Manager" ผู้เชี่ยวชาญตลาดหุ้นไทย SET100
สภาวะตลาดรวมวันนี้: {market_tone}

ข้อมูลหุ้น: {symbol}
[Technical]   {tech_ctx}
[Chronos AI]  {forecast_ctx}
[ข่าวล่าสุด]
{news_text}

คำสั่ง: วิเคราะห์หุ้น {symbol} และตอบกลับ **เป็น JSON เท่านั้น** ตามรูปแบบด้านล่าง:
{{
  "investment_rating": <float 1.0-10.0 ตามความน่าสนใจลงทุน>,
  "sentiment_label":   "<Bullish|Bearish|Neutral>",
  "reasoning":         "<วิเคราะห์ 2-3 ประโยค ภาษาไทย สั้น กระชับ>"
}}

กฎสำคัญ:
- ถ้ามีข่าว → ให้น้ำหนักข่าว 60% + Technical 40%
- ถ้าไม่มีข่าว → ให้น้ำหนัก Technical 70% + Forecast 30%
- investment_rating: 7+ = แนะนำซื้อ, 5-7 = ถือ, < 5 = หลีกเลี่ยง
- ห้ามใส่ข้อความอื่นนอกจาก JSON"""

    try:
        # [FIX 2] retry สำหรับ OpenAI API call — ป้องกัน rate limit / transient error
        resp = retry_call(
            lambda: openai_client.chat.completions.create(
                model=CURRENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.4,
            ),
            label=f"openai_summary_{symbol}",
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            "symbol":            symbol,
            "investment_rating": float(result.get("investment_rating", 5.0)),
            "sentiment_label":   result.get("sentiment_label", "Neutral"),
            "reasoning":         result.get("reasoning", ""),
            "model_name":        CURRENT_MODEL,
            "created_at":        TODAY_STR,
        }
    except Exception as e:
        log.warning("⚠️ OpenAI error for %s: %s", symbol, e)
        return None

# ─────────────────────────────────────────────
# SAVE TO SUPABASE (พร้อม retry)
# ─────────────────────────────────────────────
def save_news(news_list: list[dict]):
    if not news_list:
        return
    try:
        retry_call(
            lambda: supabase.table("stock_news")
                .upsert(news_list, on_conflict="url,symbol")
                .execute(),
            label="save_news",
        )
        log.info("💾 บันทึกข่าว %d รายการสำเร็จ", len(news_list))
    except Exception as e:
        log.error("❌ บันทึกข่าวล้มเหลว: %s", e)


def save_summary(data: dict):
    try:
        retry_call(
            lambda: supabase.table("stock_summary")
                .upsert(data, on_conflict="symbol,created_at")
                .execute(),
            label=f"save_summary_{data.get('symbol')}",
        )
    except Exception as e:
        log.error("❌ บันทึก summary ล้มเหลว (%s): %s", data.get("symbol"), e)

# ─────────────────────────────────────────────
# CLEANUP OLD NEWS
# ─────────────────────────────────────────────
def cleanup_old_news(days: int = 7):
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        supabase.table("stock_news").delete().lt("published_at", cutoff).execute()
        log.info("🧹 ลบข่าวเก่ากว่า %d วันสำเร็จ", days)
    except Exception as e:
        log.warning("⚠️ cleanup error: %s", e)

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_news_analysis(mode: str = "full"):
    """
    mode='full'  → ดึง RSS + Social + วิเคราะห์ SET100 ทุกตัว
    mode='quick' → ดึง RSS + วิเคราะห์เฉพาะ SET index
    """
    global TODAY_STR
    # [FIX 1] อัปเดต TODAY_STR ด้วย THAI_TZ ทุกครั้งที่เรียก
    TODAY_STR = datetime.now(THAI_TZ).date().isoformat()

    log.info("=" * 60)
    log.info("🚀 Trinity News Pipeline [%s] | %s",
             mode.upper(),
             datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M:%S"))
    log.info("🔖 Batch Process Started")
    log.info("=" * 60)

    # Phase 1: ดึงข่าว
    all_news: list[dict] = asyncio.run(scrape_rss_all())
    if mode == "full":
        social_news = scrape_social_apify()
        all_news.extend(social_news)
    save_news(all_news)

    # Phase 2: Market Tone
    market_tone = get_market_tone()
    log.info("📈 Market Tone: %s...", market_tone[:80])

    # Phase 3: Per-stock AI Analysis
    analysis_queue = (["SET"] + SET100_SYMBOLS) if mode == "full" else ["SET"]
    total = len(analysis_queue)

    for i, symbol in enumerate(analysis_queue, 1):
        log.info("[%d/%d] 🤖 Analyzing: %s", i, total, symbol)
        result = get_stock_summary(symbol, market_tone)
        if result:
            save_summary(result)
            log.info("✅ %s | Rating: %.1f | %s",
                     symbol, result["investment_rating"],
                     result["sentiment_label"])
        # ป้องกัน rate limit (50 req/min → 1.2 วิ/call)
        time.sleep(1.3)

    cleanup_old_news(days=7)
    log.info("✨ Pipeline เสร็จสมบูรณ์! [%s]", CURRENT_MODEL)

# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
def is_market_open() -> bool:
    """
    ตลาดหุ้นไทยเปิด 10:00 – 17:00 วันจันทร์-ศุกร์
    [FIX 1] ใช้ ZoneInfo("Asia/Bangkok") แทน timezone(timedelta(hours=7))
            เพื่อรองรับ DST (แม้ไทยไม่มี DST แต่ใช้ชื่อ TZ ชัดเจนกว่า)
    """
    now_thai = datetime.now(THAI_TZ)   # ✅ ZoneInfo แทน hardcode offset
    return now_thai.weekday() < 5 and 10 <= now_thai.hour < 17


def scheduled_job_full():
    run_news_analysis(mode="full")


def scheduled_job_quick():
    if is_market_open():
        run_news_analysis(mode="quick")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trinity News Pipeline")
    parser.add_argument("--mode", default="full",
                        choices=["full","quick","schedule"])
    parser.add_argument("--once", action="store_true",
                        help="รันครั้งเดียวแล้วออก")
    args = parser.parse_args()

    if args.mode == "schedule" and not args.once:
        log.info("⏰ Scheduler โหมด: full ทุก 1 ชั่วโมง | "
                 "quick ทุก 15 นาทีช่วงตลาดเปิด")
        schedule.every(1).hours.do(scheduled_job_full)
        schedule.every(15).minutes.do(scheduled_job_quick)
        run_news_analysis(mode="full")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_news_analysis(
            mode=args.mode if args.mode != "schedule" else "full")

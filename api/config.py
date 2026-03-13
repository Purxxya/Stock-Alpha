# config.py — SECURE VERSION (use .env for all secrets)
import os
import torch
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ══ Supabase ══
SB_URL = os.environ["SUPABASE_URL"]
SB_KEY = os.environ["SUPABASE_KEY"]

# ══ AI APIs ══
HF_TOKEN       = os.environ["HF_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ══ Model Config ══
MODEL_ID = os.getenv("CHRONOS_MODEL", "amazon/chronos-t5-base")
DEVICE   = "mps" if torch.backends.mps.is_available() else \
           "cuda" if torch.cuda.is_available() else "cpu"

# ══ Supabase Client ══
def get_supabase() -> Client:  # ✅ define ครั้งเดียว
    return create_client(SB_URL, SB_KEY)

# ══ SET100 Symbols (Single Source of Truth) ══
SET100_SYMBOLS = [
    "AAV", "ADVANC", "AEONTS", "AMATA", "AOT", "AP", "AURA", "AWC", "BA", "BAM",
    "BANPU", "BBL", "BCH", "BCP", "BCPG", "BDMS", "BEM", "BGRIM", "BH", "BJC",
    "BLA", "BTG", "BTS", "CBG", "CCET", "CENTEL", "CHG", "CK", "COM7", "CPALL",
    "CPF", "CPN", "CRC", "DELTA", "DOHOME", "EA", "EGCO", "ERW", "GFPT", "GLOBAL",
    "GPSC", "GULF", "GUNKUL", "HANA", "HMPRO", "ICHI", "IRPC", "IVL", "JAS",
    "JMART", "JMT", "JTS", "KBANK", "KCE", "KKP", "KTB", "KTC", "LH", "M",
    "MEGA", "MINT", "MOSHI", "MTC", "OR", "OSP", "PLANB", "PR9", "PRM", "PTG",
    "PTT", "PTTEP", "PTTGC", "QH", "RATCH", "RCL", "SAWAD", "SCB", "SCC", "SCGP",
    "SIRI", "SISB", "SJWD", "SPALI", "SPRC", "STA", "STECON", "STGT", "TASCO",
    "TCAP", "TFG", "TIDLOR", "TISCO", "TLI", "TOA", "TOP", "TRUE", "TTB", "TU",
    "VGI", "WHA",
]

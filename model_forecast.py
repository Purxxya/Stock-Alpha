import torch
import pandas as pd
import numpy as np
import holidays
import warnings
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from chronos import ChronosPipeline 
from huggingface_hub import login
from config import get_supabase, HF_TOKEN, SET100_SYMBOLS

warnings.filterwarnings("ignore")

MODEL_ID = "amazon/chronos-t5-base" 
DEVICE = "mps"  
login(token=HF_TOKEN)
supabase = get_supabase()

def get_thai_forecast_dates(forecast_length=7):
    th_holidays = holidays.Thailand(years=[datetime.now().year, datetime.now().year + 1])
    thai_bday = CustomBusinessDay(holidays=th_holidays)
    start_from = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_dates = pd.date_range(start=start_from + timedelta(days=1), periods=forecast_length, freq=thai_bday)
    return forecast_dates

def sync_actual_prices_and_evaluate_v2():
    print("🔄 [Step 1] Evaluating Ranking in V2...")
    today_str = datetime.now().strftime('%Y-%m-%d')
    res = supabase.table("stock_predictions_v2").select("*").is_("actual_price", "null").lte("prediction_date", today_str).execute()
    if not res.data: return
    
    for item in res.data:
        hist = supabase.table("stock_historical").select("close_price").eq("symbol", item['symbol']).eq("date", item['prediction_date']).execute()
        if hist.data:
            actual = float(hist.data[0]['close_price'])
            c_error = abs(actual - float(item['chronos_price'])) / actual * 100 if item['chronos_price'] else None
            m_error = abs(actual - float(item['moirai_price'])) / actual * 100 if item['moirai_price'] else None
            winner = "Chronos" if (c_error is not None and m_error is not None and c_error < m_error) else ("MOIRAI" if m_error is not None else None)
            supabase.table("stock_predictions_v2").update({
                "actual_price": actual,
                "chronos_error_pct": round(c_error, 2) if c_error is not None else None,
                "moirai_error_pct": round(m_error, 2) if m_error is not None else None,
                "winner_model": winner
            }).eq("id", item['id']).execute()

def prepare_and_forecast_chronos(symbol, pipeline, forecast_dates):
    print(f"🔮 Analyzing {symbol} (Chronos)...")
    forecast_length = len(forecast_dates)
    res = supabase.table("stock_historical").select("date, close_price").eq("symbol", symbol).order("date", desc=True).limit(250).execute()
    if not res.data or len(res.data) < 128: return None

    df = pd.DataFrame(res.data).iloc[::-1].reset_index(drop=True)
    context_tensor = torch.tensor(df['close_price'].values[-128:]).unsqueeze(0)
    forecast = pipeline.predict(context_tensor, prediction_length=forecast_length, num_samples=20)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    records = []
    for i in range(forecast_length):
        records.append({
            "symbol": symbol,
            "prediction_date": forecast_dates[i].strftime('%Y-%m-%d'),
            "chronos_price": round(float(median[i]), 2),
            "chronos_lower": round(float(low[i]), 2),
            "chronos_upper": round(float(high[i]), 2),
            "created_at": datetime.now().isoformat()
        })
    return records

if __name__ == "__main__":
    sync_actual_prices_and_evaluate_v2()
    forecast_dates = get_thai_forecast_dates(forecast_length=7)
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
    for symbol in SET100_SYMBOLS:
        try:
            results = prepare_and_forecast_chronos(symbol, pipeline, forecast_dates)
            if results:
                supabase.table("stock_predictions_v2").upsert(results, on_conflict="symbol, prediction_date").execute()
                print(f"✅ {symbol}: Chronos Saved.")
        except Exception as e: print(f"❌ {symbol} Error: {e}")
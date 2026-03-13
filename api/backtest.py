import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline 
from config import get_supabase, MODEL_ID, DEVICE, SET100_SYMBOLS # ดึงรายชื่อหุ้นมาจาก config

supabase = get_supabase()

def run_backtest_simulation(symbol, pipeline, days_back=7):
    print(f"🧪 Backtesting {symbol} for the last {days_back} days...")
    
    try:
        # 1. ดึงข้อมูลประวัติ (จำกัด 200 วันเพื่อให้ Context แม่นยำ)
        res = supabase.table("stock_historical").select("date, close_price")\
                .eq("symbol", symbol)\
                .order("date", desc=True)\
                .limit(200).execute()
        
        if not res.data or len(res.data) < 150:
            print(f"⚠️ {symbol}: ข้อมูลไม่เพียงพอ (ต้องการอย่างน้อย 150 วัน)")
            return

        df = pd.DataFrame(res.data).iloc[::-1].reset_index(drop=True)
        backtest_records = []
        
        # 2. เริ่ม Sliding Window Backtest
        for i in range(days_back, 0, -1):
            context_df = df.iloc[:-(i)] 
            actual_price = df.iloc[-(i)]['close_price']
            target_date = df.iloc[-(i)]['date']
            
            # ใช้ EMA 3 วันล่าสุด 128 จุด (ตามที่เราจูนไว้ในโมเดลหลัก)
            target_series = context_df['close_price'].ewm(span=3, adjust=False).mean().values[-128:]
            context_tensor = torch.tensor(target_series).float().unsqueeze(0)
            
            # ทำนายผล
            forecast = pipeline.predict(context_tensor, prediction_length=1)
            predicted_price = float(np.median(forecast[0].numpy()))
            
            error_pct = abs(predicted_price - actual_price) / actual_price * 100
            
            backtest_records.append({
                "symbol": symbol,
                "prediction_date": target_date,
                "predicted_price": round(predicted_price, 2),
                "actual_price": round(float(actual_price), 2),
                "error_percent": round(error_pct, 2),
                "model_name": f"{MODEL_ID}-backtest"
            })

        # 3. Upsert ลง Database
        if backtest_records:
            supabase.table("stock_predictions").upsert(backtest_records).execute()
            print(f"✅ {symbol}: Backtest saved.")
            
    except Exception as e:
        print(f"❌ Error in {symbol}: {str(e)}")

if __name__ == "__main__":
    # โหลดโมเดลครั้งเดียวไว้ที่ GPU (MPS)
    print(f"🚀 Powering up {MODEL_ID} on Apple M4...")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, 
        device_map=DEVICE, 
        torch_dtype=torch.bfloat16
    )
    
    # วนลูป SET100 ทั้งหมด
    total = len(SET100_SYMBOLS)
    for index, symbol in enumerate(SET100_SYMBOLS, 1):
        print(f"\n[{index}/{total}]")
        run_backtest_simulation(symbol, pipeline)

    print("\n" + "="*50)
    print("🏁 GLOBAL SET100 BACKTEST COMPLETED!")
    print("="*50)

import psycopg2

try:
    # 💡 ใช้ aws-1-ap-southeast-1.pooler.supabase.com และ User 'postgres' เฉยๆ
    conn = psycopg2.connect("postgresql://postgres:Phuri254612@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require")
    print("✅ [POOLER] เชื่อมต่อสำเร็จ!")
    conn.close()
except Exception as e:
    print(f"❌ POOLER ล้มเหลว: {e}")
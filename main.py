"""
Trinity AI — Authentication Backend (FastAPI) - FIXED VERSION
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# ──────────────────────────────────────────────
#  DATABASE CONFIG
# ──────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ──────────────────────────────────────────────
#  JWT CONFIG
# ──────────────────────────────────────────────
# แนะนำให้ใช้ค่าจาก os.getenv("SECRET_KEY") ในอนาคต
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_for_dev_only")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


# ──────────────────────────────────────────────
#  PASSWORD HASHING (FIXED BCRYPT IDENT)
# ──────────────────────────────────────────────
# แก้ไขปัญหา AttributeError: module 'bcrypt' has no attribute '__about__'
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__ident="2b" 
)


# ──────────────────────────────────────────────
#  ORM BASE & USER MODEL
# ──────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ──────────────────────────────────────────────
#  PYDANTIC SCHEMAS
# ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    model_config = {"from_attributes": True}

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ──────────────────────────────────────────────
#  FASTAPI APP
# ──────────────────────────────────────────────
app = FastAPI(title="Trinity AI — Auth API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer()

# ──────────────────────────────────────────────
#  DEPENDENCIES
# ──────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ──────────────────────────────────────────────
#  HELPERS (FIXED 72-BYTE LIMIT)
# ──────────────────────────────────────────────
def hash_password(plain: str) -> str:
    # ตัดรหัสผ่านเหลือ 72 ตัวแรกเพื่อป้องกัน ValueError ใน bcrypt
    return pwd_context.hash(plain[:72])

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)

def create_access_token(user: User) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user.id),
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "Trinity AI Auth API"}

@app.post("/api/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    new_user = User(
        name=body.name, 
        email=body.email, 
        hashed_password=hash_password(body.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return TokenResponse(
        access_token=create_access_token(new_user),
        user=UserOut.model_validate(new_user)
    )

@app.post("/api/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return TokenResponse(
        access_token=create_access_token(user),
        user=UserOut.model_validate(user)
    )

@app.get("/api/profile", response_model=UserOut)
def profile(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


@app.get("/api/verify-token")
def verify_token(current_user: User = Depends(get_current_user)):
    """
    Lightweight endpoint สำหรับ Dashboard ใช้ตรวจสอบว่า Token ยังใช้ได้อยู่หรือไม่
    คืนค่า user info เมื่อ Token valid, คืน 401 อัตโนมัติเมื่อ Token หมดอายุ
    """
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
        }
    }


@app.post("/api/logout")
def logout(current_user: User = Depends(get_current_user)):
    """
    Server-side logout acknowledgement (Token-based ไม่ต้อง revoke ฝั่ง server จริง
    แต่ endpoint นี้ยืนยันว่า user ส่ง request ออกจากระบบจริง)
    """
    return {"message": "Logged out successfully", "user": current_user.name}
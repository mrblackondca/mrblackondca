# ═══════════════════════════════════════════════════════════════
#  DCA BOT — main.py  (single file backend)
#  Config + Models + Indicators + HyperLiquid + Bot Engine + API
# ═══════════════════════════════════════════════════════════════

import asyncio, hashlib, json, os, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



# ═══════════════════════════════════════════════════════════════
#  SUPABASE REST API  (HTTP — works on Render free tier IPv6)
# ═══════════════════════════════════════════════════════════════
SUPA_URL = os.getenv("SUPABASE_URL", "")
SUPA_KEY = os.getenv("SUPABASE_KEY", "")

async def supa_get(table: str, eq: dict = None) -> list:
    """Fetch rows from Supabase table."""
    if not SUPA_URL or not SUPA_KEY:
        return []
    try:
        url     = f"{SUPA_URL}/rest/v1/{table}"
        params  = {"select": "*"}
        if eq:
            for k, v in eq.items():
                params[f"{k}"] = f"eq.{v}"
        async with httpx.AsyncClient() as c:
            r = await c.get(url, params=params, headers={
                "apikey": SUPA_KEY,
                "Authorization": f"Bearer {SUPA_KEY}"
            }, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception as e:
        logger.error(f"supa_get: {e}")
        return []

async def supa_upsert(table: str, data: dict) -> bool:
    """Upsert row in Supabase table."""
    if not SUPA_URL or not SUPA_KEY:
        return False
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{SUPA_URL}/rest/v1/{table}",
                json=data,
                headers={
                    "apikey": SUPA_KEY,
                    "Authorization": f"Bearer {SUPA_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates"
                },
                timeout=10
            )
        return r.status_code in [200, 201]
    except Exception as e:
        logger.error(f"supa_upsert: {e}")
        return False

async def supa_insert(table: str, data: dict) -> bool:
    """Insert row in Supabase table."""
    if not SUPA_URL or not SUPA_KEY:
        return False
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{SUPA_URL}/rest/v1/{table}",
                json=data,
                headers={
                    "apikey": SUPA_KEY,
                    "Authorization": f"Bearer {SUPA_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
        return r.status_code in [200, 201]
    except Exception as e:
        logger.error(f"supa_insert: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
#  CONFIG  (reads from .env / Render environment variables)
# ═══════════════════════════════════════════════════════════════
class Cfg:
    HL_WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS")
    HL_PRIVATE_KEY    = os.getenv("HL_PRIVATE_KEY")
    QUICKNODE_RPC_URL = os.getenv("QUICKNODE_RPC_URL")
    DEFAULT_COIN      = os.getenv("DEFAULT_COIN",          "HYPE")
    DEFAULT_BALANCE   = float(os.getenv("DEFAULT_BALANCE",  "100.0"))
    DEFAULT_TRADE_PCT = float(os.getenv("DEFAULT_TRADE_PCT", "90.0"))
    DEFAULT_TP        = float(os.getenv("DEFAULT_TP",         "1.0"))
    DEFAULT_SL        = float(os.getenv("DEFAULT_SL",         "2.0"))
    DEFAULT_TP_PCT    = float(os.getenv("DEFAULT_TP_CLOSE_PCT","50.0"))
    DEFAULT_TRAIL     = float(os.getenv("DEFAULT_TRAIL",      "0.5"))
    DEFAULT_RSI       = float(os.getenv("DEFAULT_RSI_ENTRY",  "45.0"))
    DEFAULT_VOL       = float(os.getenv("DEFAULT_VOL_MULT",   "1.5"))
    USDC_CONTRACT     = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

cfg = Cfg()


# ═══════════════════════════════════════════════════════════════
#  MODELS / SCHEMAS
# ═══════════════════════════════════════════════════════════════
class BotMode(str, Enum):
    PAPER = "paper"
    LIVE  = "live"

class TradeReason(str, Enum):
    TP1    = "TP1"
    TRAIL  = "TRAIL"
    SL     = "SL"
    MANUAL = "MANUAL"

class BotConfig(BaseModel):
    coin:          str     = cfg.DEFAULT_COIN
    balance:       float   = cfg.DEFAULT_BALANCE
    trade_pct:     float   = cfg.DEFAULT_TRADE_PCT
    tp1:           float   = cfg.DEFAULT_TP
    sl:            float   = cfg.DEFAULT_SL
    tp1_close_pct: float   = cfg.DEFAULT_TP_PCT
    trail_stop:    float   = cfg.DEFAULT_TRAIL
    rsi_entry:     float   = cfg.DEFAULT_RSI
    vol_mult:      float   = cfg.DEFAULT_VOL
    mode:          BotMode = BotMode.PAPER

class BotConfigUpdate(BaseModel):
    coin:          Optional[str]   = None
    trade_pct:     Optional[float] = None
    tp1:           Optional[float] = None
    sl:            Optional[float] = None
    tp1_close_pct: Optional[float] = None
    trail_stop:    Optional[float] = None
    rsi_entry:     Optional[float] = None
    vol_mult:      Optional[float] = None

class OpenTrade(BaseModel):
    id:          int
    coin:        str
    entry_price: float
    amount:      float
    entry_fee:   float
    high_price:  float
    tp1_hit:     bool
    sl_price:    float
    tp1_price:   float
    opened_at:   datetime

class ClosedTrade(BaseModel):
    id:               int
    coin:             str
    entry_price:      float
    exit_price:       float
    amount:           float
    entry_fee:        float
    exit_fee:         float
    gas_fee:          float = 0.0
    swap_fee:         float = 0.0
    maker_fee:        float = 0.0
    taker_fee:        float
    total_fees:       float
    gross_pnl:        float
    net_pnl:          float
    pnl_pct:          float
    reason:           TradeReason
    opened_at:        datetime
    closed_at:        datetime
    duration_seconds: int

class MarketData(BaseModel):
    coin:             str
    price:            float
    price_change_pct: float
    rsi:              float
    ema9:             float
    ema21:            float
    ema_signal:       str
    vol_ratio:        float
    timestamp:        datetime

class BotStatus(BaseModel):
    running:        bool
    mode:           BotMode
    config:         BotConfig
    balance:        float
    start_balance:  float
    open_trade:     Optional[OpenTrade]
    total_trades:   int
    win_trades:     int
    loss_trades:    int
    win_rate:       float
    total_pnl:      float
    total_pnl_pct:  float
    total_fees:     float
    uptime_seconds: int
    last_signal:    str
    market:         Optional[MarketData]
    closed_trades:  List[ClosedTrade] = []

class PnlSummary(BaseModel):
    period:      str
    trades:      int
    wins:        int
    losses:      int
    gross_pnl:   float
    total_fees:  float
    net_pnl:     float
    net_pnl_pct: float
    best_trade:  float
    worst_trade: float
    avg_trade:   float

class WalletInfo(BaseModel):
    address:      str
    connected:    bool
    usdc_balance: float

class ApiResponse(BaseModel):
    success: bool
    message: str
    data:    Optional[dict] = None

class TradesResponse(BaseModel):
    trades:   List[ClosedTrade]
    total:    int
    page:     int
    per_page: int


# ═══════════════════════════════════════════════════════════════
#  INDICATORS  (RSI, EMA, Volume ratio)
# ═══════════════════════════════════════════════════════════════
def calc_rsi(closes: List[float], period: int = 7) -> float:
    if len(closes) < period + 1:
        return 50.0
    c  = np.array(closes, dtype=float)
    d  = np.diff(c)
    ag = np.mean(np.where(d > 0, d, 0.0)[:period])
    al = np.mean(np.where(d < 0, -d, 0.0)[:period])
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
    return round(100 - (100 / (1 + ag / al)), 2) if al > 0 else 100.0

def calc_ema(closes: List[float], period: int) -> float:
    if not closes: return 0.0
    if len(closes) < period: return float(closes[-1])
    k, ema = 2 / (period + 1), float(closes[0])
    for p in closes[1:]: ema = float(p) * k + ema * (1 - k)
    return round(ema, 6)

def calc_vol_ratio(volumes: List[float], window: int = 20) -> float:
    if len(volumes) < 2: return 1.0
    hist = volumes[:-1]
    avg  = np.mean(hist[-window:]) if len(hist) >= window else np.mean(hist)
    return round(volumes[-1] / avg, 2) if avg > 0 else 1.0

def compute_indicators(candles: list) -> Optional[Dict]:
    if len(candles) < 22: return None
    closes, volumes = [], []
    for c in candles:
        try:
            closes.append(float(c.get("c") or c.get("close", 0)))
            volumes.append(float(c.get("v") or c.get("volume", 0)))
        except Exception: continue
    if len(closes) < 22: return None
    ema9  = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    sig   = "BULL" if ema9 > ema21 * 1.001 else "BEAR" if ema9 < ema21 * 0.999 else "NEUTRAL"
    return {
        "rsi":        calc_rsi(closes, 7),
        "ema9":       ema9, "ema21": ema21,
        "ema_signal": sig,
        "vol_ratio":  calc_vol_ratio(volumes, 20),
        "last_close": closes[-1]
    }


# ═══════════════════════════════════════════════════════════════
#  HYPERLIQUID SERVICE
# ═══════════════════════════════════════════════════════════════
class HyperLiquidService:
    INFO_URL     = "https://api.hyperliquid.xyz/info"
    EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"

    def __init__(self):
        self.mode        = BotMode.PAPER   # always starts PAPER
        self._client: Optional[httpx.AsyncClient] = None
        self._spot_ids: Dict[str, int] = {}

    async def start(self):
        self._client = httpx.AsyncClient(timeout=15.0)
        await self._load_spot_ids()
        logger.info(f"HyperLiquid ready | mode={self.mode}")

    async def stop(self):
        if self._client: await self._client.aclose()

    async def _load_spot_ids(self):
        try:
            r = await self._client.post(self.INFO_URL,
                json={"type": "spotMeta"},
                headers={"Content-Type": "application/json"})
            for i, t in enumerate(r.json().get("tokens", [])):
                name = t.get("name", "").upper()
                if name: self._spot_ids[name] = 10000 + i
            logger.info(f"Spot IDs loaded: {len(self._spot_ids)} tokens")
        except Exception as e:
            logger.error(f"_load_spot_ids: {e}")

    def _asset_id(self, coin: str) -> int:
        return self._spot_ids.get(coin.upper(),
            {"HYPE":10000,"BNB":10001,"BTC":10002,"ETH":10003,
             "SOL":10004,"AVAX":10005,"DOGE":10006}.get(coin.upper(), 10000))

    async def get_price(self, coin: str) -> float:
        try:
            r    = await self._client.post(self.INFO_URL,
                json={"type": "allMids"},
                headers={"Content-Type": "application/json"})
            data = r.json()
            for k in [f"{coin}/USDC", coin]:
                if k in data: return float(data[k])
        except Exception as e: logger.error(f"get_price: {e}")
        return 0.0

    async def get_candles(self, coin: str, interval="5m", count=60) -> list:
        try:
            ms  = {"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000}.get(interval, 300_000)
            end = int(time.time() * 1000)
            r   = await self._client.post(self.INFO_URL,
                json={"type":"candleSnapshot","req":{
                    "coin":coin,"interval":interval,
                    "startTime":end-(count*ms),"endTime":end}},
                headers={"Content-Type": "application/json"})
            return r.json() or []
        except Exception as e:
            logger.error(f"get_candles: {e}"); return []

    async def get_usdc_balance(self, address: str) -> float:
        # Primary: HyperLiquid spot clearinghouse
        try:
            r = await self._client.post(self.INFO_URL,
                json={"type":"spotClearinghouseState","user":address},
                headers={"Content-Type": "application/json"})
            for b in r.json().get("balances", []):
                if b.get("coin") == "USDC":
                    return round(float(b.get("total", 0)), 6)
        except Exception as e: logger.error(f"hl_balance: {e}")
        # Fallback: QuickNode on-chain USDC
        if cfg.QUICKNODE_RPC_URL:
            try:
                padded = address.lower().replace("0x","").zfill(64)
                r = await self._client.post(cfg.QUICKNODE_RPC_URL,
                    json={"jsonrpc":"2.0","id":1,"method":"eth_call",
                          "params":[{"to":cfg.USDC_CONTRACT,"data":"0x70a08231"+padded},"latest"]},
                    headers={"Content-Type": "application/json"})
                raw = int(r.json().get("result","0x0"), 16)
                return round(raw / 1e6, 6)
            except Exception as e: logger.error(f"quicknode_balance: {e}")
        return 0.0

    async def place_buy(self, coin: str, usdc_amount: float, price: float) -> Dict:
        size = round(usdc_amount / price, 6)
        fee  = round(usdc_amount * 0.00070, 8)
        if self.mode == BotMode.PAPER:
            logger.info(f"[PAPER] BUY {size} {coin} @ ${price:.4f}")
            return {"status":"ok","paper":True,"price":price,
                    "taker_fee":fee,"gas_fee":0.0,"swap_fee":0.0,"maker_fee":0.0}
        return await self._live_order(coin, True, size, price, fee)

    async def place_sell(self, coin: str, size: float, price: float) -> Dict:
        usdc = round(size * price, 6)
        fee  = round(usdc * 0.00070, 8)
        if self.mode == BotMode.PAPER:
            logger.info(f"[PAPER] SELL {size} {coin} @ ${price:.4f}")
            return {"status":"ok","paper":True,"price":price,
                    "taker_fee":fee,"gas_fee":0.0,"swap_fee":0.0,"maker_fee":0.0}
        return await self._live_order(coin, False, size, price, fee)

    async def _live_order(self, coin, is_buy, sz, price, taker_fee) -> Dict:
        if not cfg.HL_PRIVATE_KEY:
            return {"status":"error","message":"HL_PRIVATE_KEY not set in env"}
        try:
            from eth_account import Account
            from eth_account.messages import encode_structured_data
            account  = Account.from_key(cfg.HL_PRIVATE_KEY)
            limit_px = round(price * (1.005 if is_buy else 0.995), 6)
            nonce    = int(time.time() * 1000)
            action   = {"type":"order","orders":[{
                "a":self._asset_id(coin),"b":is_buy,
                "p":str(limit_px),"s":str(round(sz,6)),
                "r":False,"t":{"limit":{"tif":"Ioc"}},"c":None
            }],"grouping":"na"}
            raw = json.dumps({"action":action,"nonce":nonce,"vaultAddress":None},
                             separators=(",",":")  ,sort_keys=True).encode()
            conn_id  = hashlib.sha256(raw).digest()
            signed   = account.sign_message(encode_structured_data({
                "types":{"EIP712Domain":[
                    {"name":"name","type":"string"},{"name":"version","type":"string"},
                    {"name":"chainId","type":"uint256"},{"name":"verifyingContract","type":"address"}],
                    "Agent":[{"name":"source","type":"string"},{"name":"connectionId","type":"bytes32"}]},
                "domain":{"chainId":42161,"name":"Exchange",
                          "verifyingContract":"0x0000000000000000000000000000000000000000","version":"1"},
                "primaryType":"Agent","message":{"source":"a","connectionId":conn_id}
            }))
            r = await self._client.post(self.EXCHANGE_URL,
                json={"action":action,"nonce":nonce,
                      "signature":{"r":hex(signed.r),"s":hex(signed.s),"v":signed.v},
                      "vaultAddress":None},
                headers={"Content-Type":"application/json"})
            res = r.json()
            logger.info(f"Live order: {res}")
            if res.get("status") == "ok":
                fill = price
                try:
                    fills = res.get("response",{}).get("data",{}).get("statuses",[])
                    if fills and fills[0].get("filled"):
                        fill = float(fills[0]["filled"].get("avgPx", price))
                except Exception: pass
                return {"status":"ok","paper":False,"price":fill,
                        "taker_fee":taker_fee,"gas_fee":0.0,"swap_fee":0.0,"maker_fee":0.0}
            return {"status":"error","message":str(res)}
        except Exception as e:
            logger.error(f"_live_order: {e}")
            return {"status":"error","message":str(e)}

    async def switch_mode(self, mode: BotMode):
        self.mode = mode
        logger.info(f"HL mode → {mode}")


hl = HyperLiquidService()


# ═══════════════════════════════════════════════════════════════
#  BOT ENGINE
# ═══════════════════════════════════════════════════════════════
class BotEngine:
    POLL = 30  # seconds

    def __init__(self):
        self.config       = BotConfig()
        self.balance      = self.config.balance
        self.start_bal    = self.config.balance
        self.running      = False
        self.open_trade:   Optional[OpenTrade]  = None
        self.trades:       List[ClosedTrade]    = []
        self.market:       Optional[MarketData] = None
        self.signal        = "Idle"
        self._task:        Optional[asyncio.Task] = None
        self._started:     Optional[datetime]     = None
        self._tid          = 1
        self._broadcast_fn = None

    def set_broadcast(self, fn): self._broadcast_fn = fn

    # ── START / STOP ─────────────────────────────────────────────
    async def start(self):
        if self.running: return
        self.running  = True
        self._started = datetime.now(timezone.utc)
        if self.config.mode == BotMode.LIVE and cfg.HL_WALLET_ADDRESS:
            bal = await hl.get_usdc_balance(cfg.HL_WALLET_ADDRESS)
            if bal > 0:
                self.balance = self.start_bal = bal
                logger.info(f"Live balance: ${bal:.4f} USDC")
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Bot started | {self.config.mode} | {self.config.coin} | ${self.balance}")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        self.signal = "Bot stopped"
        await self._emit()

    def update_config(self, updates: Dict[str, Any]):
        for k, v in updates.items():
            if hasattr(self.config, k) and v is not None:
                setattr(self.config, k, v)
        asyncio.create_task(supa_upsert("settings", {
            "id":"default",
            "coin":self.config.coin,"balance":self.config.balance,
            "trade_pct":self.config.trade_pct,"tp1":self.config.tp1,
            "sl":self.config.sl,"trail_stop":self.config.trail_stop,
            "rsi_entry":self.config.rsi_entry,"vol_mult":self.config.vol_mult,
            "updated_at":datetime.now(timezone.utc).isoformat()
        }))

    async def switch_mode(self, mode: BotMode):
        if self.open_trade: await self._force_close()
        was_running = self.running
        if self.running: await self.stop()
        self.config.mode = mode
        await hl.switch_mode(mode)
        if mode == BotMode.LIVE:
            wallet = cfg.HL_WALLET_ADDRESS or os.getenv("HL_WALLET_ADDRESS","")
            if wallet:
                bal = await hl.get_usdc_balance(wallet)
                if bal > 0:
                    self.balance = self.start_bal = bal
                    logger.info(f"LIVE balance: ${bal:.4f} USDC")
        else:
            rows = await supa_get("settings", {"id": "default"})
            paper_bal = rows[0].get("balance", self.config.balance) if rows else self.config.balance
            self.balance = self.start_bal = paper_bal
        # Save mode to Supabase
        await supa_upsert("bot_state", {
            "id": "default", "mode": mode.value,
            "balance": self.balance, "start_balance": self.start_bal,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
        if was_running: await self.start()
        await self._emit()

    def reset(self, new_bal: Optional[float] = None):
        if self.open_trade: return False, "Cannot reset with open trade"
        b = new_bal or self.config.balance
        self.balance = self.start_bal = b
        self.trades  = []
        self._tid    = 1
        self.signal  = "Reset — ready"
        return True, f"Reset to ${b:.2f}"

    # ── LOOP ─────────────────────────────────────────────────────
    async def _loop(self):
        while self.running:
            try: await self._tick()
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Loop: {e}")
            await asyncio.sleep(self.POLL)

    async def _tick(self):
        candles = await hl.get_candles(self.config.coin, "5m", 60)
        if not candles:
            self.signal = "⚠️ No candle data"; await self._emit(); return
        inds = compute_indicators(candles)
        if not inds:
            self.signal = "⚠️ Not enough data"; await self._emit(); return

        price = await hl.get_price(self.config.coin)
        if price <= 0: price = inds["last_close"]

        # Sync live USDC balance
        if self.config.mode == BotMode.LIVE and cfg.HL_WALLET_ADDRESS and not self.open_trade:
            bal = await hl.get_usdc_balance(cfg.HL_WALLET_ADDRESS)
            if bal > 0: self.balance = bal

        p0  = float(candles[0].get("c", price)) if candles else price
        pch = ((price - p0) / p0 * 100) if p0 > 0 else 0.0
        self.market = MarketData(
            coin=self.config.coin, price=price,
            price_change_pct=round(pch,3),
            rsi=inds["rsi"], ema9=inds["ema9"], ema21=inds["ema21"],
            ema_signal=inds["ema_signal"], vol_ratio=inds["vol_ratio"],
            timestamp=datetime.now(timezone.utc)
        )
        if self.open_trade: await self._manage(price)
        else:               await self._check_entry(price, inds)
        await self._emit()

    # ── ENTRY ────────────────────────────────────────────────────
    async def _check_entry(self, price: float, inds: Dict):
        c = self.config
        ok = inds["rsi"] <= c.rsi_entry and inds["vol_ratio"] >= c.vol_mult and inds["ema_signal"] != "BEAR"
        self.signal = f"⏳ RSI:{inds['rsi']:.1f} VOL:{inds['vol_ratio']:.1f}x EMA:{inds['ema_signal']}"
        if ok: await self._enter(price)

    async def _enter(self, price: float):
        c      = self.config
        amount = round(self.balance * (c.trade_pct / 100), 6)
        if amount < 1.0: self.signal = "⚠️ Balance too low (min $1)"; return
        res = await hl.place_buy(c.coin, amount, price)
        if res.get("status") != "ok":
            self.signal = f"❌ Buy failed: {res.get('message','?')}"; return
        fee = res.get("taker_fee", round(amount * 0.00070, 8))
        self.open_trade = OpenTrade(
            id=self._tid, coin=c.coin, entry_price=price, amount=amount,
            entry_fee=fee, high_price=price, tp1_hit=False,
            sl_price=round(price*(1-c.sl/100),6),
            tp1_price=round(price*(1+c.tp1/100),6),
            opened_at=datetime.now(timezone.utc)
        )
        self._tid += 1
        if c.mode == BotMode.PAPER:
            self.balance = round(self.balance - amount, 6)
        self.signal = f"🚀 ENTRY @ ${price:.4f} | ${amount:.4f}"
        logger.info(f"Opened #{self.open_trade.id} @ {price}")

    # ── MANAGE ───────────────────────────────────────────────────
    async def _manage(self, price: float):
        t   = self.open_trade
        c   = self.config
        pct = (price - t.entry_price) / t.entry_price * 100
        if price > t.high_price:
            self.open_trade = t.model_copy(update={"high_price": price}); t = self.open_trade
        if not t.tp1_hit and pct >= c.tp1:
            self.open_trade = t.model_copy(update={"tp1_hit": True}); t = self.open_trade
            self.signal = f"✅ TP1 +{c.tp1}% HIT @ ${price:.4f}"
            logger.info(f"TP1 hit #{t.id}")
        if t.tp1_hit and (t.high_price - price) / t.high_price * 100 >= c.trail_stop:
            await self._close(price, TradeReason.TRAIL); return
        if pct <= -c.sl:
            await self._close(price, TradeReason.SL); return
        self.signal = (f"📊 HOLD {'+' if pct>=0 else ''}{pct:.2f}% "
                       f"{'| TP1 ✅ trailing' if t.tp1_hit else f'| TP1 @ ${t.tp1_price:.4f}'}")

    async def _force_close(self):
        if self.open_trade:
            p = await hl.get_price(self.config.coin)
            await self._close(p or self.open_trade.entry_price, TradeReason.MANUAL)

    async def _close(self, price: float, reason: TradeReason):
        t        = self.open_trade
        size     = round(t.amount / t.entry_price, 6)
        res      = await hl.place_sell(t.coin, size, price)
        if res.get("status") != "ok" and reason != TradeReason.MANUAL:
            self.signal = f"❌ Sell failed: {res.get('message','')}"; return
        ep         = float(res.get("price", price))
        exit_fee   = res.get("taker_fee", round(size * ep * 0.00070, 8))
        total_fees = round(t.entry_fee + exit_fee, 8)
        gross      = (ep - t.entry_price) * size
        net        = gross - total_fees
        pct        = (ep - t.entry_price) / t.entry_price * 100
        dur        = int((datetime.now(timezone.utc) - t.opened_at).total_seconds())

        self.trades.insert(0, ClosedTrade(
            id=t.id, coin=t.coin,
            entry_price=t.entry_price, exit_price=ep,
            amount=t.amount,
            entry_fee=round(t.entry_fee,8), exit_fee=round(exit_fee,8),
            gas_fee=0.0, swap_fee=0.0, maker_fee=0.0,
            taker_fee=total_fees, total_fees=total_fees,
            gross_pnl=round(gross,6), net_pnl=round(net,6),
            pnl_pct=round(pct,4), reason=reason,
            opened_at=t.opened_at, closed_at=datetime.now(timezone.utc),
            duration_seconds=dur
        ))

        if self.config.mode == BotMode.PAPER:
            self.balance = round(self.balance + t.amount + net, 6)
        elif cfg.HL_WALLET_ADDRESS:
            bal = await hl.get_usdc_balance(cfg.HL_WALLET_ADDRESS)
            if bal > 0: self.balance = bal

        self.open_trade = None
        self.signal = (f"{'✅' if net>=0 else '❌'} CLOSED ({reason.value}) | "
                       f"Net: {'+' if net>=0 else ''}${net:.4f}")
        logger.info(f"Closed #{t.id} | {reason} | net={net:.4f} | bal={self.balance:.4f}")
        # Save trade + balance to Supabase
        asyncio.create_task(supa_insert("trades", {
            "trade_id":ct.id,"coin":ct.coin,
            "entry_price":ct.entry_price,"exit_price":ct.exit_price,
            "amount":ct.amount,"entry_fee":ct.entry_fee,"exit_fee":ct.exit_fee,
            "gas_fee":0.0,"swap_fee":0.0,"maker_fee":0.0,
            "taker_fee":ct.taker_fee,"total_fees":ct.total_fees,
            "gross_pnl":ct.gross_pnl,"net_pnl":ct.net_pnl,"pnl_pct":ct.pnl_pct,
            "reason":ct.reason.value,
            "opened_at":ct.opened_at.isoformat(),"closed_at":ct.closed_at.isoformat(),
            "duration_seconds":ct.duration_seconds,"mode":self.config.mode.value
        }))
        asyncio.create_task(supa_upsert("bot_state", {
            "id":"default","balance":self.balance,"start_balance":self.start_bal,
            "total_trades":len(self.trades),
            "updated_at":datetime.now(timezone.utc).isoformat()
        }))

    # ── STATUS ───────────────────────────────────────────────────
    def status(self) -> BotStatus:
        wins  = [t for t in self.trades if t.net_pnl > 0]
        total = len(self.trades)
        pnl   = self.balance - self.start_bal
        fees  = sum(t.total_fees for t in self.trades)
        up    = int((datetime.now(timezone.utc)-self._started).total_seconds()) if self._started else 0
        return BotStatus(
            running=self.running, mode=self.config.mode, config=self.config,
            balance=round(self.balance,6), start_balance=round(self.start_bal,6),
            open_trade=self.open_trade, total_trades=total,
            win_trades=len(wins), loss_trades=total-len(wins),
            win_rate=round(len(wins)/total*100,1) if total else 0.0,
            total_pnl=round(pnl,6),
            total_pnl_pct=round(pnl/self.start_bal*100,3) if self.start_bal else 0,
            total_fees=round(fees,6), uptime_seconds=up,
            last_signal=self.signal, market=self.market,
            closed_trades=self.trades[:100]
        )

    def pnl_summary(self, period="ALL") -> PnlSummary:
        now = datetime.now(timezone.utc)
        cut = {"1D":86400,"7D":604800,"1M":2592000}.get(period)
        ts  = [t for t in self.trades if not cut or (now-t.closed_at).total_seconds()<=cut]
        if not ts:
            return PnlSummary(period=period,trades=0,wins=0,losses=0,
                gross_pnl=0,total_fees=0,net_pnl=0,net_pnl_pct=0,
                best_trade=0,worst_trade=0,avg_trade=0)
        wins = [t for t in ts if t.net_pnl > 0]
        net  = sum(t.net_pnl for t in ts)
        return PnlSummary(
            period=period, trades=len(ts), wins=len(wins), losses=len(ts)-len(wins),
            gross_pnl=round(sum(t.gross_pnl for t in ts),6),
            total_fees=round(sum(t.total_fees for t in ts),6),
            net_pnl=round(net,6),
            net_pnl_pct=round(net/self.start_bal*100,3) if self.start_bal else 0,
            best_trade=round(max(t.net_pnl for t in ts),6),
            worst_trade=round(min(t.net_pnl for t in ts),6),
            avg_trade=round(net/len(ts),6)
        )

    async def _emit(self):
        if self._broadcast_fn:
            try: await self._broadcast_fn(self.status().model_dump(mode="json"))
            except Exception as e: logger.debug(f"emit: {e}")


bot = BotEngine()


# ═══════════════════════════════════════════════════════════════
#  WEBSOCKET MANAGER
# ═══════════════════════════════════════════════════════════════
class WSManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.active:
            try: await ws.send_json(data)
            except Exception: dead.add(ws)
        self.active -= dead


ws_mgr = WSManager()

async def _broadcast(data: dict):
    await ws_mgr.broadcast({"type":"status","data":data})

bot.set_broadcast(_broadcast)


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DCA Bot...")
    await hl.start()
    # Load settings from Supabase
    rows = await supa_get("settings", {"id": "default"})
    if rows:
        s = rows[0]
        for key in ["coin","balance","trade_pct","tp1","sl","trail_stop","rsi_entry","vol_mult"]:
            if s.get(key) is not None and hasattr(bot.config, key):
                setattr(bot.config, key, s[key])
        if s.get("wallet_addr"):
            import os; os.environ["HL_WALLET_ADDRESS"] = s["wallet_addr"]
        logger.info("Settings loaded from Supabase")
    # Load bot state
    state = await supa_get("bot_state", {"id": "default"})
    if state:
        st = state[0]
        bot.balance    = st.get("balance",    bot.config.balance)
        bot.start_bal  = st.get("start_balance", bot.config.balance)
        mode_str       = st.get("mode", "paper")
        bot.config.mode = BotMode.LIVE if mode_str == "live" else BotMode.PAPER
        await hl.switch_mode(bot.config.mode)
        logger.info(f"State loaded | bal=${bot.balance:.4f} | mode={bot.config.mode}")
    # Load trade history
    trade_rows = await supa_get("trades")
    if trade_rows:
        for t in sorted(trade_rows, key=lambda x: x.get("closed_at",""), reverse=True)[:100]:
            try:
                bot.trades.append(ClosedTrade(
                    id=t["trade_id"], coin=t["coin"],
                    entry_price=t["entry_price"], exit_price=t["exit_price"],
                    amount=t["amount"], entry_fee=t["entry_fee"], exit_fee=t["exit_fee"],
                    gas_fee=t.get("gas_fee",0), swap_fee=t.get("swap_fee",0),
                    maker_fee=t.get("maker_fee",0), taker_fee=t["taker_fee"],
                    total_fees=t["total_fees"], gross_pnl=t["gross_pnl"],
                    net_pnl=t["net_pnl"], pnl_pct=t["pnl_pct"],
                    reason=TradeReason(t["reason"]),
                    opened_at=datetime.fromisoformat(t["opened_at"]),
                    closed_at=datetime.fromisoformat(t["closed_at"]),
                    duration_seconds=t.get("duration_seconds",0)
                ))
            except Exception as e: logger.error(f"Trade load: {e}")
        logger.info(f"Loaded {len(bot.trades)} trades from Supabase")
    yield
    await hl.stop()
    logger.info("Shutdown complete")

app = FastAPI(title="DCA Bot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── ROUTES ───────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found!</h1>")

@app.get("/api/v1/health")
async def health():
    return {"status":"ok","ts":datetime.utcnow().isoformat()}

@app.post("/api/v1/auth")
async def auth(body: dict):
    pwd = os.getenv("BOT_PASSWORD", "admin123")
    if body.get("password") == pwd:
        return {"success": True}
    return {"success": False, "message": "Wrong password"}

# Bot control
@app.get("/api/v1/bot/status")
async def get_status():
    return bot.status().model_dump(mode="json")

@app.post("/api/v1/bot/start")
async def start_bot():
    if bot.running: return ApiResponse(success=False,message="Already running")
    await bot.start(); return ApiResponse(success=True,message="Bot started")

@app.post("/api/v1/bot/stop")
async def stop_bot():
    if not bot.running: return ApiResponse(success=False,message="Not running")
    await bot.stop(); return ApiResponse(success=True,message="Bot stopped")

@app.post("/api/v1/bot/mode/{mode}")
async def switch_mode(mode: str):
    try: m = BotMode(mode.lower())
    except ValueError: raise HTTPException(400, f"Use: paper | live")
    await bot.switch_mode(m)
    return ApiResponse(success=True,message=f"Switched to {m.value.upper()}")

@app.get("/api/v1/bot/config")
async def get_config():
    return bot.config.model_dump()

@app.put("/api/v1/bot/config")
async def update_config(update: BotConfigUpdate):
    bot.update_config(update.model_dump(exclude_none=True))
    return ApiResponse(success=True,message="Config updated",data=bot.config.model_dump())

@app.post("/api/v1/bot/reset")
async def reset_bot(balance: Optional[float] = Query(None)):
    ok, msg = bot.reset(balance)
    return ApiResponse(success=ok,message=msg)

# Market
@app.get("/api/v1/market/{coin}")
async def get_market(coin: str):
    price   = await hl.get_price(coin.upper())
    candles = await hl.get_candles(coin.upper(),"5m",60)
    inds    = compute_indicators(candles) if candles else {}
    return {"coin":coin.upper(),"price":price,"indicators":inds}

# Trades
@app.get("/api/v1/trades")
async def get_trades(page:int=Query(1,ge=1), per_page:int=Query(50,ge=1,le=200)):
    start = (page-1)*per_page
    return TradesResponse(trades=bot.trades[start:start+per_page],
                          total=len(bot.trades),page=page,per_page=per_page)

@app.get("/api/v1/trades/open")
async def get_open_trade():
    if not bot.open_trade: return {"open":False,"trade":None}
    return {"open":True,"trade":bot.open_trade.model_dump(mode="json")}

@app.delete("/api/v1/trades/open")
async def force_close():
    if not bot.open_trade: return ApiResponse(success=False,message="No open trade")
    await bot._force_close(); return ApiResponse(success=True,message="Closed manually")

# PnL
@app.get("/api/v1/pnl")
async def get_all_pnl():
    return {p:bot.pnl_summary(p).model_dump() for p in ["1D","7D","1M","ALL"]}

@app.get("/api/v1/pnl/{period}")
async def get_pnl(period: str):
    if period.upper() not in ["1D","7D","1M","ALL"]:
        raise HTTPException(400,"Use: 1D | 7D | 1M | ALL")
    return bot.pnl_summary(period.upper()).model_dump()

# Wallet
@app.get("/api/v1/wallet/{address}")
async def get_wallet(address: str):
    bal = await hl.get_usdc_balance(address)
    return WalletInfo(address=address,connected=True,usdc_balance=bal)

# WebSocket
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_mgr.connect(websocket)
    try:
        await websocket.send_json({"type":"status","data":bot.status().model_dump(mode="json")})
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping": await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type":"ping"})
    except WebSocketDisconnect: ws_mgr.disconnect(websocket)
    except Exception:           ws_mgr.disconnect(websocket)

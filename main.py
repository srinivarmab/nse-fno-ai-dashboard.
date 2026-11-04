import asyncio
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

IST = ZoneInfo('Asia/Kolkata')
SYMBOLS = os.getenv('SYMBOLS', 'RELIANCE,HDFCBANK,ICICIBANK,SBIN,INFY,TCS,ITC,LT,BHARTIARTL,TATAMOTORS,MARUTI,SUNPHARMA,TATASTEEL,HINDALCO,AXISBANK').split(',')
SECTORS = {'RELIANCE':'Energy','HDFCBANK':'Banking','ICICIBANK':'Banking','SBIN':'Banking','AXISBANK':'Banking','INFY':'IT','TCS':'IT','ITC':'FMCG','LT':'Infrastructure','BHARTIARTL':'Telecom','TATAMOTORS':'Auto','MARUTI':'Auto','SUNPHARMA':'Pharma','TATASTEEL':'Metal','HINDALCO':'Metal'}
app = FastAPI(title='NSE F&O Research Dashboard', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS','http://localhost:3000').split(','), allow_methods=['GET','POST'], allow_headers=['*'])
cache = {'updated_at':None,'stocks':[], 'errors':[]}
trades = []

class Strategy(BaseModel):
    rsi_min: float = Field(55, ge=0, le=100)
    adx_min: float = Field(20, ge=0, le=100)
    target_pct: float = Field(2, gt=0, le=100)
    hold_days: int = Field(5, ge=1, le=20)
    lookback_days: int = Field(30, ge=5, le=90)
    symbols: list[str] = Field(default_factory=lambda: SYMBOLS[:5])

class PaperOrder(BaseModel):
    symbol: str
    side: str = 'BUY'
    quantity: int = Field(gt=0, le=100000)


def indicators(df):
    d=df.copy().dropna(subset=['Open','High','Low','Close','Volume'])
    c,h,l,v = [d[x].astype(float) for x in ['Close','High','Low','Volume']]
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); d['rsi']=100-100/(1+rs)
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0); minus=down.where((down>up)&(down>0),0)
    pdi=100*plus.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
    mdi=100*minus.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['adx']=dx.ewm(alpha=1/14,adjust=False).mean()
    # Daily OHLCV only supports a daily typical-price volume proxy, NOT true session VWAP.
    d['vwap_proxy']=( ((h+l+c)/3)*v ).rolling(5).sum()/v.rolling(5).sum().replace(0,np.nan)
    basic=(h+l)/2; upper=basic+3*atr; lower=basic-3*atr
    fu=upper.copy(); fl=lower.copy(); st=np.full(len(d),np.nan); trend=np.ones(len(d),dtype=bool)
    for i in range(1,len(d)):
        fu.iloc[i]=upper.iloc[i] if (upper.iloc[i]<fu.iloc[i-1] or c.iloc[i-1]>fu.iloc[i-1]) else fu.iloc[i-1]
        fl.iloc[i]=lower.iloc[i] if (lower.iloc[i]>fl.iloc[i-1] or c.iloc[i-1]<fl.iloc[i-1]) else fl.iloc[i-1]
        trend[i]=c.iloc[i]>fu.iloc[i-1] if not trend[i-1] else c.iloc[i]>=fl.iloc[i-1]
        st[i]=fl.iloc[i] if trend[i] else fu.iloc[i]
    d['supertrend']=st
    return d


def fetch(symbol, period='6mo', interval='1d'):
    if not symbol.replace('-','').replace('&','').isalnum(): raise ValueError('Invalid symbol')
    d=yf.Ticker(symbol+'.NS').history(period=period,interval=interval,auto_adjust=False)
    if d.empty: raise ValueError('Yahoo returned no candles')
    d=d.sort_index(); return d


def snapshot():
    rows=[]; errors=[]
    for s in SYMBOLS:
        s=s.strip().upper()
        try:
            d=indicators(fetch(s)); last=d.iloc[-1]; prev=d.iloc[-2]
            bullish=bool(last.rsi>55 and last.adx>20 and last.Close>last.vwap_proxy and last.Close>last.supertrend)
            bearish=bool(last.rsi<45 and last.adx>20 and last.Close<last.vwap_proxy and last.Close<last.supertrend)
            rows.append({'symbol':s,'sector':SECTORS.get(s,'Other'),'price':round(float(last.Close),2),'change':round(float((last.Close/prev.Close-1)*100),2),'rsi':round(float(last.rsi),1) if pd.notna(last.rsi) else None,'adx':round(float(last.adx),1) if pd.notna(last.adx) else None,'direction':'BULLISH' if bullish else 'BEARISH' if bearish else 'NEUTRAL','candle_at':d.index[-1].isoformat()})
        except Exception as exc: errors.append({'symbol':s,'error':str(exc)[:180]})
    return {'updated_at':datetime.now(IST).isoformat(),'stocks':rows,'errors':errors,'data_mode':'delayed/research; daily VWAP proxy'}

@app.get('/api/market')
def market():
    global cache
    now=datetime.now(IST)
    if cache['updated_at']:
        dt=datetime.fromisoformat(cache['updated_at'])
        if (now-dt).total_seconds()<300: return cache
    cache=snapshot(); return cache

@app.get('/api/health')
def health(): return {'ok':True,'time':datetime.now(IST).isoformat()}

@app.post('/api/backtest')
def backtest(cfg:Strategy):
    results=[]; errors=[]
    for sym in cfg.symbols[:30]:
        sym=sym.upper().removesuffix('.NS')
        try:
            d=indicators(fetch(sym,period='1y'))
            # last 30 *completed* sessions; next-session entry requires a following candle
            d=d.iloc[:-1]  # conservative: drop latest potentially incomplete daily candle
            start=max(35,len(d)-cfg.lookback_days-cfg.hold_days-1)
            busy_until=-1
            for i in range(start,len(d)-cfg.hold_days):
                if i<=busy_until: continue
                r=d.iloc[i]
                if pd.isna(r[['rsi','adx','vwap_proxy','supertrend']]).any(): continue
                if not (r.rsi>cfg.rsi_min and r.adx>cfg.adx_min and r.Close>r.vwap_proxy and r.Close>r.supertrend):continue
                entry_idx=i+1; entry=float(d.iloc[entry_idx].Open); target=entry*(1+cfg.target_pct/100)
                exit_idx=min(entry_idx+cfg.hold_days-1,len(d)-1)
                hit=False; exit_price=float(d.iloc[exit_idx].Close)
                for j in range(entry_idx,exit_idx+1):
                    if float(d.iloc[j].High)>=target:
                        hit=True;exit_idx=j;exit_price=target;break
                segment=d.iloc[entry_idx:exit_idx+1]
                mae=float(segment.Low.min()/entry-1)*100
                ret=(exit_price/entry-1)*100
                results.append({'symbol':sym,'signal_date':str(d.index[i].date()),'entry_date':str(d.index[entry_idx].date()),'exit_date':str(d.index[exit_idx].date()),'entry':round(entry,2),'exit':round(exit_price,2),'target_hit':hit,'return_pct':round(ret,3),'mae_pct':round(mae,3)})
                busy_until=exit_idx
        except Exception as exc:errors.append({'symbol':sym,'error':str(exc)[:180]})
    # independent trade return series, equal-weight sequential approximation, not portfolio equity
    ordered=sorted(results,key=lambda x:(x['exit_date'],x['symbol']))
    equity=1.; peak=1.; max_dd=0.
    for r in ordered:
        equity*=1+r['return_pct']/100
        peak=max(peak,equity);max_dd=min(max_dd,(equity/peak-1)*100)
    n=len(results)
    return {'summary':{'trades':n,'win_rate_pct':round(100*sum(x['return_pct']>0 for x in results)/n,2) if n else None,'target_hit_rate_pct':round(100*sum(x['target_hit'] for x in results)/n,2) if n else None,'mean_return_pct':round(sum(x['return_pct'] for x in results)/n,3) if n else None,'sequential_drawdown_pct':round(max_dd,2) if n else None},'trades':results,'errors':errors,'warnings':['Daily volume-weighted typical-price proxy is NOT Chartink daily session VWAP. Results do not validate exact Chartink strategy.','Last daily candle excluded; fixed 5-day exit, no stop, fees or slippage.','Only signals with sufficient forward candles are included. Current symbols are not historical F&O constituents.','Daily signals only; no intraday 5-minute entry logic.']}

@app.post('/api/paper/order')
def paper_order(order:PaperOrder):
    if order.side!='BUY':raise HTTPException(400,'Phase 1 supports BUY only')
    row=next((r for r in cache['stocks'] if r['symbol']==order.symbol.upper()),None)
    if not row:raise HTTPException(400,'Refresh market first or symbol unavailable')
    trade={'id':len(trades)+1,'symbol':row['symbol'],'side':'BUY','quantity':order.quantity,'entry_price':row['price'],'created_at':datetime.now(IST).isoformat(),'status':'OPEN','data_candle_at':row['candle_at']}
    trades.append(trade);return trade

@app.get('/api/paper/trades')
def paper_trades():
    return {'trades':[dict(t,mark_price=next((r['price'] for r in cache['stocks'] if r['symbol']==t['symbol']),None)) for t in trades],'warning':'In-memory demo only: restarts erase trades. No simulated execution costs.'}

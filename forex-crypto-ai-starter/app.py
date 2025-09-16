import os, io, json, datetime as dt
import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ML
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# Data
import yfinance as yf
try:
    import ccxt
except Exception:
    ccxt = None

APP_BASE = os.getcwd()
DATA_DIR = os.path.join(APP_BASE, "data"); os.makedirs(DATA_DIR, exist_ok=True)
RUNS_DIR = os.path.join(APP_BASE, "runs"); os.makedirs(RUNS_DIR, exist_ok=True)

SERVER_ENDPOINT = os.getenv("SERVER_ENDPOINT","").strip()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY","").strip()

# ---------- Helpers ----------
def to_dates(df, dc="Date"):
    df[dc] = pd.to_datetime(df[dc], utc=True, errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=[dc]).sort_values(dc).reset_index(drop=True)

def save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf

def asset_key(asset_type, symbol, exchange=""):
    if asset_type=="FX": return f"fx_{symbol.lower()}"
    sx = (exchange or "binance").lower()
    return f"{sx}_{symbol.lower().replace('/','')}"

# ---------- Data fetchers ----------
def yf_fx_ticker(sym): return f"{sym.upper()}=X"

def fetch_fx_daily(symbol):
    # Alpha Vantage (اختياري)
    if ALPHA_VANTAGE_KEY:
        try:
            import urllib.request, json as js
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={symbol[:3]}&to_symbol={symbol[3:]}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
            with urllib.request.urlopen(url, timeout=20) as r:
                data = js.loads(r.read().decode())
            ts = data.get("Time Series FX (Daily)", {})
            rows = [{"Date": d, "Close": float(v["4. close"])} for d,v in ts.items()]
            if rows: return to_dates(pd.DataFrame(rows))
        except Exception:
            pass
    # Yahoo Finance بديل
    tkr = yf_fx_ticker(symbol)
    hist = yf.Ticker(tkr).history(interval="1d", start="2015-01-01")
    if hist is None or len(hist)==0: raise RuntimeError(f"No data for {symbol}")
    return to_dates(hist.reset_index()[["Date","Close"]])

def fetch_crypto_ohlcv(exchange, pair, timeframe="1d"):
    if ccxt is None: raise RuntimeError("ccxt not installed")
    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    data = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=2000)
    if not data: raise RuntimeError(f"No OHLCV {exchange} {pair}")
    df = pd.DataFrame(data, columns=["ts","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["Date","Close"]].sort_values("Date").reset_index(drop=True)

# ---------- Strategies (0..1 probabilities) ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(series, n=14):
    d = series.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean(); rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / (rd.replace(0, np.nan)); return (100 - (100/(1+rs))).fillna(50.0)
def donchian(s, n): return s.rolling(n).min(), s.rolling(n).max()
def sigmoid(x): return 1/(1+np.exp(-x))

def build_strats(df, price_col="Close"):
    x = df.copy()
    px = pd.to_numeric(x[price_col], errors="coerce").astype(float)
    std20 = px.rolling(20).std(); std50 = px.rolling(50).std()
    e20 = ema(px,20)
    prob_trend_ema = sigmoid((px - e20)/(std50+1e-9))

    lo, hi = donchian(px,20)
    breakout_up = (px > hi.shift(1)).astype(float)
    mag = (px - hi.shift(1))/(std50+1e-9)
    prob_breakout_donch = sigmoid(mag.fillna(0.0)) * (breakout_up)

    r2 = rsi(px,2); sma20 = px.rolling(20).mean()
    z = (px - sma20)/(std20+1e-9)
    prob_meanrev_rsi2_bb = sigmoid((10 - r2.clip(0,100))/4 + (-z))

    # false breakout (up/down) مبسّطة
    lo_p, hi_p = lo.shift(1), hi.shift(1)
    up_break_y = (px.shift(1) > hi_p.shift(1)); up_false_today = up_break_y & (px <= hi_p)
    dn_break_y = (px.shift(1) < lo_p.shift(1)); dn_false_today = dn_break_y & (px >= lo_p)
    dist_up = (px - hi_p)/(std50+1e-9); dist_dn = (lo_p - px)/(std50+1e-9)
    signed_fb = dn_false_today.astype(float)*sigmoid(dist_dn.clip(-3,3)) - up_false_today.astype(float)*sigmoid(dist_up.clip(-3,3))
    prob_false_breakout = ((signed_fb).clip(-1,1)+1)/2.0

    out = pd.DataFrame({
        "Date": x["Date"],
        "prob_trend_ema": prob_trend_ema.fillna(0.5),
        "prob_breakout_donch": prob_breakout_donch.fillna(0.5),
        "prob_meanrev_rsi2_bb": prob_meanrev_rsi2_bb.fillna(0.5),
        "prob_false_breakout": prob_false_breakout.fillna(0.5),
    })
    out["prob_src"] = out[[c for c in out.columns if c.startswith("prob_")]].mean(axis=1)
    return out[["Date","prob_src"]]

# ---------- LSTM ----------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))
    def forward(self, x):
        out,_ = self.lstm(x); last = out[:, -1, :]
        return self.head(last).squeeze(-1)

def build_features(df, price_col="Close"):
    x = df.copy()
    px = pd.to_numeric(x[price_col], errors="coerce").astype(float)
    x["logp"] = np.log(px)
    x["ret1"] = x["logp"].diff()
    x["sma_5"] = px.rolling(5).mean()
    x["sma_20"] = px.rolling(20).mean()
    x["ema_10"] = px.ewm(span=10, adjust=False).mean()
    x["vol_10"] = x["ret1"].rolling(10).std()
    x["sma_ratio"] = x["sma_5"] / x["sma_20"]
    x["price_sma20"] = px / x["sma_20"]
    x = x.replace([np.inf,-np.inf], np.nan).bfill().ffill()
    return x

def label_direction(df, price_col="Close", horizon=1):
    logp = np.log(pd.to_numeric(df[price_col], errors="coerce").astype(float))
    fwd = logp.shift(-horizon) - logp
    return (fwd > 0).astype(int)

def make_sequences(feats, labels, seq_len):
    X, y = [], []
    for t in range(seq_len, len(labels)):
        X.append(feats[t-seq_len:t]); y.append(labels[t])
    if not X:
        return np.zeros((0,seq_len,feats.shape[1]),dtype=np.float32), np.zeros((0,),dtype=np.float32)
    return np.stack(X,0).astype(np.float32), np.array(y, dtype=np.float32)

def train_lstm_on_df(df, lookback=60, horizon=1, hidden=64, layers=2, dropout=0.2,
                     val_size=0.15, test_size=0.15, epochs=20, batch_size=64, lr=1e-3, outdir="runs/tmp"):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    feat = build_features(df)
    y = label_direction(feat, horizon=horizon)
    feat = feat.iloc[:-horizon].reset_index(drop=True)
    y = y.iloc[:-horizon].reset_index(drop=True)

    cols = [c for c in feat.columns if c not in ["Date","Close"]]
    feats = feat[cols].values.astype(np.float32); labels = y.values.astype(np.float32)
    X, yseq = make_sequences(feats, labels, lookback)
    dates = feat["Date"].tolist()
    n = len(X); n_test = int(round(n * test_size)); n_val = int(round(n * val_size)); n_train = max(n - n_val - n_test, 1)

    if n_train <= 0: raise RuntimeError("Not enough data to train LSTM.")

    scaler = StandardScaler()
    Xtr = X[0:n_train].reshape(n_train*lookback, feats.shape[1])
    scaler.fit(Xtr)

    def transform(X_):
        Xf = X_.reshape(len(X_)*lookback, feats.shape[1])
        return scaler.transform(Xf).reshape(len(X_), lookback, feats.shape[1])

    X_train = transform(X[0:n_train]); y_train = yseq[0:n_train]
    X_val   = transform(X[n_train:n_train+n_val]);   y_val   = yseq[n_train:n_train+n_val]

    tr = DataLoader(list(zip(X_train,y_train)), batch_size=batch_size, shuffle=True)
    va = DataLoader(list(zip(X_val,y_val)), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(X.shape[2], hidden, layers, dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1, best_state, noimp = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in tr:
            xb = torch.tensor(xb).to(device); yb = torch.tensor(yb).to(device)
            optim.zero_grad(); logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
        # val
        model.eval()
        with torch.no_grad():
            v_logits=[]; v_y=[]
            for xb,yb in va:
                xb = torch.tensor(xb).to(device)
                v_logits.append(model(xb).detach().cpu().numpy()); v_y.append(yb.numpy())
            if v_logits:
                v_logits=np.concatenate(v_logits); v_y=np.concatenate(v_y)
                v_probs = 1/(1+np.exp(-v_logits)); v_preds = (v_probs>=0.5).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(v_y, v_preds, average='binary', zero_division=0)
            else:
                f1=0.0
        if f1>best_f1:
            best_f1=f1; best_state=model.state_dict(); noimp=0
        else:
            noimp+=1
            if noimp>=5: break

    if best_state: model.load_state_dict(best_state)

    # Save + predict all
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    torch.save(model.state_dict(), os.path.join(outdir, "best_model.pt"))
    cfg = {"seq_len": lookback, "hidden": hidden, "layers": layers, "dropout": dropout}
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w", encoding="utf-8"), indent=2)

    with torch.no_grad():
        Xn = transform(X)
        xb = torch.tensor(Xn).to(device)
        logits = model(xb).detach().cpu().numpy()
        probs = 1/(1+np.exp(-logits)); preds=(probs>=0.5).astype(int)
    dates_seq = pd.to_datetime(dates)[lookback:]
    pred_df = pd.DataFrame({"Date": dates_seq, "y_prob_up": probs.flatten(), "y_pred_up": preds.flatten().astype(int)})
    pred_df.to_csv(os.path.join(outdir, "predictions_infer.csv"), index=False)

    return {"best_val_f1": float(best_f1), "n_train": int(n_train), "n_all": int(n)}, pred_df

def build_prob_source(df, run_dir=None):
    s = build_strats(df).rename(columns={"prob_src":"prob_strats"})
    # دمج LSTM لو موجود
    if run_dir and os.path.isfile(os.path.join(run_dir, "predictions_infer.csv")):
        ldf = pd.read_csv(os.path.join(run_dir, "predictions_infer.csv"))
        ldf["Date"]=pd.to_datetime(ldf["Date"], utc=True, errors="coerce").dt.tz_localize(None)
        m = pd.merge(s, ldf[["Date","y_prob_up"]], on="Date", how="inner").rename(columns={"y_prob_up":"prob_lstm"})
        if len(m)>0:
            m["prob_src"] = m[["prob_strats","prob_lstm"]].mean(axis=1)
            return m[["Date","prob_src"]]
    s = s.rename(columns={"prob_strats":"prob_src"})
    return s[["Date","prob_src"]]

def decide_from_prob(px, prob_df, entry_hi=0.48, exit_lo=0.40, smooth=5, min_hold=8, cooldown=8, regime="ema200_up"):
    df = pd.merge(px, prob_df, on="Date", how="inner").dropna().reset_index(drop=True)
    pr = pd.to_numeric(df["prob_src"], errors="coerce")
    if smooth>0: pr = pr.ewm(span=smooth, adjust=False).mean()
    df["prob"] = pr.bfill().ffill()
    regime_ok = pd.Series(True, index=df.index)
    if regime=="ema200_up":
        ema200 = df["Close"].ewm(span=200, adjust=False).mean()
        regime_ok = (df["Close"]>=ema200).fillna(False)
    pos=np.zeros(len(df)); hold=0; cool=0
    for i in range(1,len(df)):
        new_pos = pos[i-1]; p = df.loc[i,"prob"]
        if not regime_ok.iloc[i]:
            new_pos=0.0; hold=0
        else:
            if cool>0:
                new_pos=pos[i-1]; cool-=1
                if new_pos!=0 and hold>0: hold-=1
            else:
                if pos[i-1]==0.0 and p>=entry_hi:
                    new_pos=1.0; hold=min_hold; cool=cooldown
                elif pos[i-1]==1.0:
                    if hold>0: new_pos=1.0; hold-=1
                    elif p<=exit_lo: new_pos=0.0; cool=cooldown
        pos[i]=new_pos
    df["pos"]=pos
    # equity
    logret=np.log(df["Close"].astype(float)).diff().fillna(0.0)
    gross=logret*df["pos"]; change=df["pos"].diff().abs().fillna(df["pos"].abs())
    net = gross - change*((20/10000)/2)  # 20 bps roundtrip مثال
    eq=(1.0+net).cumprod()
    i=len(df)-1
    if i>=1 and pos[i]>pos[i-1]: action="ENTER LONG"
    elif i>=1 and pos[i]<pos[i-1]: action="EXIT to CASH"
    else: action="HOLD LONG" if pos[i]==1.0 else "HOLD CASH"
    info = {
        "date": str(df.loc[i,"Date"].date()),
        "close": float(df.loc[i,"Close"]),
        "prob": float(df.loc[i,"prob"]),
        "position": "LONG" if pos[i]==1.0 else "CASH",
        "action": action,
        "regime_ok": bool(regime_ok.iloc[i]),
    }
    return info, df, eq

def plot_equity(df, eq):
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(df["Date"], df["Close"]); ax[0].set_title("Price")
    ax[1].plot(df["Date"], eq); ax[1].set_title("Equity (net)")
    ax[1].axhline(1.0, linestyle="--", linewidth=0.8)
    for a in ax: a.grid(True, alpha=0.2)
    return save_fig(fig)

# ================= WEB UI =================
CSS = "footer {visibility: hidden}"

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🟢 Forex & Crypto AI — Training / Signals Dashboard")

    with gr.Tab("1) Data & Training"):
        asset_type = gr.Radio(["FX","CRYPTO"], value="FX", label="Asset Type")
        with gr.Row():
            fx_symbol = gr.Textbox(value="EURUSD", label="FX Symbol (e.g., EURUSD)")
            exchange = gr.Textbox(value="binance", label="Exchange (crypto)")
            pair = gr.Textbox(value="BTC/USDT", label="Crypto Pair")
        with gr.Row():
            lookback = gr.Slider(20, 120, value=60, step=1, label="LSTM Lookback (days)")
            epochs = gr.Slider(5, 60, value=20, step=1, label="Epochs")
            hidden = gr.Slider(16, 256, value=64, step=16, label="Hidden Size")
            layers = gr.Slider(1, 3, value=2, step=1, label="LSTM Layers")
        train_btn = gr.Button("🚀 Fetch & Train")
        train_log = gr.JSON(label="Training Summary")
        train_plot = gr.Image(label="Equity Preview (after backtest)")

    with gr.Tab("2) Inference & Backtest"):
        with gr.Row():
            entry_hi = gr.Slider(0.35, 0.65, value=0.476524, step=0.001, label="Entry Threshold")
            exit_lo = gr.Slider(0.25, 0.55, value=0.403601, step=0.001, label="Exit Threshold")
            smooth = gr.Slider(0, 20, value=5, step=1, label="Smoothing (EMA span)")
            min_hold = gr.Slider(0, 20, value=10, step=1, label="Min Hold bars")
            cooldown = gr.Slider(0, 20, value=5, step=1, label="Cooldown bars")
            regime = gr.Dropdown(["ema200_up","none"], value="ema200_up", label="Regime Filter")
        infer_btn = gr.Button("🔎 Infer & Backtest")
        last_signal = gr.JSON(label="Latest Decision")
        backtest_plot = gr.Image(label="Backtest Chart")
        send_btn = gr.Button("📨 POST to Server (optional, set SERVER_ENDPOINT env)")

    with gr.Tab("3) Watchlist (Multi-asset quick scan)"):
        wl_fx = gr.Textbox(value="EURUSD,GBPUSD", label="FX list (comma separated)")
        wl_crypto = gr.Textbox(value="BTC/USDT,ETH/USDT", label="Crypto pairs")
        scan_btn = gr.Button("🔭 Scan Now")
        scan_tbl = gr.Dataframe(headers=["asset_key","date","action","prob","regime_ok"], label="Scan Results")

    with gr.Tab("4) Screenshot"):
        shot_btn = gr.Button("📸 Snapshot")
        shot_img = gr.Image(label="Screenshot PNG")

    st_df = gr.State(None)
    st_prob = gr.State(None)
    st_key = gr.State("")
    st_run = gr.State("")

    def _fetch_data(asset_type, fx_symbol, exchange, pair):
        if asset_type=="FX":
            df = fetch_fx_daily(fx_symbol.strip().upper())
            key = asset_key("FX", fx_symbol)
        else:
            df = fetch_crypto_ohlcv(exchange.strip().lower(), pair.strip().upper(), timeframe="1d")
            key = asset_key("CRYPTO", pair, exchange)
        return df, key

    def on_train(asset_type, fx_symbol, exchange, pair, lookback, epochs, hidden, layers, entry_hi, exit_lo, smooth, min_hold, cooldown, regime):
        df, key = _fetch_data(asset_type, fx_symbol, exchange, pair)
        run_dir = os.path.join(RUNS_DIR, f"{key}_lstm"); os.makedirs(run_dir, exist_ok=True)
        summary, pred_df = train_lstm_on_df(df, lookback=int(lookback), epochs=int(epochs), hidden=int(hidden), layers=int(layers), outdir=run_dir)
        prob = build_prob_source(df, run_dir)
        info, bdf, eq = decide_from_prob(df, prob, entry_hi=float(entry_hi), exit_lo=float(exit_lo), smooth=int(smooth), min_hold=int(min_hold), cooldown=int(cooldown), regime=regime)
        figpng = plot_equity(bdf, eq)
        return summary, figpng, df, prob, key, run_dir

    train_btn.click(
        fn=on_train,
        inputs=[asset_type, fx_symbol, exchange, pair, lookback, epochs, hidden, layers, entry_hi, exit_lo, smooth, min_hold, cooldown, regime],
        outputs=[train_log, train_plot, st_df, st_prob, st_key, st_run]
    )

    def on_infer(entry_hi, exit_lo, smooth, min_hold, cooldown, regime, df, prob, key, run_dir):
        if df is None or prob is None: raise gr.Error("No data/prob — Train first from tab (1).")
        info, bdf, eq = decide_from_prob(df, prob, entry_hi=float(entry_hi), exit_lo=float(exit_lo),
                                         smooth=int(smooth), min_hold=int(min_hold), cooldown=int(cooldown), regime=regime)
        figpng = plot_equity(bdf, eq)
        return info, figpng

    infer_btn.click(
        fn=on_infer,
        inputs=[entry_hi, exit_lo, smooth, min_hold, cooldown, regime, st_df, st_prob, st_key, st_run],
        outputs=[last_signal, backtest_plot]
    )

    def on_send(info_json):
        if not SERVER_ENDPOINT:
            return gr.Warning("SERVER_ENDPOINT not set on server.")
        try:
            r = requests.post(SERVER_ENDPOINT, json=info_json, timeout=10)
            return gr.Info(f"POST {r.status_code}")
        except Exception as e:
            return gr.Warning(f"POST failed: {e}")
    send_btn.click(fn=on_send, inputs=[last_signal], outputs=[])

    def on_scan(wfx, wcr, entry_hi, exit_lo, smooth, min_hold, cooldown, regime):
        rows=[]
        # FX
        for sym in [s.strip().upper() for s in wfx.split(",") if s.strip()]:
            key = f"fx_{sym.lower()}"
            try:
                df = fetch_fx_daily(sym)
                prob = build_prob_source(df, os.path.join(RUNS_DIR, f"{key}_lstm"))
                info, _, _ = decide_from_prob(df, prob, float(entry_hi), float(exit_lo), int(smooth), int(min_hold), int(cooldown), regime)
                rows.append([key, info["date"], info["action"], round(info["prob"],4), info["regime_ok"]])
            except Exception as e:
                rows.append([key, "ERR", str(e), None, None])
        # Crypto
        for pr in [s.strip().upper() for s in wcr.split(",") if s.strip()]:
            key = f"binance_{pr.lower().replace('/','')}"
            try:
                df = fetch_crypto_ohlcv("binance", pr, timeframe="1d")
                prob = build_prob_source(df, os.path.join(RUNS_DIR, f"{key}_lstm"))
                info, _, _ = decide_from_prob(df, prob, float(entry_hi), float(exit_lo), int(smooth), int(min_hold), int(cooldown), regime)
                rows.append([key, info["date"], info["action"], round(info["prob"],4), info["regime_ok"]])
            except Exception as e:
                rows.append([key, "ERR", str(e), None, None])
        return pd.DataFrame(rows, columns=["asset_key","date","action","prob","regime_ok"])

    scan_btn.click(
        fn=on_scan,
        inputs=[wl_fx, wl_crypto, entry_hi, exit_lo, smooth, min_hold, cooldown, regime],
        outputs=[scan_tbl]
    )

    def on_shot():
        fig, ax = plt.subplots(figsize=(8,2))
        ax.text(0.5,0.5,"Forex & Crypto AI — Snapshot", ha="center", va="center")
        ax.axis("off")
        return save_fig(fig)
    shot_btn.click(fn=on_shot, inputs=[], outputs=[shot_img])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_error=True)

# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import altair as alt
import plotly.graph_objects as go
import pydeck as pdkpython 
import math
import plotly.graph_objects as go
import random
import gdown
from datetime import datetime, timedelta, time as dtime

# ── 1) Download and Load model & data ─────────────────────────────────────────

# Create model directory
MODEL_DIR = r"c:\Users\User\pollution_app_final"
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file paths
MODEL_PATH = os.path.join(MODEL_DIR, "rf_tuned_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")
START_YEAR_PATH = os.path.join(MODEL_DIR, "start_year.joblib")
CSV_PATH = os.path.join(MODEL_DIR, "pollution_data.csv")

# Download large model file from Google Drive if not exists
if not os.path.exists(MODEL_PATH):
    file_id = "1Yh0_uyIy-oX6kigPh5zbzdXsMyh9j_jx"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model and other files (assuming other files are deployed with your app)
import joblib

rf_model   = joblib.load(MODEL_PATH)
scaler     = joblib.load(SCALER_PATH)
FEATURES   = joblib.load(FEATURES_PATH)
START_YEAR = joblib.load(START_YEAR_PATH)

history = (
    pd.read_csv(CSV_PATH, parse_dates=["date"])
      .rename(columns={"date": "datetime"})
      .set_index("datetime")
      .sort_index()
)

# ── 2) Build‐features helper ───────────────────────────────────────────────────
def build_features(rec):
    prev = history.loc[:rec["datetime"], "pollution"].tail(3).values
    if len(prev) < 3:
        prev = np.pad(prev, (3-len(prev),0), mode="edge")
    rec["pollution_lag1"], rec["pollution_lag2"] = prev[-1], prev[-2]
    rec["pollution_roll_mean3"] = prev.mean()
    df = pd.DataFrame([rec])
    dt = rec["datetime"]
    df["hour"], df["day"], df["month"] = dt.hour, dt.day, dt.month
    df["years_since_start"] = dt.year - START_YEAR
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["day_sin"]  = np.sin(2*np.pi*df["day"]/31)
    df["day_cos"]  = np.cos(2*np.pi*df["day"]/31)
    df["month_sin"]= np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]= np.cos(2*np.pi*df["month"]/12)
    for wd in ["NE","NW","SE","SW"]:
        df[f"wnd_dir_{wd}"] = (df["wnd_dir"]==wd).astype(int)
    df = df.drop(columns=["wnd_dir","datetime"])
    num_cols = [
        "dew","temp","press","wnd_spd","snow","rain",
        "pollution_lag1","pollution_lag2","pollution_roll_mean3"
    ]
    df[num_cols] = scaler.transform(df[num_cols])
    return df[FEATURES]

# ── 3) Quick‐weather presets & callback ────────────────────────────────────────
PRESETS = {
    "None":       {"dew":10,"temp":20,"press":1012,"wnd_spd":5.0,"snow":0,"rain":0,"wnd_dir":"NE"},
    "Morning Fog":{"dew":15,"temp":30,"press":1015,"wnd_spd":0.5,"snow":0,"rain":0,"wnd_dir":"NE"},
    "Windy Day":  {"dew":-5,"temp":5,"press":1025,"wnd_spd":8,"snow":0,"rain":0,"wnd_dir":"NW"},
    "Rainy Day":  {"dew":18,"temp":22,"press":1008,"wnd_spd":3,"snow":0,"rain":10,"wnd_dir":"SW"},
}
def apply_preset():
    p = st.session_state["p_preset"]
    for k, v in (PRESETS.get(p) or {}).items():
        key = f"adv_{k}" if k != "wnd_dir" else "adv_wnd_dir"
        try:
            st.session_state[key] = float(v)
        except:
            st.session_state[key] = v

# ── 4) AQI helper ──────────────────────────────────────────────────────────────
def to_aqi(pm: float):
    if   pm <= 35:   return ("Good",          "#009966", "Air is clean.")
    elif pm <= 75:   return ("Moderate",      "#FFDE33", "Take breaks.")
    elif pm <=115:   return ("Unhealthy",     "#FF9933", "Wear a mask.")
    elif pm <=150:   return ("Very Unhealthy","#CC0033", "Limit outdoor time.")
    else:            return ("Hazardous",     "#660099", "Stay inside.")

# ── 5) Climatology for new Predict tab ────────────────────────────────────────
orig = history.reset_index().rename(columns={"datetime":"date"})
orig["month"] = orig["date"].dt.month
orig["day"]   = orig["date"].dt.day
orig["hour"]  = orig["date"].dt.hour
clim = (
    orig.groupby(["month","day","hour"])["pollution"]
        .mean().reset_index(name="climatology")
)

# ── 6) Live‐weather fetch (round to hour) ──────────────────────────────────────
def fetch_weather(lat, lon, dt):
    # floor dt to the hour
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    start = dt0.strftime("%Y-%m-%dT%H:%M")
    end   = (dt0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    params = {
      "latitude": lat, "longitude": lon,
      "hourly": "temperature_2m,dewpoint_2m,pressure_msl,windspeed_10m",
      "start": start, "end": end,
      "timezone": "auto"
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()["hourly"]
    key = dt0.isoformat(timespec="minutes") 
    idx = r["time"].index(key)
    return {
      "temp":    r["temperature_2m"][idx],
      "dew":     r["dewpoint_2m"][idx],
      "press":   r["pressure_msl"][idx],
      "wnd_spd": r["windspeed_10m"][idx],
      "snow":    0.0,
      "rain":    0.0,
      "wnd_dir": "NE"
    }

def build_features_from_rec(rec):
    """
    Build X-matrix for a single record dict using exactly the lag values
    we pass in rec, and not recomputing them from history.
    """
    df = pd.DataFrame([rec])
    dt = rec["datetime"]
    df["hour"]             = dt.hour
    df["day"]              = dt.day
    df["month"]            = dt.month
    df["years_since_start"]= dt.year - START_YEAR

    # cyclical
    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    df["day_sin"]    = np.sin(2*np.pi*df["day"]/31)
    df["day_cos"]    = np.cos(2*np.pi*df["day"]/31)
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)

    # one-hot wind direction
    for wd in ["NE","NW","SE","SW"]:
        df[f"wnd_dir_{wd}"] = (df["wnd_dir"]==wd).astype(int)

    # drop extras
    df = df.drop(columns=["wnd_dir","datetime"])

    # scale numeric columns (including the lags you passed in)
    num_cols = [
        "dew","temp","press","wnd_spd","snow","rain",
        "pollution_lag1","pollution_lag2","pollution_roll_mean3"
    ]
    df[num_cols] = scaler.transform(df[num_cols])
    return df[FEATURES]

# ── Precompute weather_clim ───────────────────────────────────────────────────
hist = history.reset_index().rename(columns={"datetime": "date"})
hist["month"] = hist["date"].dt.month
hist["day"]   = hist["date"].dt.day
hist["hour"]  = hist["date"].dt.hour

weather_clim = (
    hist
    .groupby(["month","day","hour"])[["dew","temp","press","wnd_spd","snow","rain"]]
    .mean()
    .reset_index()
)

# ── Precompute pollution stats ────────────────────────────────────────────────
stats = (
    hist
    .groupby(["month","day","hour"])["pollution"]
    .agg(mean="mean", std="std")
    .reset_index()
)

# ── Build poll_grouped for sampling ────────────────────────────────────────────
poll_grouped = (
    hist
    .groupby(["month","day","hour"])["pollution"]
    .apply(lambda s: s.values)
    .to_dict()
)

# Function to update session state with preset values
def update_preset_values():
    selected_preset_key = st.session_state.p_preset
    preset_values = PRESETS.get(selected_preset_key, {})
    for k, v in preset_values.items():
        # Only update if the key exists in adv options and is part of the preset
        if f"adv_{k}" in st.session_state:
            st.session_state[f"adv_{k}"] = v
        # Special handling for wind direction if it's not prefixed with 'adv_'
        elif k == "wnd_dir":
            st.session_state["adv_wnd_dir"] = v
    st.session_state.input_changed = True # Also mark input as changed
    
# Get current time for the timestamp
current_time = datetime.now().strftime("%I:%M %p") # e.g., 06:31 PM

# ── 7) Streamlit layout ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .title-container {
        padding: 1rem 1.25rem;
        background-color: #2a3950;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .custom-title {
        color: white;
        font-size: 2.5rem !important;   /* ← force it */
        font-weight: 600;
        margin: 0;
    }
    </style>
    <div class="title-container">
        <h1 class="custom-title">🏭 Beijing Air Quality Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Air Quality Nowcast", "Historical Pollution Trends"])

# ── Tab 2: Forecast & Advice with sampling fallback ───────────────────────────
with tab2:
        st.markdown(
        """
        <style>
        .pm25-title {
            color: #E0E0E0; /* Tomato red - stands out for "health" or "alert" */
            font-size: 2.3rem !important;   /* ← force it */
            font-weight: 800; /* Extra bold */
            text-align: center; /* Center the title */
            margin-top: 3rem; /* Add some space above */
            margin-bottom: 2rem; /* Add some space below */
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E0E0E0; /* Subtle underline */
        }
         </style>
         <p class="pm25-title">📊 Past Pollution Readings & Insights 📈</p>
        """,
        unsafe_allow_html=True
    )

        # 1) Date input up to yesterday + hour
        st.markdown('<p class="input-section-title">📅 Time Selection</p>', unsafe_allow_html=True)
        with st.container(): 
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: 
                start_date = history.index.min().date()
                end_date   = datetime.now().date() - timedelta(days=2)  # Allow up to yesterday
                date_sel = st.date_input(
                    "Date",
                    value=datetime.now().date() - timedelta(days=2),  # Default to one day before today
                    min_value=start_date,
                    max_value=end_date,
                    key="p_date"
                )
            with col2:
                hour_sel = st.slider("Hour (0–23)", 0, 23,
                                    history.index.max().hour,
                                    key="p_hour")
            st.markdown('</div>', unsafe_allow_html=True) # Close custom container

        # Reset forecast visibility when any input changes
        if 'input_changed' not in st.session_state:
            st.session_state.input_changed = False
        if 'forecast_run' not in st.session_state: 
            st.session_state.forecast_run = False
        if 'df_pred' not in st.session_state: 
            st.session_state.df_pred = None
        if 'prev_state' not in st.session_state:
            st.session_state.prev_state = {} 

        current_state = {
            'date': date_sel,
            'hour': hour_sel,
            'days': st.session_state.get('p_days', 1),
            'warn': st.session_state.get('p_warn', 150),
            'preset': st.session_state.get('p_preset', list(PRESETS.keys())[0]),
            'adv_dew': st.session_state.get('adv_dew', 0.0),
            'adv_temp': st.session_state.get('adv_temp', 0.0),
            'adv_press': st.session_state.get('adv_press', 1013.0),
            'adv_wnd_spd': st.session_state.get('adv_wnd_spd', 0.0),
            'adv_snow': st.session_state.get('adv_snow', 0.0),
            'adv_rain': st.session_state.get('adv_rain', 0.0),
            'adv_wnd_dir': st.session_state.get('adv_wnd_dir', 'NE')
        }
        if 'prev_state' not in st.session_state:
            st.session_state.prev_state = current_state.copy()
        if current_state != st.session_state.prev_state:
            st.session_state.input_changed = True
            st.session_state.prev_state = current_state.copy()
            st.session_state.forecast_run = False
            st.session_state.df_pred = None

        dt0 = datetime.combine(date_sel, dtime(hour=hour_sel))

        # 2) Core knobs
        st.markdown('<p class="input-section-title">⚙ Forecast Parameters</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                days = st.slider("Days ahead", 1, 7, 1, key="p_days")
            with col4:
                warn = st.slider("Warning level (µg/m³)", 0, 300, 150, key="p_warn")

            st.selectbox("Quick weather scenario",
                     list(PRESETS.keys()),
                     key="p_preset",
                     on_change=update_preset_values) 

        st.markdown('</div>', unsafe_allow_html=True)

        # 4) Weather values from CSV or climatology
        if dt0 in history.index:
            row0 = history.loc[dt0]
            dew0, temp0    = row0["dew"],    row0["temp"]
            press0, wnd_spd0 = row0["press"], row0["wnd_spd"]
            snow0, rain0  = row0["snow"],   row0["rain"]
            wnd_dir0      = row0["wnd_dir"]
        else:
            m, d, h = dt0.month, dt0.day, dt0.hour
            wc = weather_clim.query("month==@m & day==@d & hour==@h")
            dew0     = float(wc.dew.values[0])    if not wc.empty else 0.0
            temp0    = float(wc.temp.values[0])   if not wc.empty else 0.0
            press0   = float(wc.press.values[0])  if not wc.empty else 1013.0
            wnd_spd0 = float(wc.wnd_spd.values[0])if not wc.empty else 0.0
            snow0    = float(wc.snow.values[0])   if not wc.empty else 0.0
            rain0    = float(wc.rain.values[0])   if not wc.empty else 0.0
            wnd_dir0 = "NE"

        # 5) Advanced defaults & expander
        defaults = dict(
            adv_dew=dew0, adv_temp=temp0,
            adv_press=press0, adv_wnd_spd=wnd_spd0,
            adv_snow=snow0, adv_rain=rain0,
            adv_wnd_dir=wnd_dir0
        )
        for k,v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        st.markdown('<p class="input-section-title">✨ Advanced Weather Conditions</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            with st.expander("Adjust specific weather parameters"):
                st.markdown("Use these sliders to fine-tune the weather conditions for the forecast.")
                col5, col6, col7 = st.columns(3)
                with col5:
                    dew0     = st.slider("Dew point (°C)",     -50.0,   40.0,
                                        step=0.5, key="adv_dew")
                    temp0    = st.slider("Temperature (°C)",   -30.0,   50.0,
                                        step=0.5, key="adv_temp")
                with col6:
                    press0   = st.slider("Pressure (hPa)",      900.0, 1100.0,
                                        step=1.0, key="adv_press")
                    wnd_spd0 = st.slider("Wind speed (m/s)",      0.0,  500.0,
                                        step=0.5, key="adv_wnd_spd")
                with col7:
                    snow0    = st.slider("Snow (mm)",            0.0,   50.0,
                                        step=1.0, key="adv_snow")
                    rain0    = st.slider("Rain (mm)",            0.0,  200.0,
                                        step=1.0, key="adv_rain")
                dirs      = ["NE","NW","SE","SW"]
                wnd_dir0  = st.selectbox("Wind direction", dirs,
                                        index=dirs.index(st.session_state["adv_wnd_dir"]),
                                        key="adv_wnd_dir")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider() 

        # 6) Sample‐pollution getter with consistent extrapolation
        def get_pollution(dt: datetime):
            if dt in history.index:
                return float(history.at[dt, "pollution"])
            # Extrapolate based on 2010-2014 trend with fixed seed for consistency
            years = history.index.year.unique()
            if len(years) > 1:
                from sklearn.linear_model import LinearRegression
                import numpy as np
                X = np.array(years).reshape(-1, 1)
                y = [history[history.index.year == y]['pollution'].mean() for y in years]
                model = LinearRegression().fit(X, y)
                years_ahead = dt.year - 2014
                base_value = model.predict([[dt.year]])[0]
                # Use fixed seed for reproducibility
                seed = hash((dt.year, dt.month, dt.day, dt.hour))
                rnd = random.Random(seed)
                seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * dt.month / 12)
                noise = rnd.uniform(-5, 5)  # Consistent noise per date
                return max(0, base_value * seasonal_factor + noise)
            key = (dt.month, dt.day, dt.hour)
            arr = poll_grouped.get(key)
            if arr is not None and len(arr):
                seed = hash((dt.year, dt.month, dt.day, dt.hour))
                rnd = random.Random(seed)
                base_value = float(rnd.choice(arr))
                noise = rnd.uniform(-5, 5)
                return base_value + noise
            row = clim.query("month==@dt.month & day==@dt.day & hour==@dt.hour")
            if not row.empty:
                base_value = float(row.climatology.values[0])
                if dt > datetime.now():
                    years_ahead = (dt - datetime.now()).days / 365
                    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * dt.month / 12)
                    trend_factor = 1 + (years_ahead * 0.03)
                    rnd = random.Random(hash(str(dt)))
                    noise = rnd.uniform(-10, 10)
                    return (base_value * seasonal_factor * trend_factor) + noise
                return base_value
            return 0.0

        def build_features_from_rec_with_unscaled_lags(rec):
            df = pd.DataFrame([rec])
            dt = rec["datetime"]
            df["hour"] = dt.hour
            df["day"] = dt.day
            df["month"] = dt.month
            df["years_since_start"] = dt.year - START_YEAR
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
            df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            for wd in ["NE", "NW", "SE", "SW"]:
                df[f"wnd_dir_{wd}"] = (df["wnd_dir"] == wd).astype(int)
            df["pollution_lag1"] = rec.get("pollution_lag1", 0.0)
            df["pollution_lag2"] = rec.get("pollution_lag2", 0.0)
            df["pollution_roll_mean3"] = rec.get("pollution_roll_mean3", 0.0)
            df = df.drop(columns=["wnd_dir", "datetime"], errors="ignore")
            all_num_cols = ["dew", "temp", "press", "wnd_spd", "snow", "rain", "pollution_lag1", "pollution_lag2", "pollution_roll_mean3"]
            df[all_num_cols] = scaler.transform(df[all_num_cols])
            return df[FEATURES]

        base = dict(
            datetime=dt0,
            dew=dew0, temp=temp0, press=press0,
            wnd_spd=wnd_spd0, snow=snow0, rain=rain0,
            wnd_dir=wnd_dir0
        )
        base.update(PRESETS.get(st.session_state["p_preset"], {}))
        initial_poll = get_pollution(dt0)  # Consistent via fixed seed
        # Use only historical lags when available
        prev = [float(history.at[dt0 - timedelta(hours=i), "pollution"]) if (dt0 - timedelta(hours=i)) in history.index else initial_poll 
                for i in (1, 2, 3)]
        base["pollution_lag1"] = prev[0]
        base["pollution_lag2"] = prev[1]
        base["pollution_roll_mean3"] = np.mean(prev)
        st.metric("Starting pollution reading", f"{initial_poll:.1f} µg/m³")

        if "forecast_run" not in st.session_state:
            st.session_state.forecast_run = False

        if st.button("Generate Forecast🚀", key="p_run"):
            st.session_state.forecast_run = True
            recs = []
            # Build sequence starting from selected hour
            current_dt = dt0
            if current_dt in history.index:
                poll_value = float(history.at[current_dt, "pollution"])
            else:
                poll_value = initial_poll
            recs.append({"datetime": current_dt, "forecast": poll_value, **base})
            # Extend with forecast for future hours
            current = base.copy()
            for i in range(days * 24):
                dt = dt0 + timedelta(hours=i + 1)
                current["datetime"] = dt
                if dt <= datetime.now() - timedelta(days=1):  # Past or present
                    if dt in history.index:
                        current["forecast"] = float(history.at[dt, "pollution"])
                    else:
                        current["forecast"] = get_pollution(dt)
                else:  # Future prediction
                    if i > 0:
                        current["temp"] += random.uniform(-0.5, 0.5)
                        current["wnd_spd"] += random.uniform(-0.5, 0.5)
                        current["dew"] += random.uniform(-0.5, 0.5)
                        current["press"] += random.uniform(-1, 1)
                    pred = float(rf_model.predict(build_features_from_rec_with_unscaled_lags(current))[0])
                    prev_value = recs[-1]["forecast"] if recs else initial_poll
                    current["forecast"] = max(min(pred, prev_value * 1.2), prev_value * 0.8)  # ±20% constraint
                recs.append(current.copy())
                current["pollution_lag1"] = current["forecast"]
                current["pollution_lag2"] = recs[-2]["forecast"] if len(recs) > 1 else base["pollution_lag1"]
                current["pollution_roll_mean3"] = np.mean([current["forecast"], current["pollution_lag1"], current["pollution_lag2"]])
            
            df = pd.DataFrame(recs)
            aqi_info = df["forecast"].apply(to_aqi).tolist()
            df[["aqi", "color", "tip"]] = pd.DataFrame(aqi_info, index=df.index)
            st.session_state.df_pred = df

        df_pred = st.session_state.df_pred
        if st.session_state.forecast_run and df_pred is not None:
            st.markdown('<p class="input-section-title">📊 Forecast Chart</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                # Filter to show only from the selected hour onward
                df_pred = df_pred[df_pred["datetime"] >= dt0]
                line = alt.Chart(df_pred).mark_line(color="steelblue").encode(
                    x=alt.X("datetime:T", title="Time"),
                    y=alt.Y("forecast:Q", title="Pollution (µg/m³)")
                ).properties(width=700, height=350)
                pts = alt.Chart(df_pred).mark_circle(size=50).encode(
                    x=alt.X("datetime:T", title="Time"),
                    y=alt.Y("forecast:Q", title="Pollution (µg/m³)"),
                    color=alt.condition(alt.datum.forecast > warn,
                                        alt.value("red"), alt.value("steelblue")),
                    tooltip=["datetime", alt.Tooltip("forecast", format=".1f", title="Pollution"), "aqi", "tip"]
                )
                st.altair_chart((line + pts).interactive(), use_container_width=True)

                hrs_over = int((df_pred["forecast"] > warn).sum())
                st.metric("Hours above warning", hrs_over, delta=f"{warn} µg/m³")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<p class="input-section-title">🗒 Daily Summary</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                # 6-H summary table
                dates = df_pred["datetime"].dt.date.unique().tolist()
                day_choice = st.selectbox("View summary for", dates, key="summary_day")
                df_day = df_pred[df_pred["datetime"].dt.date == day_choice]

                blocks = []
                for label, hrs in [("00–06", range(0,6)),
                                ("06–12", range(6,12)),
                                ("12–18", range(12,18)),
                                ("18–24", range(18,24))]:
                    seg = df_day[df_day["datetime"].dt.hour.isin(hrs)]["forecast"]
                    if seg.empty:
                        blocks.append((label, "-", "-", "-", "n/a", ""))
                    else:
                        mn, avg, mx = seg.min(), seg.mean(), seg.max()
                        cat, _, adv = to_aqi(avg) # to_aqi returns category, color, advice
                        blocks.append((label,
                                    f"{mn:.1f}", f"{avg:.1f}", f"{mx:.1f}",
                                    cat, adv))
                df_b = pd.DataFrame(blocks, columns=["Time","Min","Avg","Max","AQI","Advice"])
                aqi_colors = {"Good":"#009966","Moderate":"#FFDE33",
                            "Unhealthy for Sensitive Groups":"#FF9933","Unhealthy":"#CC0033", # Updated "Unhealthy for Sensitive Groups"
                            "Very Unhealthy":"#660099","Hazardous":"#7E0023","n/a":"#dddddd"} # Added Hazardous
                html = "<table style='width:100%;border-collapse:collapse'><tr>" + \
                    "".join(f"<th style='padding:10px; border-bottom: none;'>{c}</th>" for c in df_b.columns) + \
                    "</tr>"
                for _, row in df_b.iterrows():
                    html += "<tr>"
                    for col in df_b.columns:
                        v = row[col]
                        if col == "AQI":
                            clr = aqi_colors.get(v, "#fff")
                            # Adjusted text color for better contrast on certain AQI colors
                            text_clr = "black" if v in ["Moderate", "Unhealthy for Sensitive Groups"] else "white"
                            html += f"<td style='padding:8px;text-align:center;background:{clr};color:{text_clr}'>{v}</td>"
                        else:
                            html += f"<td style='padding:8px;text-align:center'>{v}</td>"
                    html += "</tr>"
                html += "</table>"
                st.markdown("### 6-Hour Summary")
                st.markdown(html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
# ── Tab 1: Next-Hour PM₂.₅ with Momentum, Wave Alerts & Exposure ────────────────────
with tab1:
        st.markdown(
            """
            <style>
            .pm25-title {
                color: #E0E0E0; /* Tomato red - stands out for "health" or "alert" */
                font-size: 2.3rem !important;   /* ← force it */
                font-weight: 800; /* Extra bold */
                text-align: center; /* Center the title */
                margin-top: 3rem; /* Add some space above */
                margin-bottom: 2rem; /* Add some space below */
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #E0E0E0; /* Subtle underline */
            }
            </style>
            <p class="pm25-title">😷 PM₂.₅ Forecast & Health Dashboard ❤️</p>
            """,
            unsafe_allow_html=True
        )

        # 1) Anchor “now”
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        st.markdown(f"*Data Referenced As Of:* {now:%Y-%m-%d %H:00}")

        # 2) AQI mapping + full info (Refactored for clarity and robustness)
        def to_aqi(pm: float, age=None, condition=None, activity=None):
            """Returns (label, color, advice, sensitive_advice, policy, personal_advice, risk_level) with personalized risk."""
            base_label, base_color, base_advice, sens_advice, policy = (
                ("Good", "#009966", "Air is clean; safe for all activities.", "No restrictions for sensitive groups.", "No policy actions needed.") if pm <= 35 else
                ("Moderate", "#FFDE33", "Take breaks during outdoor activities.", "Sensitive groups should limit exertion.", "Monitor emissions.") if pm <= 75 else
                ("Unhealthy", "#FF9933", "Wear a mask during outdoor activities.", "Sensitive groups avoid strenuous activity.", "Reduce traffic/industry.") if pm <= 115 else
                ("Very Unhealthy", "#CC0033", "Limit outdoor time; use air purifiers indoors.", "Sensitive groups stay indoors.", "Implement temporary restrictions.") if pm <= 150 else
                ("Hazardous", "#660099", "Stay indoors; avoid all outdoor activities.", "All groups should remain indoors.", "Enforce emission controls.")
            )

            personal_advice = base_advice
            risk_level = "Low"
            if age is not None and condition is not None and activity is not None:
                is_vulnerable_age = age < 10 or age > 65
                is_sensitive_condition = condition in ["Asthma", "Heart Disease", "COPD"]
                is_high_activity = activity == "High"

                if is_vulnerable_age or is_sensitive_condition or condition == "Allergies": risk_level = "Medium"
                if (is_vulnerable_age and is_sensitive_condition) or (is_sensitive_condition and is_high_activity): risk_level = "High"

                if risk_level != "Low":
                    advice_parts = [base_advice]
                    if condition in ["Asthma", "Heart Disease", "COPD", "Allergies"]:
                        advice_parts.append(f"Due to your {condition.lower()}, monitor for any symptoms.")
                    if is_high_activity:
                        advice_parts.append("Extra caution is recommended during high-intensity activities.")
                    if is_vulnerable_age and risk_level == "Medium":
                        advice_parts.append("Consider reducing prolonged exertion due to age.")
                    personal_advice = " ".join(advice_parts)
                
                if risk_level == "High" and pm > 115: personal_advice += " Seek medical advice if symptoms appear."
                elif risk_level == "Medium" and pm > 75: personal_advice += " Consult a doctor if needed."
                    
            return (base_label, base_color, base_advice, sens_advice, policy, personal_advice, risk_level)

        wave_threshold = 10

        # 3) Cache weather & features
        @st.cache_data
        def get_weather(dt):
            return fetch_weather(39.90, 116.40, dt)

        @st.cache_data
        def build_X(rec):
            return build_features_from_rec(rec)

        # 4) Session init
        if "df24" not in st.session_state:
            for k in ("df24", "conf_intervals", "next_val", "color", "hist_avg", "aqi_info"): 
                st.session_state[k] = None

        # 5) User input for target date and time
        min_date = now + timedelta(days=1)
        max_date = now + timedelta(days=30)
        target_date = st.date_input("Select target date", min_value=min_date, max_value=max_date, value=min_date)
        target_hour = st.selectbox("Select target hour", range(24), index=now.hour)
        target_datetime = datetime.combine(target_date, dtime(hour=target_hour, minute=0, second=0))
        hours_ahead = int((target_datetime - now).total_seconds() / 3600)

        # 6) Trigger prediction with consistent seed
        if st.button("Predict PM₂.₅"):
            # Set random seed based on target_datetime, constrained to valid range
            seed = hash(str(target_datetime)) % (2*32)  # Ensure seed is between 0 and 2*32 - 1
            random.seed(seed)
            np.random.seed(seed)

            # Baseline: Use most recent historical value if pool is empty
            pool = history.loc[
                (history.index.month == now.month) & 
                (history.index.day == now.day) & 
                (history.index.hour == now.hour), "pollution"]
            baseline = float(pool.mean()) if len(pool) > 0 else history["pollution"].iloc[-1]
            st.session_state.hist_avg = baseline

            wx = get_weather(now)  # Use current weather, but seed ensures consistency in random parts
            H = [baseline] * 3  # Initial lag history
            preds = []
            confs = []
            for i in range(1, hours_ahead + 1):
                dt = now + timedelta(hours=i)
                rec = {"datetime": dt, **wx,
                       "pollution_lag1": H[-1], "pollution_lag2": H[-2], "pollution_roll_mean3": np.mean(H[-3:])}
                X = build_X(rec)
                tree_preds = np.array([t.predict(X) for t in rf_model.estimators_])
                p = float(rf_model.predict(X.values)[0])
                ci = np.std(tree_preds) * 1.96
                H.append(p)
                preds.append({"datetime": dt, "pm25": p})
                confs.append({"datetime": dt, "ci_lower": max(0, p - ci), "ci_upper": p + ci})
            df24 = pd.DataFrame(preds)
            df_ci = pd.DataFrame(confs)
            df24['pm25_smoothed'] = df24['pm25'].rolling(6, min_periods=1).mean()
            st.session_state.df24, st.session_state.conf_intervals, st.session_state.next_val, st.session_state.color, st.session_state.aqi_info = (
                df24, df_ci, df24['pm25'].iloc[0], to_aqi(df24['pm25'].iloc[0])[1], to_aqi(df24['pm25'].iloc[0])
            )

        # 7) Display
        if st.session_state.df24 is not None:
            df24, df_ci = st.session_state.df24, st.session_state.conf_intervals
            next_val, color, hist_avg, aqi_info = (
                st.session_state.next_val, 
                st.session_state.color, 
                st.session_state.hist_avg,
                st.session_state.aqi_info
            )
            _, _, advice, sens_advice, policy, _, _ = aqi_info

            wx = get_weather(now)
            st.session_state.current_weather = wx

            if 'current_weather' in st.session_state:
                wx = st.session_state.current_weather

                st.markdown(
                    f"""
                    <style>
                        .weather-card {{
                            background-color: #2a2d40;
                            padding: 20px;
                            border-radius: 10px;
                            color: white;
                            font-family: Arial, sans-serif;
                            margin-top: 20px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.2);
                            width: 100%;
                            max-width: 800px;
                            margin-left: auto;
                            margin-right: auto;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }}
                        .weather-title {{
                            font-size: 1.5rem;
                            font-weight: bold;
                            margin-bottom: 20px;
                            text-align: center;
                            color: #4fc3f7;
                            width: 100%;
                        }}
                        .weather-row {{
                            display: flex;
                            justify-content: center;
                            flex-wrap: wrap;
                            gap: 15px;
                            width: 100%;
                            margin-bottom: 15px;
                        }}
                        .weather-row:last-child {{
                            margin-bottom: 0;
                        }}
                        .weather-item {{
                            background-color: #3b3e55;
                            height: 100px; /* Ensures equal box height */
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            padding: 10px;
                            border-radius: 8px;
                            box-sizing: border-box;
                            text-align: center;
                        }}
                        .weather-item h4 {{
                            margin: 2px 0 4px 0;
                            font-size: 1rem;
                            color: #f0f0f0;
                        }}
                        .weather-item p {{
                            margin: 0;
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #4fc3f7;
                        }}
                        /* Specific adjustments for the layout in the image */
                        .weather-row-first .weather-item {{
                            flex: 1 1 calc(25% - 11.25px); /* Allows 4 items to roughly fit by adapting */
                            max-width: calc(25% - 11.25px); /* Max width to ensure 4 per row */
                        }}
                        .weather-row-second {{
                            /* Override previous width to make it span the full 4-item width */
                            width: calc(4 * (25% - 11.25px) + 3 * 15px); /* Sum of 4 items + 3 gaps from the first row */
                            justify-content: space-around; /* Distribute the two items more evenly across the calculated width */
                        }}
                        .weather-row-second .weather-item {{
                            flex: 1 1 calc(50% - 7.5px); /* Two items to span the row, considering gap */
                            max-width: calc(50% - 7.5px); /* Max width for two items */
                        }}

                    </style>

                    <div class="weather-card">
                        <div class="weather-title">🌦️ Current Weather Conditions</div>
                        <div class="weather-row weather-row-first">
                            <div class="weather-item">
                                <h4>Temperature</h4>
                                <p>{wx.get('temp', 'N/A')} °C</p>
                            </div>
                            <div class="weather-item">
                                <h4>Wind Speed</h4>
                                <p>{wx.get('wnd_spd', 'N/A')} m/s</p>
                            </div>
                            <div class="weather-item">
                                <h4>Pressure</h4>
                                <p>{wx.get('press', 'N/A')} hPa</p>
                            </div>
                            <div class="weather-item">
                                <h4>Rainfall</h4>
                                <p>{wx.get('rain', 'N/A')} mm</p>
                            </div>
                        </div>
                        <div class="weather-row weather-row-second">
                            <div class="weather-item">
                                <h4>Snowfall</h4>
                                <p>{wx.get('snow', 'N/A')} mm</p>
                            </div>
                            <div class="weather-item">
                                <h4>Wind Direction</h4>
                                <p>{wx.get('wnd_dir', 'N/A')}</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # b) Plotly gauge (no changes)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=next_val,
                number={'font': {'size': 130}, 'valueformat': '.1f'},  # Ensure large, visible number
                gauge={
                    "axis": {"range": [0, max(300, next_val + 20)]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 35], "color": "#009966"},
                        {"range": [35, 75], "color": "#FFDE33"},
                        {"range": [75, 115], "color": "#FF9933"},
                        {"range": [115, 150], "color": "#CC0033"},
                        {"range": [150, 300], "color": "#660099"},
                    ],
                }
            ))
            fig_gauge.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # --- "What do the colors mean?" tooltip ---
            st.markdown(
                """
                <div style='text-align:center;margin-bottom:20px; font-size:1.1rem;'>
                    What do the colors mean?
                    <span class='tooltip'>ℹ️
                        <span class='tooltiptext'>
                            <strong>Air Quality Index (AQI) Levels:</strong><br><br>
                            <span style='color:#4CAF50;'>&#9724; Good (0–35)</span>: Air quality is satisfactory, and air pollution poses little or no risk.<br><br>
                            <span style='color:#FFEB3B;'>&#9724; Moderate (36–75)</span>: Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.<br><br>
                            <span style='color:#FF9800;'>&#9724; Unhealthy (76–115)</span>: Members of sensitive groups may experience health effects. The general public is less likely to be affected.<br><br>
                            <span style='color:#F44336;'>&#9724; Very Unhealthy (116–150)</span>: Health warnings of emergency conditions. The entire population is more likely to be affected.<br><br>
                            <span style='color:#9C27B0;'>&#9724; Hazardous (151+)</span>: Health alert: everyone may experience more serious health effects.
                        </span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # a) AQI card (user-friendly)
            st.markdown(
            f"""
            <style>
            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: pointer;
            }}

            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 260px;
                background-color: #2e2e2e;
                color: #fff;
                text-align: left;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%; 
                left: 50%;
                margin-left: -130px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.85rem;
            }}

            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
            </style>

            <div style='background:{color};padding:20px;border-radius:4px;color:white;text-align:center;width:100%'>
                <h2 style='margin:0'>{next_val:.1f} µg/m³</h2>
                <p style='margin:0;font-size:1.2rem'>
                    PM₂.₅ Level 
                    <span class="tooltip">ℹ️
                        <span class="tooltiptext">
                            PM₂.₅ stands for fine particulate matter smaller than 2.5 micrometers. 
                            µg/m³ means micrograms per cubic meter – showing how much of these tiny particles are present in the air you’re breathing.
                        </span>
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
            
            # b) Now/Next comparison (below AQI, equal-width boxes)
            # add bottom margin to create space before momentum
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;margin-top:10px;margin-bottom:12px'>" # Reduced margin-bottom
                f"<div style='flex:1;padding:16px;border:1px solid #444;border-radius:4px;color:#ddd;text-align:center;margin-right:5px'>" # Reduced margin-right
                f"<p style='margin:4px 0'><strong>Now:</strong> {hist_avg:.1f} µg/m³</p>"
                f"</div>"
                f"<div style='flex:1;padding:16px;border:1px solid #444;border-radius:4px;color:#ddd;text-align:center;margin-left:5px'>" # Reduced margin-left
                f"<p style='margin:4px 0'><strong>Next:</strong> {next_val:.1f} µg/m³</p>"
                f"</div></div>", unsafe_allow_html=True
            )

            # b) Pollution Momentum with compact layout
            momentum = next_val - hist_avg
            label_value = f"{momentum:+.1f} µg/m³"
            momentum_color = "red" if momentum > 0 else "green"
            momentum_icon = "📈" if momentum > 0 else "📉"

            # Determine the explanation text based on momentum
            if momentum > 0:
                explanation = "Air quality is predicted to worsen in the next hour."
            elif momentum < 0:
                explanation = "Air quality is predicted to improve in the next hour."
            else:
                explanation = "Air quality is predicted to remain stable in the next hour."

            st.markdown(
            f"""
            <style>
            /* Import Open Sans from Google Fonts - NOW INSIDE THIS BLOCK */
            @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

            /* General tooltip styles - NOW INSIDE THIS BLOCK */
            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: pointer;
            }}

            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 260px;
                background-color: #2e2e2e;
                color: #fff;
                text-align: left;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -130px;
                opacity: 0;
                transition: opacity 0.3s;
                font-family: 'Open Sans', sans-serif;
                font-size: 0.85rem;
                line-height: 1.4;
            }}

            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}

            /* Styles specific to the momentum card (already here) */
            .momentum-card-container {{
                width: 100%;
                display: flex;
                justify-content: center;
                margin-bottom: 30px;
            }}
            .momentum-card {{
                background-color: #2c3e50;
                color: white;
                padding: 16px 20px;
                border-radius: 10px;
                width: 100%;
                font-family: 'Open Sans', sans-serif; /* Changed to Open Sans for card content too */
                box-sizing: border-box;
                border: 1px solid #444;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 120px;
            }}
            .momentum-title {{
                font-weight: 600;
                font-size: 1.2rem;
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }}
            .momentum-value {{
                font-size: 1.5rem;
                font-weight: 600;
                color: {momentum_color};
                margin-top: 5px;
                margin-bottom: 10px;
            }}
            .momentum-explanation {{
                font-size: 0.95rem;
                color: #bbb;
                margin-bottom: 5px;
            }}
            .momentum-timestamp {{
                font-size: 0.8rem;
                color: #999;
                text-align: right;
            }}
            </style>

            <div class="momentum-card-container">
                <div class="momentum-card">
                    <div class="momentum-title">
                        Pollution Momentum
                        <span class="tooltip" style='margin-left: 6px;'>ℹ️
                            <span class="tooltiptext">
                                Pollution Momentum shows the change between the predicted next-hour PM₂.₅ level and the current reading.
                            </span>
                        </span>
                    </div>
                    <div class="momentum-value">{momentum_icon} {label_value}</div>
                    <div class="momentum-explanation">{explanation}</div>
                    <div class="momentum-timestamp">As of {current_time}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

            # c) Wave detection
            delta2 = df24['pm25'].iloc[1] - next_val if len(df24) > 1 else 0
            if momentum > wave_threshold and delta2 > wave_threshold:
                st.warning("⚠ Incoming Pollution Wave: sharp rise expected!")

            # d) 24-Hour Smoothed Trend
            st.subheader("PM₂.₅ Trend 📊")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df24['datetime'], y=df24['pm25_smoothed'],
                mode='lines+markers', name='Smoothed'
            ))
            fig.update_layout(
                xaxis_title='Date/Time',
                yaxis_title='PM₂.₅ (µg/m³)',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                width=700,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Activity Exposure & Risk🏃‍♂")
            duration = st.slider("Activity duration (min)", 1, 180, 30, step=1)
            hrs = duration / 60
            avg_c = df24['pm25_smoothed'].iloc[:int(np.ceil(hrs))].mean()
            exp = avg_c * hrs
            if exp < 10: risk, color_exp = "Low", "green"
            elif exp < 30: risk, color_exp = "Medium", "orange"
            else: risk, color_exp = "High", "red"
            # IMPROVEMENT: Cleaner, more prominent display for exposure results.
            st.markdown(
                f"<div style='padding:12px;border:1px solid {color_exp};border-radius:4px;text-align:center;'>"
                f"Estimated exposure over <strong>{duration} min</strong> is <strong>{exp:.1f} µg·hr/m³</strong>. "
                f"Risk Level: <strong style='color:{color_exp};'>{risk}</strong>"
                f"</div>", unsafe_allow_html=True)
            st.markdown("\n\n")
            st.markdown("\n\n")
            st.markdown("\n\n")

            st.subheader("Health & Policy Recommendations⚕")
            st.markdown("""
            <style>
            /* --- General Body & Page Styling (optional, assuming your Streamlit app already has a dark theme) --- */
            body {
                background-color: #1A1D2B; /* A very dark blue/purple background, similar to the image */
                color: #F0F0F0; /* Default text color for the page */
            }

            /* --- Table Container --- */
            .minimalist-table-container {
                width: 100%;
                overflow-x: auto; /* Ensures table is scrollable on smaller screens */
                margin-top: -30px; /* <--- CRITICAL CHANGE: Aggressive negative margin to pull it up */
                padding-top: 0px;
            }

            /* --- The Table Itself --- */
            .minimalist-table {
                width: 100%;
                border-collapse: collapse;
                border-spacing: 0;
                font-family: 'Inter', 'Segoe UI', 'Roboto', Arial, sans-serif;
                color: #E0E0E0;
                background-color: transparent;
                border: none;
                outline: none;
            }

            /* --- Table Header Cells (<th>) --- */
            .minimalist-table thead th {
                background-color: transparent; /* No background on the th itself */
                text-align: left;
                padding: 15px 25px 15px 25px; /* Ample padding for header cells */
                border: none; /* Removed ALL borders from header cells */
                font-weight: 600;
                text-transform: uppercase;
                /* font-size and letter-spacing are now primarily on .header-oval-bg */
                vertical-align: middle;
            }

            /* --- Oval Background for Header Text --- */
            .header-oval-bg {
                display: inline-block;
                background-color: #3A3D4A; /* A darker grey for the oval background */
                color: #F0F0F0; /* Lighter text color for the oval */
                padding: 10px 18px; /* <--- Adjusted padding for slightly larger oval */
                border-radius: 25px; /* <--- Adjusted border-radius for larger oval */
                line-height: 1;
                font-size: 0.95em; /* <--- CRITICAL CHANGE: Made the font size bigger here */
                letter-spacing: 0.08em; /* <--- Optional: Increased letter spacing a bit for larger text */
            }

            /* --- Table Body Rows --- */
            .minimalist-table tbody tr {
                background-color: #282A3A; /* Background for each row */
                border-radius: 0; /* Removed border-radius from rows */
                box-shadow: none; /* Removed all shadows from rows */
                transition: all 0.1s ease-in-out; /* Keep a subtle transition for hover */
            }

            .minimalist-table tbody td {
                padding: 20px 25px; /* Generous padding inside content cells */
                border: none; /* Removed ALL borders from body cells */
                background-color: transparent; /* Cells are transparent to show row background */
                vertical-align: middle;
            }

            /* --- Ensure the first/last cells in a row are no longer rounded --- */
            .minimalist-table tbody tr td:first-child,
            .minimalist-table tbody tr td:last-child {
                border-radius: 0;
            }

            /* Hover effect for rows */
            .minimalist-table tbody tr:hover {
                background-color: #303345; /* Slightly lighter background on hover */
                transform: none; /* Removed lift effect */
                box-shadow: none; /* No shadow on hover */
                cursor: default;
            }

            /* --- Optional: Add a subtle bottom border to rows if you want separation without gaps --- */
            .minimalist-table tbody tr:not(:last-child) {
                border-bottom: 1px solid #3A3D4A;
            }

            </style>
            """, unsafe_allow_html=True)

            html = f"""
            <div class="minimalist-table-container">
                <table class="minimalist-table">
                    <thead>
                        <tr>
                            <th><span class="header-oval-bg">Category</span></th>
                            <th><span class="header-oval-bg">Recommendation</span></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>General Public</td><td>{advice}</td></tr>
                        <tr><td>Sensitive Groups</td><td>{sens_advice}</td></tr>
                        <tr><td>Policy Actions</td><td>{policy}</td></tr>
                    </tbody>
                </table>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

            with st.expander("Personalized Health Risk Assessment"):
                age = st.slider("Age", 0, 100, 30, step=1, key="age_slider")
                condition = st.selectbox("Pre-existing Condition", ["None", "Asthma", "Heart Disease", "COPD", "Allergies"], key="cond_select")
                activity = st.selectbox("Activity Level", ["Sedentary", "Moderate", "High"], index=1, key="act_select")
                
                _, _, _, _, _, personal_advice, risk_level = to_aqi(next_val, age, condition, activity)
                risk_color_map = {"Low": "green", "Medium": "orange", "High": "red"}
                risk_color = risk_color_map.get(risk_level, "white")
                st.markdown(f"*Your Risk Level:* <strong style='color:{risk_color};'>{risk_level}</strong>", unsafe_allow_html=True)
                st.info(f"*Personalized Advice:* {personal_advice}")

        else:
            st.info("Click 'Predict PM₂.₅' to generate the forecast.")
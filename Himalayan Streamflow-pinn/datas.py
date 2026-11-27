import numpy as np
import pandas as pd
import requests

BASINS = {
    "karnali_train": {"lat": 28.64, "lon": 81.28, "area_km2": 43679.0, "snap": True},
    "koshi_test1": {"lat": 26.87, "lon": 87.16, "area_km2": 54100.0, "snap": True},
    "babai_test2": {"lat": 28.35, "lon": 81.72, "area_km2": 2570.0, "snap": False},
}

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def fetch_main_channel_discharge(lat: float, lon: float, start_date: str, end_date: str, snap: bool = True) -> pd.DataFrame:
    """Scans a 3x3 (~5km) neighborhood to snap onto the main river channel cell in GloFAS."""
    offsets = [-0.05, 0.0, 0.05] if snap else [0.0]
    best_df = None
    best_mean_q = -1.0

    for dlat in offsets:
        for dlon in offsets:
            url = (
                f"https://flood-api.open-meteo.com/v1/flood?"
                f"latitude={lat + dlat:.3f}&longitude={lon + dlon:.3f}"
                f"&daily=river_discharge&start_date={start_date}&end_date={end_date}"
            )
            resp = requests.get(url, timeout=30).json()
            if "daily" not in resp:
                continue
            df_q = pd.DataFrame(resp["daily"]).rename(
                columns={"time": "date",
                         "river_discharge": "streamflow_cumecs"}
            )
            mean_q = float(df_q["streamflow_cumecs"].mean())
            if mean_q > best_mean_q:
                best_mean_q = mean_q
                best_df = df_q

    return best_df


for name, cfg in BASINS.items():
    df_q = fetch_main_channel_discharge(
        cfg["lat"], cfg["lon"], START_DATE, END_DATE, snap=cfg["snap"])

    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={cfg['lat']}&longitude={cfg['lon']}"
        f"&daily=precipitation_sum,temperature_2m_mean,et0_fao_evapotranspiration,"
        f"shortwave_radiation_sum,relative_humidity_2m_mean,soil_moisture_0_to_7cm_mean"
        f"&start_date={START_DATE}&end_date={END_DATE}&timezone=Asia%2FKathmandu"
    )
    w_data = requests.get(weather_url, timeout=30).json()["daily"]
    df_w = pd.DataFrame(w_data).rename(
        columns={
            "time": "date",
            "precipitation_sum": "rainfall_mm",
            "temperature_2m_mean": "temp_c",
            "et0_fao_evapotranspiration": "evapotranspiration_mm",
            "shortwave_radiation_sum": "solar_radiation_mj",
            "relative_humidity_2m_mean": "humidity_pct",
            "soil_moisture_0_to_7cm_mean": "soil_moisture",
        }
    )

    df = pd.merge(df_w, df_q, on="date").dropna()
    df["basin_area_km2"] = cfg["area_km2"]
    df["specific_discharge_mm"] = (
        df["streamflow_cumecs"] * 86.4) / cfg["area_km2"]

    filename = f"{name}_2020_2024.csv"
    df.to_csv(filename, index=False)
    print(
        f"Saved {filename} ({len(df)} rows) | Mean Q: {df['streamflow_cumecs'].mean():.1f} m3/s")

"""
Tarvittavat paketit (asennus terminaalissa):

    pip install streamlit
    pip install pandas
    pip install numpy
    pip install scipy
    pip install plotly
    pip install folium
    pip install streamlit-folium

Sovellus ajetaan projektikansiossa:

    python -m streamlit run app.py

Jos haluat ajaa sovelluksen suoraan GitHubista:

    streamlit run https://raw.githubusercontent.com/Web-ja-hybriditek-mobiiliohjelmoinnissa/Fysiikan_loppuprojekti/main/app.py


"""

import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import butter, filtfilt
import plotly.express as px

import folium
from streamlit_folium import st_folium


# -----------------------------
# PARAMETRIT
# -----------------------------

CUT_START = 40.0   # s, datan alusta leikattava pätkä
CUT_END = 5.0      # s, datan lopusta leikattava pätkä
CUTOFF_HZ = 5.0    # lowpass cutoff

ACC_URL = "https://raw.githubusercontent.com/Web-ja-hybriditek-mobiiliohjelmoinnissa/Fysiikan_loppuprojekti/main/data/LinearAcceleration.csv"
GPS_URL = "https://raw.githubusercontent.com/Web-ja-hybriditek-mobiiliohjelmoinnissa/Fysiikan_loppuprojekti/main/data/Location.csv"


# -----------------------------
# APUFUNKTIOT: PERUS
# -----------------------------

def estimate_sampling_rate(time_array: np.ndarray) -> float:
    
    dt = np.diff(time_array)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0
    fs = 1.0 / np.median(dt)
    return fs


# -----------------------------
# APUFUNKTIOT: SIGNAALINKÄSITTELY
# -----------------------------

def butter_lowpass_filter(data, cutoff, fs, order=3):
    
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


# -----------------------------
# APUFUNKTIOT: TAJUUSANALYYSI (FFT/PSD)
# -----------------------------

def compute_psd_and_dominant_freq(signal: np.ndarray, fs: float):
    
    N = len(signal)
    dt = 1.0 / fs

    fourier = np.fft.fft(signal, N)
    psd = fourier * np.conj(fourier) / N
    freq = np.fft.fftfreq(N, dt)

    mask = freq > 0
    freq_pos = freq[mask]
    psd_pos = psd[mask].real

    if len(psd_pos) == 0:
        return freq_pos, psd_pos, 0.0

    idx = int(np.argmax(psd_pos))
    f_dom = float(freq_pos[idx])

    return freq_pos, psd_pos, f_dom


def count_steps_zero_crossings(signal: np.ndarray) -> int:
    
    crossings = (signal[:-1] < 0) & (signal[1:] >= 0)
    steps = int(np.sum(crossings))
    return steps


def estimate_steps_from_frequency(f_dom: float, duration_s: float) -> int:
    
    steps = int(np.round(f_dom * duration_s))
    return steps


# -----------------------------
# APUFUNKTIOT: GPS
# -----------------------------

def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000.0  # Maan säde metreissä

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    d = R * c
    return d


def compute_distance_and_speed(df_gps: pd.DataFrame):
    
    time_col = [c for c in df_gps.columns if "Time" in c][0]
    lat_col = [c for c in df_gps.columns if "Lat" in c][0]
    lon_col = [c for c in df_gps.columns if "Lon" in c][0]

    t = df_gps[time_col].to_numpy()
    lat = df_gps[lat_col].to_numpy()
    lon = df_gps[lon_col].to_numpy()

    distances = []
    for i in range(1, len(df_gps)):
        d = haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])
        distances.append(d)

    total_distance_m = float(np.sum(distances))

    if len(t) > 1:
        duration_s = float(t[-1] - t[0])
    else:
        duration_s = 0.0

    if duration_s > 0:
        mean_speed_m_s = total_distance_m / duration_s
    else:
        mean_speed_m_s = 0.0

    return total_distance_m, mean_speed_m_s, duration_s


# -----------------------------
# APUFUNKTIOT: DATAN LATAUS
# -----------------------------

def load_acceleration_data(acc_file):
    
    if acc_file is not None:
        df_acc = pd.read_csv(acc_file)
    else:
        df_acc = pd.read_csv(ACC_URL)

    df_acc = df_acc.dropna()
    return df_acc


def load_gps_data(gps_file):
    
    if gps_file is not None:
        df_gps = pd.read_csv(gps_file)
    else:
        df_gps = pd.read_csv(GPS_URL)

    df_gps = df_gps.dropna()
    return df_gps


# -----------------------------
# APUFUNKTIOT: KARTTA
# -----------------------------

def create_route_map(df_gps: pd.DataFrame) -> folium.Map:
    
    lat_col = [c for c in df_gps.columns if "Lat" in c][0]
    lon_col = [c for c in df_gps.columns if "Lon" in c][0]

    lat = df_gps[lat_col].to_numpy()
    lon = df_gps[lon_col].to_numpy()

    center_lat = float(np.mean(lat))
    center_lon = float(np.mean(lon))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    points = list(zip(lat, lon))
    folium.PolyLine(points, color="blue", weight=4, opacity=0.8).add_to(m)

    return m


# -----------------------------
# APUFUNKTIOT: OSIOEROTIN
# -----------------------------

def section_divider():
    
    st.markdown(
        "<hr style='border: 3px solid #666666; margin: 30px 0;'>",
        unsafe_allow_html=True
    )


# -----------------------------
# STREAMLIT-SOVELLUS
# -----------------------------

def main():
    st.set_page_config(
        page_title="Fysiikan loppuprojekti – Liikunta-analyysi",
        layout="wide"
    )

    st.markdown(
        "<h1 style='text-align: center;'>Fysiikan loppuprojekti – Liikunta-analyysi</h1>",
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown(
        "Tässä sovelluksessa analysoidaan Phyphoxilla mitattua kävelydataa. "
        "Kiihtyvyysdatan perusteella estimoidaan askeleiden määrä kahdella eri tavalla, "
        "ja GPS-datan perusteella lasketaan matka, keskinopeus ja askelpituus."
    )

    st.sidebar.header("Datan lataus")

    acc_file = st.sidebar.file_uploader(
        "Lataa LinearAcceleration.csv (tai käytä oletusdataa GitHubista)",
        type="csv"
    )
    gps_file = st.sidebar.file_uploader(
        "Lataa Location.csv (tai käytä oletusdataa GitHubista)",
        type="csv"
    )

    try:
        df_acc = load_acceleration_data(acc_file)
        df_gps = load_gps_data(gps_file)
    except Exception:
        st.error("Datan lataus epäonnistui. Tarkista, että CSV-linkit ja/tai uploadit ovat kunnossa.")
        return

    time_acc_col = [c for c in df_acc.columns if "Time" in c][0]
    time_gps_col = [c for c in df_gps.columns if "Time" in c][0]

    df_acc = df_acc[df_acc[time_acc_col] >= CUT_START].reset_index(drop=True)
    df_gps = df_gps[df_gps[time_gps_col] >= CUT_START].reset_index(drop=True)

    if len(df_acc) == 0 or len(df_gps) == 0:
        st.error("Aikaleikkaus poisti kaiken datan. Pienennä CUT_START-arvoa.")
        return

    last_time_acc = float(df_acc[time_acc_col].iloc[-1])
    end_limit_acc = last_time_acc - CUT_END
    df_acc = df_acc[df_acc[time_acc_col] <= end_limit_acc].reset_index(drop=True)

    last_time_gps = float(df_gps[time_gps_col].iloc[-1])
    end_limit_gps = last_time_gps - CUT_END
    df_gps = df_gps[df_gps[time_gps_col] <= end_limit_gps].reset_index(drop=True)

    t_acc = df_acc[time_acc_col].to_numpy()
    if len(t_acc) < 10:
        st.error("Kiihtyvyysdataa jäi liian vähän aikaleikkausten jälkeen.")
        return

    fs = estimate_sampling_rate(t_acc)
    duration_s = float(t_acc[-1] - t_acc[0])

    acc_col_y = None
    for c in df_acc.columns:
        if "linear acceleration" in c.lower() and "y" in c.lower():
            acc_col_y = c
            break

    if acc_col_y is None:
        st.error("Linear Acceleration Y -komponenttia ei löytynyt datasta.")
        return

    acc_raw_y = df_acc[acc_col_y].to_numpy()
    acc_filt_y = butter_lowpass_filter(acc_raw_y, cutoff=CUTOFF_HZ, fs=fs, order=3)

    freq_y, psd_y, f_dom_y = compute_psd_and_dominant_freq(acc_filt_y, fs)

    steps_time = count_steps_zero_crossings(acc_filt_y)
    steps_fft = estimate_steps_from_frequency(f_dom_y, duration_s)

    total_distance_m, mean_speed_m_s, duration_gps_s = compute_distance_and_speed(df_gps)

    if steps_time > 0:
        step_length_m = total_distance_m / steps_time
    else:
        step_length_m = 0.0

    st.subheader("Yhteenveto mitatusta suorituksesta")

    st.write(f"Askelmäärä (suodatettu signaali): **{steps_time:d}**")
    st.write(f"Askelmäärä (Fourier-analyysi): **{steps_fft:d}**")
    st.write(f"Matka: **{total_distance_m/1000:.2f} km**")
    st.write(f"Keskinopeus: **{mean_speed_m_s:.2f} m/s ({mean_speed_m_s*3.6:.1f} km/h)**")
    st.write(f"Askelpituus: **{step_length_m:.2f} m**")

    st.markdown(
        f"Askelanalyysi on tehty **Y-komponentin** perusteella. "
        f"Tehokkain taajuus tehospektristä: **{f_dom_y:.2f} Hz**. "
        f"Mittauksen kesto kiihtyvyysdatan perusteella: **{duration_s:.1f} s**."
    )
    st.caption(
        f"Mittauksen alusta poistettiin {CUT_START:.0f} sekuntia ja lopusta {CUT_END:.0f} sekuntia, "
        "jotta GPS ehtii lukittua ja kävely on tasaisessa vaiheessa."
    )

    section_divider()

    st.subheader("Suodatettu kiihtyvyysdata")

    df_plot_all = pd.DataFrame({
        "Time (s)": t_acc,
        "Filtered Acceleration Y (m/s²)": acc_filt_y
    })

    fig_all = px.line(
        df_plot_all,
        x="Time (s)",
        y="Filtered Acceleration Y (m/s²)",
        title="Suodatettu kiihtyvyys (Y) koko mittausajalta"
    )
    st.plotly_chart(fig_all, use_container_width=True)

    mask_window = (t_acc >= 100.0) & (t_acc <= 105.0)
    if np.sum(mask_window) >= 2:
        df_window = pd.DataFrame({
            "Time (s)": t_acc[mask_window],
            "Filtered Acceleration Y (m/s²)": acc_filt_y[mask_window]
        })
        fig_window = px.line(
            df_window,
            x="Time (s)",
            y="Filtered Acceleration Y (m/s²)",
            title="Suodatettu kiihtyvyys (Y), 100–105 s kohdalta"
        )
        st.plotly_chart(fig_window, use_container_width=True)
    else:
        st.info("Aikavälillä 100–105 s ei ole riittävästi datapisteitä zoomattua kuvaajaa varten.")

    st.caption(
        f"Kiihtyvyyden komponentti: {acc_col_y}. "
        f"Signaali on suodatettu lowpass-suodattimella (cutoff = {CUTOFF_HZ:.1f} Hz, order = 3), "
        "jotta askelten jaksollinen liike korostuu."
    )

    section_divider()

    st.subheader("Tehospektritiheys (PSD) – Y-komponentti")

    df_psd_y = pd.DataFrame({
        "Frequency (Hz)": freq_y,
        "Power": psd_y
    })

    fig_psd_all = px.line(
        df_psd_y,
        x="Frequency (Hz)",
        y="Power",
        title="Tehospektri suodatetulle kiihtyvyydelle (Y), koko alue"
    )
    st.plotly_chart(fig_psd_all, use_container_width=True)

    fig_psd_zoom = px.line(
        df_psd_y,
        x="Frequency (Hz)",
        y="Power",
        title="Tehospektri suodatetulle kiihtyvyydelle (Y), zoom 0–5 Hz"
    )
    fig_psd_zoom.update_xaxes(range=[0, 5])
    st.plotly_chart(fig_psd_zoom, use_container_width=True)

    st.caption(
        "Tehospektrin huipun taajuus vastaa askeltaajuutta. "
        "Askelmäärä Fourier-analyysin perusteella on arvioitu kertomalla tehokkain taajuus mittauksen kestolla."
    )

    section_divider()

    st.subheader("Reitti kartalla")

    route_map = create_route_map(df_gps)
    st_folium(route_map, width=800, height=500)

    st.caption(
        "Kartta on piirretty GPS-datan perusteella. "
        "Kuljettu matka on laskettu peräkkäisten pisteiden välisistä etäisyyksistä Haversinen kaavalla."
    )


if __name__ == "__main__":
    main()

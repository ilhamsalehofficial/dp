import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.set_page_config(page_title="Dashboard Clustering UMKM", layout="wide")
st.title("📊 Dashboard Interaktif Clustering dan Pemetaan UMKM")

# Upload file CSV
uploaded_file = st.file_uploader("📤 Unggah file CSV data UMKM", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    with st.expander("📋 Lihat Data Awal"):
        st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    st.subheader("⚙️ Pilih Variabel dan Konfigurasi Clustering")
    col1, col2 = st.columns(2)
    with col1:
        features = st.multiselect("🔢 Pilih kolom numerik:", options=numeric_cols, default=numeric_cols)
    with col2:
        k = st.slider("📍 Jumlah Cluster (K)", 2, 10, 3)

    if len(features) >= 2:
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        with st.expander("📁 Data Setelah Clustering"):
            st.dataframe(df, use_container_width=True)

        st.subheader("📈 Visualisasi Cluster (2D)")
        x_axis = st.selectbox("Sumbu X", features, index=0)
        y_axis = st.selectbox("Sumbu Y", features, index=1)

        palette = sns.color_palette("tab10", k)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette=palette, s=70, ax=ax)
        plt.title("Visualisasi Klaster UMKM", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Rekomendasi
        st.subheader("🧠 Rekomendasi Pengembangan UMKM per Klaster")
        cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
        st.dataframe(cluster_summary, use_container_width=True)

        with st.expander("📦 Detail Rekomendasi per Klaster"):
            for idx, row in cluster_summary.iterrows():
                st.markdown(f"#### 🟡 Klaster {int(row['Cluster'])}")
                rekomendasi = ""

                if 'Omzet_Bulanan' in row:
                    if row['Omzet_Bulanan'] < 10:
                        rekomendasi += "- Fokus pada pelatihan pemasaran digital dan branding produk.\n"
                    elif row['Omzet_Bulanan'] > 50:
                        rekomendasi += "- Siap untuk ekspansi usaha dan kemitraan distribusi regional.\n"
                    else:
                        rekomendasi += "- Dorong efisiensi produksi dan perluas akses ke permodalan.\n"

                if 'Jumlah_Karyawan' in row:
                    if row['Jumlah_Karyawan'] <= 2:
                        rekomendasi += "- Perlu perekrutan atau pelatihan staf tambahan.\n"
                    elif row['Jumlah_Karyawan'] >= 10:
                        rekomendasi += "- Implementasi sistem manajemen usaha dan SOP lebih lanjut.\n"

                if 'Lama_Usaha' in row:
                    if row['Lama_Usaha'] < 3:
                        rekomendasi += "- Pendampingan intensif dan inkubasi bisnis.\n"
                    elif row['Lama_Usaha'] >= 10:
                        rekomendasi += "- Siap untuk sertifikasi usaha dan masuk pasar nasional/internasional.\n"

                st.markdown(rekomendasi)

        # Pemetaan
        st.subheader("🗺️ Pemetaan UMKM Berdasarkan Lokasi")
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            cluster_colors = {
                0: [255, 0, 0],
                1: [0, 255, 0],
                2: [0, 0, 255],
                3: [255, 255, 0],
                4: [255, 0, 255],
                5: [0, 255, 255],
                6: [128, 0, 128],
                7: [255, 165, 0],
                8: [0, 128, 0],
                9: [128, 128, 0]
            }

            df['Cluster_color'] = df['Cluster'].apply(lambda x: cluster_colors.get(x, [100, 100, 100]))

            tooltip = {
                "html": "<b>Nama Usaha:</b> {Nama_Usaha}<br/>"
                        "<b>Omzet Bulanan:</b> {Omzet_Bulanan} Juta<br/>"
                        "<b>Jumlah Karyawan:</b> {Jumlah_Karyawan}<br/>"
                        "<b>Lama Usaha:</b> {Lama_Usaha} Tahun<br/>"
                        "<b>Klaster:</b> {Cluster}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }

            layer = pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[Longitude, Latitude]',
                get_color='Cluster_color',
                get_radius=200,
                pickable=True
            )

            view_state = pdk.ViewState(
                latitude=df['Latitude'].mean(),
                longitude=df['Longitude'].mean(),
                zoom=7.5,
                pitch=0,
            )

            map_chart = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )

            st.pydeck_chart(map_chart, use_container_width=True)

            with st.expander("🎨 Legenda Warna Klaster"):
                for cluster_id in sorted(df['Cluster'].unique()):
                    color = cluster_colors.get(cluster_id, [100, 100, 100])
                    hex_color = '#%02x%02x%02x' % tuple(color)
                    st.markdown(f"<span style='color:{hex_color}; font-weight:bold;'>●</span> Klaster {cluster_id}",
                                unsafe_allow_html=True)
        else:
            st.warning("📍 Data belum memiliki kolom 'Latitude' dan 'Longitude' untuk ditampilkan di peta.")
    else:
        st.warning("⚠️ Pilih minimal 2 variabel untuk clustering.")
else:
    st.info("⬆️ Silakan unggah file CSV terlebih dahulu.")

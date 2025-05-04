import pandas as pd
import folium
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ========================== Data Login Sementara
USERS = {
    "admin": "admin123",
    "ilham": "umkm2025"
}

def login(username, password):
    return USERS.get(username) == password

# ========================== Fungsi Halaman Utama
def main_app():
    st.title('📊 Pemetaan UMKM dengan K-Means dan SIG')

    wilayah_sumsel = [
        'Palembang', 'Prabumulih', 'Pagar Alam', 'Lahat', 'Muara Enim', 'Baturaja',
        'Tanjung Enim', 'Indralaya', 'Muaradua', 'Banyuasin', 'Ogan Ilir', 'Ogan Komering Ilir'
    ]

    st.sidebar.subheader("⚙️ Filter Wilayah")
    selected_wilayah = st.sidebar.selectbox("Pilih Wilayah", ["Semua Wilayah"] + wilayah_sumsel)

    jalan_wilayah = {
        'Palembang': ['Jalan Merdeka', 'Jalan Pahlawan', 'Jalan Jenderal Sudirman'],
        'Prabumulih': ['Jalan Raya', 'Jalan Sudirman'],
        'Pagar Alam': ['Jalan Pantai', 'Jalan Raya Pagar Alam'],
        'Lahat': ['Jalan Lahat Raya', 'Jalan Sudirman'],
        'Muara Enim': ['Jalan Muara Enim', 'Jalan Raya Enim'],
        'Baturaja': ['Jalan Baturaja Utara', 'Jalan Baturaja Selatan'],
        'Tanjung Enim': ['Jalan Tanjung Enim'],
        'Indralaya': ['Jalan Indralaya'],
        'Muaradua': ['Jalan Raya Muaradua'],
        'Banyuasin': ['Jalan Raya Banyuasin'],
        'Ogan Ilir': ['Jalan Ogan Ilir'],
        'Ogan Komering Ilir': ['Jalan Ogan Komering Ilir']
    }

    if selected_wilayah != "Semua Wilayah":
        selected_jalan = st.sidebar.selectbox(f"Pilih Jalan di {selected_wilayah}", jalan_wilayah[selected_wilayah])
    else:
        selected_jalan = None

    uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        required_columns = ['nama_umkm', 'latitude', 'longitude', 'omset_tahunan', 'kategori_usaha', 'kabupaten']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"❌ Kolom berikut tidak ditemukan di file CSV: {', '.join(missing_columns)}")
            st.stop()

        data = data.dropna(subset=required_columns)
        data = data.drop_duplicates()

        if selected_wilayah != "Semua Wilayah":
            data = data[data['kabupaten'] == selected_wilayah]

        st.subheader("🔍 Data UMKM")
        st.write(data)

        data['omset_asli'] = data['omset_tahunan']
        kategori_mapping = {'Perdagangan': 0, 'Jasa': 1, 'Manufaktur': 2}
        data['kategori_usaha_numerik'] = data['kategori_usaha'].map(kategori_mapping)

        if data['kategori_usaha_numerik'].isnull().any():
            st.warning("⚠️ Ada kategori usaha yang belum terdaftar dalam mapping.")

        scaler = StandardScaler()
        data['omset_tahunan_norm'] = scaler.fit_transform(data[['omset_tahunan']])

        n_samples = len(data)
        max_clusters = min(10, n_samples)
        st.sidebar.subheader("⚙️ Pengaturan Klaster")
        n_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=max_clusters, value=3)

        X = data[['omset_tahunan_norm', 'kategori_usaha_numerik']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(X)
        data['klaster'] = data['cluster'].apply(lambda x: f"Klaster {x + 1}")

        st.subheader("📊 Hasil Klasterisasi UMKM")
        st.write(data[['nama_umkm', 'omset_asli', 'kategori_usaha', 'klaster']])

        st.subheader("📦 Distribusi Omset per Klaster")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='klaster', y='omset_asli', data=data, palette='pastel')
        plt.xlabel("Klaster")
        plt.ylabel("Omset (Rp)")
        plt.title("Distribusi Omset Berdasarkan Klaster")
        st.pyplot(plt)

        st.subheader("📍 Visualisasi Klaster dan Centroid")
        centroids = kmeans.cluster_centers_
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='omset_tahunan_norm', y='kategori_usaha_numerik',
                        hue='klaster', data=data, palette='Set2', s=80)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroid')
        plt.xlabel("Omset (Ternormalisasi)")
        plt.ylabel("Kategori Usaha (Numerik)")
        plt.title("Hasil K-Means Clustering")
        plt.legend()
        st.pyplot(plt)

        st.subheader("🗺️ Peta Lokasi UMKM")
        cluster_colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred',
                          'cadetblue', 'darkgreen', 'pink', 'gray']
        map_center = [data['latitude'].mean(), data['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        for _, row in data.iterrows():
            cluster_idx = int(row['cluster']) % len(cluster_colors)
            color = cluster_colors[cluster_idx]
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=(f"Nama UMKM: {row['nama_umkm']}<br>"
                       f"Klaster: {row['klaster']}<br>"
                       f"Omset: Rp{int(row['omset_asli']):,}<br>"
                       f"Usaha: {row['kategori_usaha']}"),
                icon=folium.Icon(color=color)
            ).add_to(m)

        st.components.v1.html(m._repr_html_(), height=500)

        m.save("umkm_map.html")
        st.download_button("📥 Download Peta (HTML)", open("umkm_map.html", "r").read(),
                           file_name="umkm_map.html", mime="text/html")

        st.subheader("📁 Simpan Hasil Klasterisasi ke Excel")
        output_excel = BytesIO()
        data.to_excel(output_excel, index=False, sheet_name="Hasil Klaster")
        st.download_button(
            label="📥 Download Excel",
            data=output_excel.getvalue(),
            file_name="hasil_klaster_umkm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ========================== Aplikasi dengan Login
def main():
    st.set_page_config(page_title="Login UMKM", layout="centered")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Selamat datang, {username}!")
                st.rerun()
            else:
                st.error("Username atau password salah.")
    else:
        st.sidebar.success(f"Login sebagai: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        main_app()

if __name__ == "__main__":
    main()

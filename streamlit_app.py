import pandas as pd
import folium
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ========== Login Sederhana
USERS = {
    "admin": "admin123",
    "ilham": "umkm2025"
}

def login(username, password):
    return USERS.get(username) == password

# ========== Halaman Utama
def main_app():
    st.title('📊 Pemetaan UMKM dengan K-Means dan SIG')

    wilayah_sumsel = [
        'Palembang', 'Prabumulih', 'Pagar Alam', 'Lahat', 'Muara Enim', 'Baturaja',
        'Tanjung Enim', 'Indralaya', 'Muaradua', 'Banyuasin', 'Ogan Ilir', 'Ogan Komering Ilir'
    ]

    st.sidebar.subheader("⚙️ Filter Wilayah")
    selected_wilayah = st.sidebar.selectbox("Pilih Wilayah", ["Semua Wilayah"] + wilayah_sumsel)

    uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        required_columns = [
            'nama_umkm', 'no_telpon', 'alamat', 'latitude', 'longitude',
            'deskripsi', 'kategori_usaha', 'omset_tahunan', 'kabupaten'
        ]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            st.error(f"❌ Kolom tidak ditemukan: {', '.join(missing)}")
            st.stop()

        data = data.dropna(subset=required_columns)
        data = data.drop_duplicates()

        if selected_wilayah != "Semua Wilayah":
            data = data[data['kabupaten'] == selected_wilayah]

        st.subheader("📋 Data Lengkap UMKM")
        st.dataframe(data)

        # ========== Pra-pemrosesan Data
        data['omset_asli'] = data['omset_tahunan']

        kategori_mapping = {
            'Perdagangan': 0, 'Jasa': 1, 'Manufaktur': 2,
            'Makanan': 3, 'Fashion': 4, 'Kerajinan': 5
        }
        data['kategori_usaha_numerik'] = data['kategori_usaha'].map(kategori_mapping)

        if data['kategori_usaha_numerik'].isnull().any():
            st.warning("⚠️ Ada kategori usaha yang tidak dikenal. Tambahkan ke mapping.")

        scaler = StandardScaler()
        data['omset_tahunan_norm'] = scaler.fit_transform(data[['omset_tahunan']])

        # ========== Klasterisasi
        n_samples = len(data)
        max_clusters = min(10, n_samples)
        st.sidebar.subheader("⚙️ Pengaturan Klaster")
        n_clusters = st.sidebar.slider("Jumlah Klaster", 2, max_clusters, value=3)

        X = data[['omset_tahunan_norm', 'kategori_usaha_numerik']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(X)
        data['klaster'] = data['cluster'].apply(lambda x: f"Klaster {x + 1}")

        # ========== Tabel Hasil Klasterisasi
        st.subheader("📊 Hasil Klasterisasi UMKM")
        st.write(data[[
            'nama_umkm', 'alamat', 'no_telpon', 'deskripsi',
            'kategori_usaha', 'omset_asli', 'klaster'
        ]])

        # ========== Visualisasi Boxplot
        st.subheader("📦 Distribusi Omset per Klaster")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='klaster', y='omset_asli', data=data, palette='pastel')
        plt.xlabel("Klaster")
        plt.ylabel("Omset (Rp)")
        plt.title("Distribusi Omset Berdasarkan Klaster")
        st.pyplot(plt)

        # ========== Visualisasi Centroid
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

        # ========== Peta Lokasi
        st.subheader("🗺️ Peta Lokasi UMKM")
        cluster_colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred',
                          'cadetblue', 'darkgreen', 'pink', 'gray']
        map_center = [data['latitude'].mean(), data['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        for _, row in data.iterrows():
            color = cluster_colors[int(row['cluster']) % len(cluster_colors)]
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=(f"<b>{row['nama_umkm']}</b><br>"
                       f"Klaster: {row['klaster']}<br>"
                       f"Omset: Rp{int(row['omset_asli']):,}<br>"
                       f"Usaha: {row['kategori_usaha']}"),
                icon=folium.Icon(color=color)
            ).add_to(m)

        st.components.v1.html(m._repr_html_(), height=500)

        # ========== Download Peta
        m.save("umkm_map.html")
        st.download_button("📥 Download Peta (HTML)",
                           open("umkm_map.html", "r").read(),
                           file_name="umkm_map.html",
                           mime="text/html")

        # ========== Download Excel
        st.subheader("📁 Simpan Hasil Klasterisasi ke Excel")
        output_excel = BytesIO()
        data.to_excel(output_excel, index=False, sheet_name="Hasil Klaster")
        st.download_button(
            label="📥 Download Excel",
            data=output_excel.getvalue(),
            file_name="hasil_klaster_umkm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ========== Fungsi Login
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

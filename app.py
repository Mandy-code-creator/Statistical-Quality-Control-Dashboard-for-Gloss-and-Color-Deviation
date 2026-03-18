import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel Quality Dashboard - Color Lab", layout="wide")
st.title("📊 Hệ thống Phân tích Chất lượng & Thông số Màu sắc")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên các cột kỹ thuật (E, L, a, b cho cả 2 vị trí North/South)
        df = df.rename(columns={
            '生產日期': 'Ngay_SX',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_DELTA_E': 'dE_N', 'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'SOUTH_TOP_DELTA_E': 'dE_S', 'SOUTH_TOP_DELTA_L': 'dL_S', 'SOUTH_TOP_DELTA_A': 'da_S', 'SOUTH_TOP_DELTA_B': 'db_S',
            'NORTH_TOP_BLANCH': 'Gloss_N', 'SOUTH_TOP_BLANCH': 'Gloss_S',
            '光澤60度反射(下限)': 'LSL', '光澤60度反射(上限)': 'USL'
        })
        
        # Danh sách các cột cần chuyển sang dạng số
        cols_numeric = [
            'dE_N', 'dL_N', 'da_N', 'db_N', 
            'dE_S', 'dL_S', 'da_S', 'db_S',
            'Gloss_N', 'Gloss_S', 'LSL', 'USL'
        ]
        for col in cols_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce')
        
        # BƯỚC 1: TÍNH TRUNG BÌNH MỖI CUỘN (AVERAGE PER COIL)
        df['dE_Avg'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['dL_Avg'] = df[['dL_N', 'dL_S']].mean(axis=1)
        df['da_Avg'] = df[['da_N', 'da_S']].mean(axis=1)
        df['db_Avg'] = df[['db_N', 'db_S']].mean(axis=1)
        df['Gloss_Avg'] = df[['Gloss_N', 'Gloss_S']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.stop()

# --- 3. BỘ LỌC SIDEBAR ---
st.sidebar.header("🔍 Lọc theo Mã Sơn")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn (塗料編號):", list_ma_son)

df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. XỬ LÝ GỘP THEO BATCH_LOT (TRUNG BÌNH LẦN 2) ---
df_batch_summary = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max',
    'Ma_Son': 'first',
    'dE_Avg': 'mean',
    'dL_Avg': 'mean',
    'da_Avg': 'mean',
    'db_Avg': 'mean',
    'Gloss_Avg': 'mean',
    'LSL': 'first',
    'USL': 'first'
}).sort_values(by='Ngay_SX')

# --- 5. HIỂN THỊ ---
tab1, tab2 = st.tabs(["📋 Bảng Tổng hợp Thông số Màu", "📉 Biểu đồ Xu hướng"])

with tab1:
    st.subheader(f"Thông số chi tiết Batch cho mã: {ma_son_selected}")
    
    # Hiển thị Metric tổng quan
    m1, m2, m3 = st.columns(3)
    m1.metric("Số lượng Batch", len(df_batch_summary))
    m2.metric("dE Trung bình mã", f"{df_batch_summary['dE_Avg'].mean():.3f}")
    m3.metric("Gloss Trung bình mã", f"{df_batch_summary['Gloss_Avg'].mean():.1f}")

    st.markdown("**Bảng tổng hợp LAB (Trung bình cả lô):**")
    # Làm đẹp bảng dữ liệu
    st.dataframe(
        df_batch_summary.style.format({
            'dE_Avg': '{:.3f}', 'dL_Avg': '{:.3f}', 
            'da_Avg': '{:.3f}', 'db_Avg': '{:.3f}',
            'Gloss_Avg': '{:.2f}'
        }),
        use_container_width=True
    )

with tab2:
    if not df_batch_summary.empty:
        # Biểu đồ dE và LAB
        st.subheader("Phân tích biến động dE & Tọa độ màu (L, a, b)")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='dE_Avg', marker='o', label='Delta E', color='black', linewidth=2)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='dL_Avg', marker='x', label='Delta L', alpha=0.6)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='da_Avg', marker='.', label='Delta a', alpha=0.6)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='db_Avg', marker='.', label='Delta b', alpha=0.6)
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Biến động Lab theo Batch")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # Biểu đồ Gloss
        st.subheader("Phân tích biến động Độ bóng (Gloss)")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Gloss_Avg', marker='s', color='tab:blue')
        if pd.notna(df_batch_summary['LSL'].iloc[0]):
            ax2.axhline(df_batch_summary['LSL'].iloc[0], color='orange', label='LSL')
            ax2.axhline(df_batch_summary['USL'].iloc[0], color='orange', label='USL')
        plt.xticks(rotation=45)
        ax2.legend()
        st.pyplot(fig2)

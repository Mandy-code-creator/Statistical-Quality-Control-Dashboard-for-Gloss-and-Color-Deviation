import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="QA Color Dashboard", layout="wide")
st.title("📊 Hệ thống Phân tích Chỉ số Màu sắc & Độ bóng")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên các cột kỹ thuật
        df = df.rename(columns={
            '生產日期': 'Ngay_SX',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_DELTA_E': 'dE_N', 'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'SOUTH_TOP_DELTA_E': 'dE_S', 'SOUTH_TOP_DELTA_L': 'dL_S', 'SOUTH_TOP_DELTA_A': 'da_S', 'SOUTH_TOP_DELTA_B': 'db_S',
            'NORTH_TOP_BLANCH': 'Gloss_N', 'SOUTH_TOP_BLANCH': 'Gloss_S',
            '光澤60度反射(下限)': 'LSL', '光澤60度反射(上限)': 'USL'
        })
        
        cols_numeric = ['dE_N', 'dL_N', 'da_N', 'db_N', 'dE_S', 'dL_S', 'da_S', 'db_S', 'Gloss_N', 'Gloss_S', 'LSL', 'USL']
        for col in cols_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce')
        
        # BƯỚC 1: TÍNH TRUNG BÌNH MỖI CUỘN
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['ΔL'] = df[['dL_N', 'dL_S']].mean(axis=1)
        df['Δa'] = df[['da_N', 'da_S']].mean(axis=1)
        df['Δb'] = df[['db_N', 'db_S']].mean(axis=1)
        df['Gloss'] = df[['Gloss_N', 'Gloss_S']].mean(axis=1)
        
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
# Dùng ký hiệu Delta trực tiếp trong tên cột để hiển thị lên bảng
df_batch_summary = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max',
    'Ma_Son': 'first',
    'ΔE': 'mean',
    'ΔL': 'mean',
    'Δa': 'mean',
    'Δb': 'mean',
    'Gloss': 'mean',
    'LSL': 'first',
    'USL': 'first'
}).sort_values(by='Ngay_SX')

# --- 5. HIỂN THỊ ---
tab1, tab2 = st.tabs(["📋 Bảng Tổng hợp Lab", "📉 Biểu đồ Xu hướng"])

with tab1:
    st.subheader(f"Thông số Batch chi tiết: {ma_son_selected}")
    
    # Metrics tổng quan
    m1, m2, m3 = st.columns(3)
    m1.metric("Số lượng Batch", len(df_batch_summary))
    m2.metric("ΔE Trung bình", f"{df_batch_summary['ΔE'].mean():.3f}")
    m3.metric("Gloss Trung bình", f"{df_batch_summary['Gloss'].mean():.1f}")

    st.markdown("**Bảng tổng hợp chỉ số Δ (Trung bình lô):**")
    # Định dạng bảng với ký hiệu Delta
    st.dataframe(
        df_batch_summary[['Batch_Lot', 'Ngay_SX', 'ΔE', 'ΔL', 'Δa', 'Δb', 'Gloss', 'LSL', 'USL']].style.format({
            'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 
            'Δa': '{:.3f}', 'Δb': '{:.3f}',
            'Gloss': '{:.2f}', 'LSL': '{:.1f}', 'USL': '{:.1f}'
        }),
        use_container_width=True
    )

with tab2:
    if not df_batch_summary.empty:
        # Biểu đồ dE và LAB
        st.subheader("Phân tích biến động ΔE & Tọa độ màu (ΔL, Δa, Δb)")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='ΔE', marker='o', label='ΔE (Total)', color='black', linewidth=2.5)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='ΔL', marker='x', label='ΔL (Lightness)', alpha=0.7)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Δa', marker='.', label='Δa (Red-Green)', alpha=0.7)
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Δb', marker='.', label='Δb (Yellow-Blue)', alpha=0.7)
        
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Limit ΔE=1.0')
        ax1.set_title(f"Biến động các chỉ số Delta (Δ) - Mã {ma_son_selected}", fontsize=14)
        ax1.set_ylabel("Giá trị Delta")
        plt.xticks(rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)

        # Biểu đồ Gloss
        st.subheader("Phân tích biến động Độ bóng (Gloss)")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Gloss', marker='s', color='tab:blue', label='Gloss Avg')
        
        lsl_val = df_batch_summary['LSL'].iloc[0]
        usl_val = df_batch_summary['USL'].iloc[0]
        if pd.notna(lsl_val):
            ax2.axhline(lsl_val, color='orange', linestyle='--', label=f'LSL: {lsl_val}')
            ax2.axhline(usl_val, color='orange', linestyle='--', label=f'USL: {usl_val}')
            
        ax2.set_title("Kiểm soát Độ bóng theo Batch", fontsize=14)
        ax2.set_ylabel("Gloss (60°)")
        plt.xticks(rotation=45)
        ax2.legend()
        st.pyplot(fig2)

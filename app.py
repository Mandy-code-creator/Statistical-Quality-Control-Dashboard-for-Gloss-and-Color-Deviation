import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP CẤU HÌNH ---
st.set_page_config(page_title="QA Steel Dashboard", layout="wide")
st.title("📊 Hệ thống Kiểm soát Chất lượng Sơn Tôn")
st.markdown("---")

# --- 2. TẢI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên cột chuẩn
        df = df.rename(columns={
            '生產日期': 'Ngay_SX',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_BLANCH': 'Gloss_N', 
            'SOUTH_TOP_BLANCH': 'Gloss_S',
            'NORTH_TOP_DELTA_E': 'dE_N',
            'SOUTH_TOP_DELTA_E': 'dE_S',
            '光澤60度反射(下限)': 'LSL',
            '光澤60度反射(上限)': 'USL'
        })
        # Chuyển kiểu số
        for col in ['Gloss_N', 'Gloss_S', 'dE_N', 'dE_S', 'LSL', 'USL']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # BƯỚC 1: TRUNG BÌNH MỖI CUỘN (N + S) / 2
        df['Gloss_Coil_Avg'] = df[['Gloss_N', 'Gloss_S']].mean(axis=1)
        df['dE_Coil_Avg'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.stop()

# --- 3. SIDEBAR: LỌC THEO MÃ SƠN (BỎ LỌC MÀU) ---
st.sidebar.header("🔍 Bộ lọc Kỹ thuật")
# Lấy danh sách mã sơn duy nhất
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn (塗料編號):", list_ma_son)

# Lọc dữ liệu theo mã sơn đã chọn
df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. HIỂN THỊ TABS ---
tab1, tab2 = st.tabs(["📋 Bảng Dữ liệu Tổng hợp", "📉 Biểu đồ Xu hướng Batch"])

with tab1:
    st.subheader(f"Dữ liệu chi tiết cho mã: {ma_son_selected}")
    
    # Hiển thị bảng dữ liệu từng cuộn trước để Mandy kiểm tra
    st.markdown("**1. Chi tiết từng cuộn (Dữ liệu thô):**")
    st.dataframe(df_filtered[['Ngay_SX', 'Batch_Lot', 'Ma_Son', 'Gloss_N', 'Gloss_S', 'Gloss_Coil_Avg', 'dE_Coil_Avg']], use_container_width=True)

    # BƯỚC 2: TRUNG BÌNH LẦN 2 THEO BATCH
    # Quan trọng: Gộp theo cả Ngày và Batch để không bị sót
    df_batch_summary = df_filtered.groupby(['Ngay_SX', 'Batch_Lot'], as_index=False).agg({
        'Ma_Son': 'first',
        'Gloss_Coil_Avg': 'mean',
        'dE_Coil_Avg': 'mean',
        'LSL': 'first',
        'USL': 'first'
    }).rename(columns={'Gloss_Coil_Avg': 'Gloss_Batch_Avg', 'dE_Coil_Avg': 'dE_Batch_Avg'})

    st.markdown(f"**2. Tổng hợp theo từng Batch Lot (Đã tính trung bình lần 2):**")
    st.write(f"Tìm thấy **{len(df_batch_summary)}** Batch cho mã sơn này.")
    st.dataframe(df_batch_summary, use_container_width=True)

with tab2:
    if not df_batch_summary.empty:
        st.subheader(f"Biểu đồ kiểm soát mã {ma_son_selected}")
        
        # Vẽ biểu đồ dE và Gloss
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Chart dE
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='dE_Batch_Avg', marker='o', ax=ax1, color='red')
        ax1.axhline(1.0, color='black', linestyle='--', label='Target dE=1.0')
        ax1.set_title("Biến động dE theo từng Batch")
        ax1.tick_params(axis='x', rotation=45)
        
        # Chart Gloss
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Gloss_Batch_Avg', marker='s', ax=ax2, color='blue')
        # Vẽ LSL/USL nếu có
        if pd.notna(df_batch_summary['LSL'].iloc[0]):
            ax2.axhline(df_batch_summary['LSL'].iloc[0], color='orange', linestyle='--', label='LSL')
            ax2.axhline(df_batch_summary['USL'].iloc[0], color='orange', linestyle='--', label='USL')
        
        ax2.set_title("Biến động Độ bóng theo từng Batch")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

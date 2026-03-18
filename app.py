import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Strategic Dashboard", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU & BÓC TÁCH MÃ SƠN ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        col_mapping = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N'
        }
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_mapping[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_mapping[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_mapping)
        
        # --- LOGIC BÓC TÁCH MÃ SƠN (PJ6CD39ZS) ---
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str)

        # 2.1 Tách Nhà cung cấp (Ký tự thứ 2)
        vendor_dict = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        df['Nha_Cung_Cap'] = df['Ma_Son_Str'].str[1].map(vendor_dict).fillna("Unknown")

        # 2.2 Tách Loại Nhựa (Ký tự thứ 3)
        resin_dict = {
            '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC', '5': 'PVDF', 
            '6': 'SMP', '7': 'AC', '8': 'WB', '9': 'IP', 'A': 'PVB', 'B': 'PVF'
        }
        df['Loai_Nhua'] = df['Ma_Son_Str'].str[2].map(resin_dict).fillna("Other")

        # 2.3 Tách Nhóm Màu (Ký tự thứ 7)
        color_dict = {
            '0': 'Clear', '1': 'Red', 'R': 'Red', 'O': 'Orange', '2': 'Orange',
            'Y': 'Yellow', '3': 'Yellow', '4': 'Green', 'G': 'Green',
            '5': 'Blue', 'L': 'Blue', 'V': 'Violet', '6': 'Violet',
            'N': 'Brown', '7': 'Brown', 'T': 'White', 'H': 'White', 'W': 'White', '8': 'White',
            'A': 'Gray', 'C': 'Gray', '9': 'Gray', 'B': 'Black', 'S': 'Silver', 'M': 'Metallic'
        }
        df['Nhom_Mau'] = df['Ma_Son_Str'].str[6].map(color_dict).fillna("Other")

        # Xử lý số liệu
        cols_num = ['Gloss_Lab', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR MENU ---
with st.sidebar:
    st.title("📊 QA Smart Dashboard")
    st.markdown("---")
    view_mode = st.radio(
        "Lựa chọn View:",
        [
            "🏢 Supplier Benchmarking",
            "✨ Gloss Detailed Analysis",
            "🎨 Color Deviation Analysis"
        ]
    )
    st.markdown("---")
    st.write("**Thông tin giải mã:**")
    st.caption(f"Nhà cung cấp: {len(df_raw['Nha_Cung_Cap'].unique())}")
    st.caption(f"Loại nhựa: {len(df_raw['Loai_Nhua'].unique())}")
    st.caption(f"Nhóm màu: {len(df_raw['Nhom_Mau'].unique())}")

# --- 4. XỬ LÝ CÁC VIEW ---

# --- VIEW 1: SUPPLIER BENCHMARKING ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header("🏢 Supplier Benchmarking (So sánh tổng thể)")
    
    # Bộ lọc phụ cho View này
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        nhua_list = ['Tất cả'] + sorted(df_raw['Loai_Nhua'].unique().tolist())
        nhua_sel = st.selectbox("Lọc theo loại nhựa:", nhua_list)
    with c_f2:
        mau_list = ['Tất cả'] + sorted(df_raw['Nhom_Mau'].unique().tolist())
        mau_sel = st.selectbox("Lọc theo nhóm màu:", mau_list)
    
    df_bench = df_raw.copy()
    if nhua_sel != 'Tất cả': df_bench = df_bench[df_bench['Loai_Nhua'] == nhua_sel]
    if mau_sel != 'Tất cả': df_bench = df_bench[df_bench['Nhom_Mau'] == mau_sel]

    # PHÂN TÍCH ĐỘ BÓNG
    st.subheader("📊 So sánh Độ ổn định Gloss (Độ bóng)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_bench, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set2', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col2:
        st.write("**Bảng xếp hạng năng lực**")
        stats = df_bench.groupby('Nha_Cung_Cap')['Gloss_Lab'].agg(['mean', 'std']).dropna()
        stats.columns = ['Gloss TB', 'Độ lệch (Std)']
        st.dataframe(stats.sort_values('Độ lệch (Std)').style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

# --- VIEW 2: PHÂN TÍCH RIÊNG ĐỘ BÓNG ---
elif view_mode == "✨ Gloss Detailed Analysis":
    st.header("✨ Phân tích Độ bóng chi tiết")
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã sơn:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        st.markdown(f"**NCC:** `{df_sub['Nha_Cung_Cap'].iloc[0]}` | **Nhựa:** `{df_sub['Loai_Nhua'].iloc[0]}` | **Màu:** `{df_sub['Nhom_Mau'].iloc[0]}`")
        
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_sub, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab Gloss', linewidth=2)
        ax2.axhline(df_sub['Gloss_LSL'].iloc[0], color='red', ls='--')
        ax2.axhline(df_sub['Gloss_USL'].iloc[0], color='red', ls='--')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# --- VIEW 3: PHÂN TÍCH CHÊNH LỆCH MÀU ---
elif view_mode == "🎨 Color Deviation Analysis":
    st.header("🎨 Phân tích Biến động Màu sắc")
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã sơn:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Tọa độ (Δa / Δb)**")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=df_sub, x='da_N', y='db_N', size='ΔE', hue='ΔE', palette='coolwarm')
            ax3.axhline(0, color='black'); ax3.axvline(0, color='black')
            ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); st.pyplot(fig3)
        with col_b:
            st.write("**Độ sáng (ΔL)**")
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            sns.barplot(data=df_sub, x='Batch_Lot', y='dL_N', color='gray')
            ax4.axhline(0, color='red'); plt.xticks(rotation=90); st.pyplot(fig4)

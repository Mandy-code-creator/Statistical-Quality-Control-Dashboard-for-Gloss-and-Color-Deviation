import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Strategic Dashboard", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU & GIẢI MÃ (DECODER) ---
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
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str)

        # --- LOGIC GIẢI MÃ THEO YÊU CẦU CỦA MANDY ---
        
        # Ký tự 2: Nhà cung cấp
        vendor_dict = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        df['Nha_Cung_Cap'] = df['Ma_Son_Str'].str[1].str.upper().map(vendor_dict).fillna("Other Vendor")

        # Ký tự 3: Loại nhựa
        resin_dict = {
            '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC', '5': 'PVDF', 
            '6': 'SMP', '7': 'AC', '8': 'WB', '9': 'IP', 'A': 'PVB', 'B': 'PVF'
        }
        df['Loai_Nhua'] = df['Ma_Son_Str'].str[2].str.upper().map(resin_dict).fillna("Other Resin")

        # Ký tự 7: Nhóm màu
        color_dict = {
            '0': 'Clear', '1': 'Red', 'R': 'Red', 'O': 'Orange', '2': 'Orange',
            'Y': 'Yellow', '3': 'Yellow', '4': 'Green', 'G': 'Green',
            '5': 'Blue', 'L': 'Blue', 'V': 'Violet', '6': 'Violet',
            'N': 'Brown', '7': 'Brown', 'T': 'White', 'H': 'White', 'W': 'White', '8': 'White',
            'A': 'Gray', 'C': 'Gray', '9': 'Gray', 'B': 'Black', 'S': 'Silver', 'M': 'Metallic'
        }
        df['Nhom_Mau'] = df['Ma_Son_Str'].str[6].str.upper().map(color_dict).fillna("Other Color")

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
    st.title("📊 QA Smart System")
    st.markdown("---")
    view_mode = st.radio(
        "Chọn chế độ xem:",
        [
            "🏢 Supplier Benchmarking",
            "✨ Gloss Detailed Analysis",
            "🎨 Color Deviation Analysis"
        ]
    )
    st.markdown("---")
    st.write("**Thống kê nhanh:**")
    st.info(f"Đang có {df_raw['Nha_Cung_Cap'].nunique()} Nhà cung cấp")

# --- 4. XỬ LÝ CÁC VIEW ---

# --- VIEW 1: SUPPLIER BENCHMARKING (SO SÁNH TỔNG THỂ) ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header("🏢 Supplier Benchmarking (So sánh năng lực Nhà cung cấp)")
    
    # Bộ lọc phụ
    c1, c2 = st.columns(2)
    with c1:
        nhua_list = ['Tất cả'] + sorted(df_raw['Loai_Nhua'].unique().tolist())
        sel_nhua = st.selectbox("Lọc Loại nhựa:", nhua_list)
    with c2:
        mau_list = ['Tất cả'] + sorted(df_raw['Nhom_Mau'].unique().tolist())
        sel_mau = st.selectbox("Lọc Nhóm màu:", mau_list)

    df_bench = df_raw.copy()
    if sel_nhua != 'Tất cả': df_bench = df_bench[df_bench['Loai_Nhua'] == sel_nhua]
    if sel_mau != 'Tất cả': df_bench = df_bench[df_bench['Nhom_Mau'] == sel_mau]

    # BIỂU ĐỒ ĐỘ BÓNG
    st.subheader("📈 So sánh Độ ổn định Gloss")
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_bench, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set2', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col_g2:
        st.write("**Chỉ số ổn định (Std Dev)**")
        # Tính độ lệch chuẩn - Std càng thấp càng tốt
        stats = df_bench.groupby('Nha_Cung_Cap')['Gloss_Lab'].agg(['mean', 'std']).dropna()
        stats.columns = ['Gloss TB', 'Độ lệch (Std)']
        st.dataframe(stats.sort_values('Độ lệch (Std)').style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

# --- VIEW 2: PHÂN TÍCH RIÊNG ĐỘ BÓNG (CHI TIẾT MÃ SƠN) ---
elif view_mode == "✨ Gloss Detailed Analysis":
    st.header("✨ Phân tích Độ bóng Chi tiết theo Mã sơn")
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã sơn cần soi:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        # Thông tin giải mã
        st.success(f"📌 **NCC:** {df_sub['Nha_Cung_Cap'].iloc[0]} | **Nhựa:** {df_sub['Loai_Nhua'].iloc[0]} | **Màu:** {df_sub['Nhom_Mau'].iloc[0]}")
        
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_sub, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab Gloss', linewidth=2)
        ax2.axhline(df_sub['Gloss_LSL'].iloc[0], color='red', ls='--')
        ax2.axhline(df_sub['Gloss_USL'].iloc[0], color='red', ls='--')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# --- VIEW 3: PHÂN TÍCH CHÊNH LỆCH MÀU ---
elif view_mode == "🎨 Color Deviation Analysis":
    st.header("🎨 Phân tích Chênh lệch Màu sắc (ΔE)")
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã sơn:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Tọa độ màu (Δa / Δb)**")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=df_sub, x='da_N', y='db_N', size='ΔE', hue='ΔE', palette='Reds', ax=ax3)
            ax3.axhline(0, color='black'); ax3.axvline(0, color='black')
            ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); st.pyplot(fig3)
        with col_b:
            st.write("**Độ sáng (ΔL)**")
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            sns.barplot(data=df_sub, x='Batch_Lot', y='dL_N', palette='Greys_d', ax=ax4)
            ax4.axhline(0, color='red'); plt.xticks(rotation=90); st.pyplot(fig4)

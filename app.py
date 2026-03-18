import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Multi-Vendor Analysis", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU & GIẢI MÃ ---
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
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper()

        # GIẢI MÃ VENDOR (Ký tự 2)
        v_map = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        df['Nha_Cung_Cap'] = df['Ma_Son_Str'].str[1].map(v_map)
        df['Ma_Mau_4So'] = df['Ma_Son_Str'].str[-4:] # Tách 4 ký tự cuối

        # Ép kiểu số
        cols_num = ['Gloss_Lab', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        return df.dropna(subset=['Nha_Cung_Cap'])
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ QA Strategy Dashboard")
    view_mode = st.radio("Chế độ phân tích:", ["🏢 Supplier Benchmarking", "📊 Trend & Detail"])

# --- 4. VIEW 1: SUPPLIER BENCHMARKING (SO SÁNH ĐA VENDOR) ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header("🏢 Đối soát năng lực giữa các Nhà cung cấp")
    
    # 1. Chọn mã màu gốc
    list_4so = sorted(df_raw['Ma_Mau_4So'].unique().tolist())
    sel_4so = st.selectbox("🎯 Chọn Mã màu gốc (4 số cuối):", list_4so)
    
    # 2. Lọc dữ liệu theo mã màu 4 số cuối (Lấy TẤT CẢ các vendor cung cấp mã này)
    df_b = df_raw[df_raw['Ma_Mau_4So'] == sel_4so].copy()
    
    if not df_b.empty:
        # Lấy giới hạn chung (Min LSL và Max USL của tất cả các vendor)
        common_lsl = df_b['Gloss_LSL'].min()
        common_usl = df_b['Gloss_USL'].max()
        
        st.write(f"👉 Tìm thấy **{df_b['Nha_Cung_Cap'].nunique()}** nhà cung cấp cho mã màu `{sel_4so}`")
        
        col_c, col_t = st.columns([2, 1])
        with col_c:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            # Boxplot hiện thị tất cả Vendor
            sns.boxplot(data=df_b, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set2', ax=ax1)
            
            # Vẽ giới hạn kiểm soát
            ax1.axhline(common_lsl, color='red', linestyle='--', label=f'LSL: {common_lsl}')
            ax1.axhline(common_usl, color='red', linestyle='--', label=f'USL: {common_usl}')
            ax1.fill_between([-0.5, len(df_b['Nha_Cung_Cap'].unique())-0.5], common_lsl, common_usl, color='green', alpha=0.05)
            
            plt.xticks(rotation=15)
            plt.legend()
            st.pyplot(fig1)
            
        with col_t:
            st.write("**Bảng so sánh chi tiết**")
            # Tính % đạt tiêu chuẩn của từng Vendor cho mã màu này
            df_b['Is_Pass'] = (df_b['Gloss_Lab'] >= df_b['Gloss_LSL']) & (df_b['Gloss_Lab'] <= df_b['Gloss_USL'])
            vendor_comparison = df_b.groupby('Nha_Cung_Cap').agg({
                'Gloss_Lab': ['mean', 'std'],
                'Is_Pass': 'mean'
            })
            vendor_comparison.columns = ['Gloss TB', 'Độ biến động (Std)', 'Tỷ lệ đạt (%)']
            vendor_comparison['Tỷ lệ đạt (%)'] *= 100
            
            st.dataframe(vendor_comparison.sort_values('Tỷ lệ đạt (%)', ascending=False).style.format("{:.1f}"), use_container_width=True)

        st.markdown("---")
        # Phân tích ΔE cho các Vendor
        st.subheader("So sánh Sai lệch màu (ΔE) giữa các bên")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df_b, x='Nha_Cung_Cap', y='ΔE', palette='Reds', ax=ax2)
        ax2.axhline(1.0, color='black', linestyle='--', label='Ngưỡng 1.0')
        st.pyplot(fig2)

# --- 5. VIEW 2: TREND & DETAIL (GIỮ NGUYÊN ĐỂ SOI TỪNG LÔ) ---
elif view_mode == "📊 Trend & Detail":
    st.header("📊 Theo dõi xu hướng chi tiết theo Mã sơn đầy đủ")
    ma_son = st.selectbox("🎯 Chọn Mã sơn đầy đủ (Ví dụ: PJ6CD39ZS):", sorted(df_raw['Ma_Son'].unique()))
    df_s = df_raw[df_raw['Ma_Son'] == ma_son].copy()
    
    if not df_s.empty:
        st.info(f"📌 NCC: {df_s['Nha_Cung_Cap'].iloc[0]} | Loại nhựa: {df_s['Loai_Nhua'].iloc[0]}")
        
        # Biểu đồ Control Chart cho Gloss
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_s['Batch_Lot'], df_s['Gloss_Lab'], marker='o', linewidth=2)
        ax.axhline(df_s['Gloss_LSL'].iloc[0], color='red', label='LSL')
        ax.axhline(df_s['Gloss_USL'].iloc[0], color='red', label='USL')
        plt.xticks(rotation=45); plt.legend(); st.pyplot(fig)

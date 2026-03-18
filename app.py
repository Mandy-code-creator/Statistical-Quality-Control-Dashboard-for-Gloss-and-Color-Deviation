import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Control Chart Dashboard", layout="wide")

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

        # --- GIẢI MÃ VENDOR ---
        v_map = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        df['Nha_Cung_Cap'] = df['Ma_Son_Str'].str[1].map(v_map)
        df['Ma_Mau_4So'] = df['Ma_Son_Str'].str[-4:]

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
    st.title("🛡️ QA Control System")
    view_mode = st.radio("Chọn View:", ["🏢 Supplier Benchmarking", "✨ Gloss Control Chart", "🎨 Color Control Chart"])

# --- 4. VIEW 1: SUPPLIER BENCHMARKING (VỚI GIỚI HẠN) ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header("🏢 So sánh năng lực Vendor (4 ký tự cuối)")
    sel_4so = st.selectbox("🎯 Chọn Mã màu gốc:", sorted(df_raw['Ma_Mau_4So'].unique()))
    df_b = df_raw[df_raw['Ma_Mau_4So'] == sel_4so].copy()
    
    if not df_b.empty:
        # Lấy giới hạn chuẩn của mã màu này (giả định các vendor dùng chung spec)
        lsl = df_b['Gloss_LSL'].iloc[0]
        usl = df_b['Gloss_USL'].iloc[0]
        
        col_c, col_t = st.columns([2, 1])
        with col_c:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_b, x='Nha_Cung_Cap', y='Gloss_Lab', palette='vlag', ax=ax1)
            # Vẽ đường giới hạn LSL và USL
            ax1.axhline(lsl, color='red', linestyle='--', label=f'LSL: {lsl}')
            ax1.axhline(usl, color='red', linestyle='--', label=f'USL: {usl}')
            ax1.fill_between([-0.5, len(df_b['Nha_Cung_Cap'].unique())-0.5], lsl, usl, color='green', alpha=0.1, label='Vùng đạt')
            plt.legend(loc='upper right')
            st.pyplot(fig1)
            
        with col_t:
            st.write("**Bảng đối soát Vendor**")
            # Tính % đạt (Yield Rate) cho Gloss
            df_b['Gloss_Pass'] = (df_b['Gloss_Lab'] >= df_b['Gloss_LSL']) & (df_b['Gloss_Lab'] <= df_b['Gloss_USL'])
            stats = df_b.groupby('Nha_Cung_Cap').agg({'Gloss_Lab': 'std', 'Gloss_Pass': 'mean'}).dropna()
            stats['Gloss_Pass'] *= 100
            stats.columns = ['Độ lệch (Std)', 'Tỷ lệ đạt (%)']
            st.dataframe(stats.sort_values('Tỷ lệ đạt (%)', ascending=False).style.format("{:.1f}"), use_container_width=True)

# --- VIEW 2: GLOSS CONTROL CHART (CHI TIẾT LÔ) ---
elif view_mode == "✨ Gloss Control Chart":
    st.header("✨ Biểu đồ kiểm soát Độ bóng (Gloss Control Chart)")
    ma_son = st.selectbox("🎯 Chọn Mã sơn đầy đủ:", sorted(df_raw['Ma_Son'].unique()))
    df_s = df_raw[df_raw['Ma_Son'] == ma_son].copy()
    
    if not df_s.empty:
        lsl, usl = df_s['Gloss_LSL'].iloc[0], df_s['Gloss_USL'].iloc[0]
        target = (lsl + usl) / 2
        
        fig, ax = plt.subplots(figsize=(12, 5))
        # Vẽ dữ liệu thực tế
        ax.plot(df_s['Batch_Lot'], df_s['Gloss_Lab'], marker='o', color='#1f77b4', linewidth=2, label='Lab Gloss')
        # Vẽ các đường giới hạn
        ax.axhline(usl, color='red', linestyle='-', linewidth=2, label=f'USL ({usl})')
        ax.axhline(lsl, color='red', linestyle='-', linewidth=2, label=f'LSL ({lsl})')
        ax.axhline(target, color='green', linestyle=':', label=f'Target ({target})')
        
        # Tô màu vùng Out of Control
        ax.fill_between(range(len(df_s)), usl, df_s['Gloss_Lab'].max()+2, color='red', alpha=0.1)
        ax.fill_between(range(len(df_s)), lsl, df_s['Gloss_Lab'].min()-2, color='red', alpha=0.1)
        
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

# --- VIEW 3: COLOR CONTROL CHART (ΔE) ---
elif view_mode == "🎨 Color Control Chart":
    st.header("🎨 Biểu đồ kiểm soát Sai lệch màu (ΔE Control)")
    ma_son = st.selectbox("🎯 Chọn Mã sơn:", sorted(df_raw['Ma_Son'].unique()))
    df_c = df_raw[df_raw['Ma_Son'] == ma_son].copy()

    if not df_c.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(df_c['Batch_Lot'], df_c['ΔE'], color=['red' if x > 1.0 else 'skyblue' for x in df_c['ΔE']])
        ax.axhline(1.0, color='red', linestyle='--', label='Limit (ΔE = 1.0)')
        ax.axhline(0.7, color='orange', linestyle=':', label='Warning (0.7)')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
        st.caption("Cột màu đỏ = Vượt ngưỡng cho phép (ΔE > 1.0)")

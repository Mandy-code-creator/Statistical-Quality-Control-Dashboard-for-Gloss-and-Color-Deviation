import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA Dashboards", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU ---
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
        
        # --- TÁCH NHÀ CUNG CẤP (Ký tự thứ 2) ---
        df['Vendor_Code'] = df['Ma_Son'].astype(str).str[1]
        
        # Định nghĩa Vendor (Mandy tự sửa tên ở đây nhé)
        vendor_map = {'P': 'PPG', 'K': 'KCC', 'A': 'Akzo', 'V': 'Valspar', 'N': 'Nippon'}
        df['Nha_Cung_Cap'] = df['Vendor_Code'].map(vendor_map).fillna(df['Vendor_Code'])
        
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
    st.title("🛡️ Strategic Quality View")
    st.markdown("---")
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã màu:", list_ma_son)
    df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()
    
    st.markdown("---")
    view_mode = st.radio(
        "Chế độ phân tích:",
        ["🏢 Supplier Benchmarking", "📦 Inter-Batch Control", "📋 Raw Data"]
    )

# --- 4. XỬ LÝ VIEW ---

if view_mode == "🏢 Supplier Benchmarking":
    st.header(f"📊 So sánh Gloss & Color giữa các Nhà cung cấp")
    st.info("Phân tích dựa trên tất cả dữ liệu có cùng ký tự thứ 2 trong mã sơn.")

    # KHU VỰC PHÂN TÍCH ĐỘ BÓNG (GLOSS)
    st.subheader("1. Phân tích Độ bóng (Gloss Performance)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Biểu đồ Boxplot so sánh độ ổn định Gloss giữa các NCC
        fig_g, ax_g = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_raw, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set3', ax=ax_g)
        ax_g.set_title("Độ phân tán Gloss theo Nhà cung cấp")
        ax_g.set_ylabel("Gloss (Lab)")
        st.pyplot(fig_g)
    
    with col2:
        # Bảng chỉ số Gloss TB và Độ lệch chuẩn (Std)
        gloss_stats = df_raw.groupby('Nha_Cung_Cap')['Gloss_Lab'].agg(['mean', 'std', 'count']).reset_index()
        gloss_stats.columns = ['NCC', 'Gloss TB', 'Độ lệch (Std)', 'Số lô']
        st.write("**Chỉ số ổn định Gloss**")
        st.dataframe(gloss_stats.style.format({'Gloss TB': '{:.1f}', 'Độ lệch (Std)': '{:.2f}'}), use_container_width=True)
        st.caption("Std càng thấp = Chất lượng càng đồng nhất.")

    st.markdown("---")

    # KHU VỰC PHÂN TÍCH MÀU SẮC (COLOR)
    st.subheader("2. Phân tích Sai lệch màu (Color Performance)")
    col3, col4 = st.columns([2, 1])
    
    with col3:
        fig_c, ax_c = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_raw, x='Nha_Cung_Cap', y='ΔE', palette='Reds', ax=ax_c)
        ax_c.axhline(1.0, color='red', linestyle='--')
        ax_c.set_title("Sai lệch màu ΔE trung bình")
        st.pyplot(fig_c)
        
    with col4:
        st.write("**Tỷ lệ hàng đạt (ΔE ≤ 1.0)**")
        df_raw['Pass_Color'] = df_raw['ΔE'] <= 1.0
        pass_rate = df_raw.groupby('Nha_Cung_Cap')['Pass_Color'].mean() * 100
        st.table(pass_rate.rename("Tỷ lệ đạt (%)"))

elif view_mode == "📦 Inter-Batch Control":
    st.header(f"📦 Kiểm soát biến động lô - Mã: {ma_son_selected}")
    
    # Run Chart cho Gloss
    st.subheader("Xu hướng Gloss qua từng lô")
    fig_line, ax_line = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_filtered, x='Batch_Lot', y='Gloss_Lab', marker='o', color='blue', label='Gloss Lab')
    if not df_filtered.empty:
        ax_line.axhline(df_filtered['Gloss_LSL'].iloc[0], color='red', linestyle='--')
        ax_line.axhline(df_filtered['Gloss_USL'].iloc[0], color='red', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig_line)

    st.markdown("---")
    st.subheader("Xu hướng Sai lệch màu (ΔE)")
    fig_de, ax_de = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_filtered, x='Batch_Lot', y='ΔE', marker='s', color='red')
    ax_de.axhline(1.0, color='black', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig_de)

elif view_mode == "📋 Raw Data":
    st.subheader("Dữ liệu nguồn")
    st.dataframe(df_filtered, use_container_width=True)

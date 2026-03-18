import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA Strategic Dashboard", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        col_mapping = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N'
        }
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_mapping[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_mapping[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_mapping)
        
        # Tách Vendor từ ký tự thứ 2
        df['Vendor_Code'] = df['Ma_Son'].astype(str).str[1]
        vendor_map = {'P': 'PPG', 'K': 'KCC', 'A': 'Akzo', 'V': 'Valspar', 'N': 'Nippon'}
        df['Nha_Cung_Cap'] = df['Vendor_Code'].map(vendor_map).fillna(df['Vendor_Code'])
        
        # Ép kiểu số & Tính toán
        cols_num = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['Gloss_Line_Avg'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR MENU ---
with st.sidebar:
    st.title("🛡️ QA Executive View")
    st.markdown("---")
    view_mode = st.radio(
        "Lựa chọn phân tích:",
        [
            "🏢 Supplier Benchmarking",
            "✨ Gloss Detailed Analysis",
            "🎨 Color Deviation Analysis",
            "📋 Raw Data View"
        ]
    )
    st.markdown("---")
    st.info("💡 Ký tự thứ 2 của mã sơn được dùng để định danh Nhà cung cấp.")

# --- 4. XỬ LÝ CÁC VIEW ---

# --- VIEW 1: SUPPLIER BENCHMARKING (HIỆN ĐẦU TIÊN) ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header("🏢 Supplier Benchmarking (Tổng thể các Nhà cung cấp)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("So sánh ổn định Gloss")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_raw, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set2', ax=ax1)
        st.pyplot(fig1)
        st.caption("Biên độ hộp càng hẹp, NCC đó kiểm soát độ bóng càng ổn định.")

    with col2:
        st.subheader("Tỷ lệ đạt chuẩn Màu sắc (ΔE ≤ 1.0)")
        df_raw['Pass_Color'] = df_raw['ΔE'] <= 1.0
        pass_rate = df_raw.groupby('Nha_Cung_Cap')['Pass_Color'].mean().reset_index()
        pass_rate['Pass_Color'] *= 100
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pass_rate, x='Nha_Cung_Cap', y='Pass_Color', palette='Greens_r', ax=ax2)
        ax2.set_ylabel("Tỷ lệ đạt (%)")
        st.pyplot(fig2)

# --- VIEW 2: PHÂN TÍCH RIÊNG ĐỘ BÓNG ---
elif view_mode == "✨ Gloss Detailed Analysis":
    st.header("✨ Phân tích chi tiết Độ bóng (Gloss)")
    
    # Bộ chọn mã màu nằm bên trong View này
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã màu cần soi Gloss:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        g_lsl = df_sub['Gloss_LSL'].iloc[0]
        g_usl = df_sub['Gloss_USL'].iloc[0]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Gloss TB", f"{df_sub['Gloss_Lab'].mean():.1f}")
        m2.metric("Tiêu chuẩn", f"{g_lsl} - {g_usl}")
        
        # Tính Cpk Gloss
        std_g = df_sub['Gloss_Lab'].std()
        cpk = min((g_usl - df_sub['Gloss_Lab'].mean())/(3*std_g), (df_sub['Gloss_Lab'].mean() - g_lsl)/(3*std_g)) if std_g > 0 else 0
        m3.metric("Năng lực Cpk", f"{cpk:.2f}")

        st.subheader("Biểu đồ Xu hướng & Sai lệch Lab-Line")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_sub, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab', linewidth=2, ax=ax3)
        sns.lineplot(data=df_sub, x='Batch_Lot', y='Gloss_Line_Avg', marker='s', label='Line (Top Avg)', alpha=0.6, ax=ax3)
        ax3.axhline(g_lsl, color='red', ls='--'); ax3.axhline(g_usl, color='red', ls='--')
        plt.xticks(rotation=45)
        st.pyplot(fig3)

# --- VIEW 3: PHÂN TÍCH CHÊNH LỆCH MÀU ---
elif view_mode == "🎨 Color Deviation Analysis":
    st.header("🎨 Phân tích chi tiết Chênh lệch màu (Color)")
    
    # Bộ chọn mã màu nằm bên trong View này
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã màu cần soi Màu sắc:", list_ma_son)
    df_sub = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

    if not df_sub.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Biểu đồ Tọa độ màu (Δa vs Δb)**")
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=df_sub, x='da_N', y='db_N', size='ΔE', hue='ΔE', palette='Reds', ax=ax4)
            ax4.axhline(0, color='black', lw=1); ax4.axvline(0, color='black', lw=1)
            ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
            st.pyplot(fig4)
            st.caption("Chấm càng xa tâm (0,0) màu càng lệch.")

        with c2:
            st.write("**Biến động Độ sáng (ΔL) qua các lô**")
            fig5, ax5 = plt.subplots(figsize=(6, 6))
            colors = ['#444444' if x < 0 else '#aaaaaa' for x in df_sub['dL_N']]
            ax5.barh(df_sub['Batch_Lot'].astype(str), df_sub['dL_N'], color=colors)
            ax5.axvline(0, color='red', ls='--')
            st.pyplot(fig5)
            st.caption("Bên trái vạch đỏ: Tối hơn / Bên phải: Sáng hơn")

# --- VIEW 4: RAW DATA ---
elif view_mode == "📋 Raw Data View":
    st.subheader("Dữ liệu chi tiết tổng hợp")
    st.dataframe(df_raw, use_container_width=True)

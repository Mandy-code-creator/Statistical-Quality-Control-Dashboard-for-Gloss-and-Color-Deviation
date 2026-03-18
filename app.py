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
        
        # --- BÓC TÁCH NHÀ CUNG CẤP TỪ MÃ SƠN (KÝ TỰ THỨ 2) ---
        # Mandy lưu ý: str[1] trong Python chính là ký tự thứ 2 (đếm từ 0)
        df['Vendor_Code'] = df['Ma_Son'].astype(str).str[1]
        
        # Định nghĩa tên Nhà cung cấp (Mandy có thể sửa lại tên cho đúng thực tế nhé)
        vendor_map = {
            'P': 'PPG',
            'K': 'KCC',
            'A': 'AkzoNobel',
            'V': 'Valspar',
            'B': 'Beckers'
        }
        df['Nha_Cung_Cap'] = df['Vendor_Code'].map(vendor_map).fillna(df['Vendor_Code'])
        
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
    st.title("🛡️ Quality Strategic Analysis")
    st.markdown("---")
    
    # Lọc theo Mã sơn (Toàn bộ hoặc theo nhóm)
    list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
    ma_son_selected = st.selectbox("🎯 Chọn Mã màu phân tích:", list_ma_son)
    df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()
    
    st.markdown("---")
    view_mode = st.radio(
        "Phân tích theo cấp độ:",
        [
            "🏢 Supplier Benchmarking",
            "📦 Inter-Batch Color Control",
            "📋 Raw Data View"
        ]
    )

# --- 4. XỬ LÝ CÁC VIEW ---

if view_mode == "🏢 Supplier Benchmarking":
    st.header(f"So sánh năng lực Nhà cung cấp (Dựa trên ký tự thứ 2: {df_filtered['Vendor_Code'].unique()})")
    
    # Để so sánh khách quan, chúng ta nên so sánh trên toàn bộ dữ liệu (không chỉ 1 mã màu) 
    # để thấy năng lực chung của Vendor đó
    st.info("💡 Hệ thống đang so sánh tất cả các lô hàng của các Nhà cung cấp để tìm ra đơn vị ổn định nhất.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Độ ổn định Gloss (Boxplot)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_raw, x='Nha_Cung_Cap', y='Gloss_Lab', palette='viridis')
        st.pyplot(fig1)
        st.caption("Cột càng ngắn = Nhà cung cấp càng ổn định về độ bóng.")

    with c2:
        st.subheader("Sai lệch Màu sắc trung bình (ΔE)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        vendor_de = df_raw.groupby('Nha_Cung_Cap')['ΔE'].mean().reset_index()
        sns.barplot(data=vendor_de, x='Nha_Cung_Cap', y='ΔE', palette='Reds_d')
        ax2.axhline(1.0, color='red', linestyle='--', label='Limit')
        st.pyplot(fig2)
        st.caption("Cột càng thấp = Màu sắc càng sát với mẫu chuẩn.")

elif view_mode == "📦 Inter-Batch Color Control":
    st.header(f"Phân tích Biến động Màu sắc giữa các Lô (Batch-to-Batch)")
    
    # Lọc dữ liệu theo mã màu đã chọn
    df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
        'Ngay_SX': 'max', 'ΔE': 'mean', 'dL_N': 'mean', 'da_N': 'mean', 'db_N': 'mean', 'Nha_Cung_Cap': 'first'
    }).sort_values(by='Ngay_SX')

    st.subheader(f"Biểu đồ Run Chart ΔE - Vendor: {df_batch['Nha_Cung_Cap'].unique()}")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_batch, x='Batch_Lot', y='ΔE', marker='o', color='darkred', linewidth=2)
    ax3.axhline(0.7, color='orange', linestyle='--', label='Warning (0.7)')
    ax3.axhline(1.0, color='red', label='Reject (1.0)')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Ma trận Phân tích Hướng lệch màu (Color Matrix)")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Biểu đồ Tọa độ (Δa / Δb)**")
        fig4, ax4 = plt.subplots(figsize=(6,6))
        sns.scatterplot(data=df_batch, x='da_N', y='db_N', size='ΔE', hue='ΔE', palette='coolwarm')
        ax4.axhline(0, color='black', lw=1); ax4.axvline(0, color='black', lw=1)
        ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
        st.pyplot(fig4)

    with col_b:
        st.write("**Biến động Độ sáng (ΔL)**")
        fig5, ax5 = plt.subplots(figsize=(6,6))
        plt.bar(df_batch['Batch_Lot'].astype(str), df_batch['dL_N'], color='gray')
        ax5.axhline(0, color='red', linestyle='--')
        plt.xticks(rotation=90)
        st.pyplot(fig5)

elif view_mode == "📋 Raw Data View":
    st.subheader("Bảng dữ liệu đã bóc tách Vendor")
    st.dataframe(df_filtered[['Ngay_SX', 'Batch_Lot', 'Ma_Son', 'Nha_Cung_Cap', 'Gloss_Lab', 'ΔE']], use_container_width=True)

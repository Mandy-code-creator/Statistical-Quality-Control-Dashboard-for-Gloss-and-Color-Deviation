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
        # Giả sử cột '廠商' hoặc '塗料廠' là Nhà cung cấp. 
        # Nếu file của bạn dùng tên khác, hãy đổi 'Nha_Cung_Cap' bên dưới nhé.
        col_mapping = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '供應商': 'Nha_Cung_Cap', # Tên cột giả định cho Nhà cung cấp
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'SOUTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N'
        }
        # Tự động map LSL/USL
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_mapping[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_mapping[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_mapping)
        
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
    
    # Lọc Mã Màu chung (Ví dụ: Blue, White, Red...)
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

# --- VIEW 1: SO SÁNH NHÀ CUNG CẤP ---
if view_mode == "🏢 Supplier Benchmarking":
    st.header(f"So sánh năng lực Nhà cung cấp - Mã: {ma_son_selected}")
    
    if 'Nha_Cung_Cap' in df_filtered.columns:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("So sánh Độ ổn định Gloss")
            fig1, ax1 = plt.subplots()
            sns.boxplot(data=df_filtered, x='Nha_Cung_Cap', y='Gloss_Lab', palette='Set2')
            # Vẽ đường biên tiêu chuẩn
            lsl, usl = df_filtered['Gloss_LSL'].iloc[0], df_filtered['Gloss_USL'].iloc[0]
            ax1.axhline(lsl, color='red', linestyle='--')
            ax1.axhline(usl, color='red', linestyle='--')
            st.pyplot(fig1)
            st.caption("Biểu đồ hộp thể hiện độ phân tán Gloss. Hộp càng ngắn, NCC đó càng ổn định.")

        with c2:
            st.subheader("So sánh Sai lệch Màu (ΔE)")
            fig2, ax2 = plt.subplots()
            sns.barplot(data=df_filtered, x='Nha_Cung_Cap', y='ΔE', estimator='mean', palette='Reds')
            ax2.axhline(1.0, color='red', label='Limit (1.0)')
            st.pyplot(fig2)
            st.caption("NCC nào có cột thấp hơn thì màu sắc gần với mẫu chuẩn hơn.")
    else:
        st.warning("Dữ liệu của bạn chưa có cột 'Nha_Cung_Cap' (供應商). Vui lòng kiểm tra lại file gốc.")

# --- VIEW 2: QUẢN LÝ KIỂM SOÁT LỆCH MÀU GIỮA CÁC LÔ ---
elif view_mode == "📦 Inter-Batch Color Control":
    st.header(f"Phân tích Biến động Màu sắc giữa các Lô (Batch-to-Batch)")
    
    df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
        'Ngay_SX': 'max', 'ΔE': 'mean', 'dL_N': 'mean', 'da_N': 'mean', 'db_N': 'mean'
    }).sort_values(by='Ngay_SX')

    # Biểu đồ Run Chart cho Delta E
    st.subheader("Biểu đồ Run Chart ΔE qua từng lô sản xuất")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_batch, x='Batch_Lot', y='ΔE', marker='o', color='darkred', linewidth=2)
    ax3.axhline(1.0, color='red', linestyle='-', label='Rejection Limit')
    ax3.axhline(0.7, color='orange', linestyle='--', label='Warning Limit')
    plt.xticks(rotation=45)
    ax3.legend()
    st.pyplot(fig3)

    st.markdown("---")
    
    # Phân tích hướng lệch màu
    st.subheader("Ma trận Phân tích Hướng lệch (Color Direction)")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.write("**Lệch Đỏ/Xanh (Δa) & Vàng/Lục (Δb)**")
        fig4, ax4 = plt.subplots(figsize=(6,6))
        sns.scatterplot(data=df_batch, x='da_N', y='db_N', size='ΔE', hue='ΔE', palette='coolwarm')
        ax4.axhline(0, color='black', lw=1); ax4.axvline(0, color='black', lw=1)
        ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
        st.pyplot(fig4)
        st.info("🎯 Mục tiêu là các chấm phải tập trung tại tâm (0,0).")

    with col_b:
        st.write("**Biến động độ sáng (ΔL)**")
        # Dùng biểu đồ Area để thấy sự trồi sụt của độ sáng
        fig5, ax5 = plt.subplots(figsize=(6,6))
        plt.fill_between(df_batch['Batch_Lot'].astype(str), df_batch['dL_N'], color="gray", alpha=0.3)
        plt.plot(df_batch['Batch_Lot'].astype(str), df_batch['dL_N'], color="black", marker='s')
        ax5.axhline(0, color='red', linestyle='--')
        plt.xticks(rotation=90)
        st.pyplot(fig5)

# --- VIEW 3: DỮ LIỆU THÔ ---
elif view_mode == "📋 Raw Data View":
    st.subheader("Dữ liệu chi tiết")
    st.dataframe(df_filtered, use_container_width=True)

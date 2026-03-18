import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA Dashboards", layout="wide")
st.title("📊 Hệ thống Phân tích Chất lượng Thép Mạ Màu")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # Mapping các cột kỹ thuật
        col_mapping = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_BACK_BLANCH': 'G_Back_N', 'SOUTH_BACK_BLANCH': 'G_Back_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'SOUTH_TOP_DELTA_L': 'dL_S', 'SOUTH_TOP_DELTA_A': 'da_S', 'SOUTH_TOP_DELTA_B': 'db_S'
        }
        
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_mapping[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_mapping[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_mapping)
        if 'Gloss_LSL' not in df.columns: df['Gloss_LSL'] = 0
        if 'Gloss_USL' not in df.columns: df['Gloss_USL'] = 0
        
        # Ép kiểu số
        cols_num = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'dL_S', 'da_S', 'db_S', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        
        # Tính toán các chỉ số bổ sung
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['ΔL'] = df[['dL_N', 'dL_S']].mean(axis=1)
        df['Δa'] = df[['da_N', 'da_S']].mean(axis=1)
        df['Δb'] = df[['db_N', 'db_S']].mean(axis=1)
        df['Gloss_Line_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['Gloss_Line_Back'] = df[['G_Back_N', 'G_Back_S']].mean(axis=1)
        
        df['Status'] = '✅ PASS'
        fail_cond = (df['Gloss_Lab'] < df['Gloss_LSL']) | (df['Gloss_Lab'] > df['Gloss_USL']) | (df['ΔE'] > 1.0)
        df.loc[fail_cond, 'Status'] = '❌ FAIL'
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- 3. BỘ LỌC SIDEBAR ---
st.sidebar.header("🔍 Bộ lọc")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn:", list_ma_son)
df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. TỔNG HỢP THEO BATCH ---
df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max', 'Gloss_Lab': 'mean', 'Gloss_Line_Top': 'mean', 'Gloss_Line_Back': 'mean',
    'ΔE': 'mean', 'ΔL': 'mean', 'Δa': 'mean', 'Δb': 'mean', 'Gloss_LSL': 'first', 'Gloss_USL': 'first'
}).sort_values(by='Ngay_SX')

# --- 5. QUẢN LÝ CÁC VIEW (TABS) ---
tab1, tab2, tab3 = st.tabs(["📋 VIEW SUMMARY DATA", "📉 VIEW PHÂN TÍCH GLOSS", "🎨 VIEW PHÂN TÍCH MÀU SẮC"])

# --- VIEW 1: TỔNG HỢP DỮ LIỆU ---
with tab1:
    st.subheader(f"Bảng tổng hợp dữ liệu chi tiết: {ma_son_selected}")
    g_lsl, g_usl = df_filtered['Gloss_LSL'].iloc[0], df_filtered['Gloss_USL'].iloc[0]
    st.info(f"Tiêu chuẩn Gloss: {g_lsl} - {g_usl} | Tiêu chuẩn ΔE: ≤ 1.0")
    
    display_cols = ['Ngay_SX', 'Batch_Lot', 'Status', 'Gloss_Lab', 'Gloss_Line_Top', 'Gloss_Line_Back', 'Gloss_LSL', 'Gloss_USL', 'ΔE', 'ΔL', 'Δa', 'Δb']
    def style_fail(row):
        return ['background-color: #ffebee' if row['Status'] == '❌ FAIL' else '' for _ in row]
    
    st.dataframe(df_filtered[display_cols].style.apply(style_fail, axis=1).format({
        'Gloss_Lab': '{:.1f}', 'Gloss_Line_Top': '{:.2f}', 'Gloss_Line_Back': '{:.2f}',
        'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 'Δa': '{:.3f}', 'Δb': '{:.3f}', 'Gloss_LSL': '{:.1f}', 'Gloss_USL': '{:.1f}'
    }), use_container_width=True)

# --- VIEW 2: PHÂN TÍCH GLOSS ---
with tab2:
    st.subheader("🚀 Phân tích năng lực Quá trình & Thiết bị đo (Gloss)")
    if not df_batch.empty:
        df_batch['Gloss_Gap'] = df_batch['Gloss_Lab'] - df_batch['Gloss_Line_Top']
        avg_lab = df_batch['Gloss_Lab'].mean()
        avg_gap = df_batch['Gloss_Gap'].mean()
        
        # Chỉ số KPI
        m1, m2, m3 = st.columns(3)
        m1.metric("Gloss TB (Lab)", f"{avg_lab:.1f}")
        m2.metric("Sai lệch TB (Lab-Line)", f"{avg_gap:.2f}", delta="Cần hiệu chuẩn" if abs(avg_gap) > 2 else "Ổn định")
        
        std_dev = df_batch['Gloss_Lab'].std()
        if std_dev > 0:
            cpk = min((g_usl - avg_lab)/(3*std_dev), (avg_lab - g_lsl)/(3*std_dev))
            m3.metric("Năng lực Cpk", f"{cpk:.2f}")

        # Biểu đồ
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.write("**Xu hướng & Biên kiểm soát**")
            fig_t, ax1 = plt.subplots(figsize=(10, 5))
            plot_d = df_batch.copy(); plot_d['Batch_Lot'] = plot_d['Batch_Lot'].astype(str)
            sns.lineplot(data=plot_d, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab', linewidth=3, ax=ax1)
            sns.lineplot(data=plot_d, x='Batch_Lot', y='Gloss_Line_Top', marker='s', label='Line', alpha=0.5, ax=ax1)
            ax1.axhline(g_lsl, color='red', linestyle='--'); ax1.axhline(g_usl, color='red', linestyle='--')
            plt.xticks(rotation=45); st.pyplot(fig_t)
        
        with col_c2:
            st.write("**Phân bổ sai lệch (Gap)**")
            fig_g, ax2 = plt.subplots(figsize=(5, 8.5))
            sns.boxplot(y=df_batch['Gloss_Gap'], color='lightblue', ax=ax2)
            sns.swarmplot(y=df_batch['Gloss_Gap'], color='darkblue', ax=ax2)
            ax2.axhline(0, color='black'); st.pyplot(fig_g)

# --- VIEW 3: PHÂN TÍCH MÀU SẮC ---
with tab3:
    st.subheader("🎨 Phân tích Biến động Màu sắc (Delta Analysis)")
    summary_color = df_batch[['Batch_Lot', 'Ngay_SX', 'ΔE', 'ΔL', 'Δa', 'Δb']]
    st.dataframe(summary_color.style.format({'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 'Δa': '{:.3f}', 'Δb': '{:.3f}'}), use_container_width=True)
    st.info("💡 Mẹo: ΔL > 0 là màu sáng hơn, Δa > 0 là màu đỏ hơn, Δb > 0 là màu vàng hơn so với mẫu chuẩn.")

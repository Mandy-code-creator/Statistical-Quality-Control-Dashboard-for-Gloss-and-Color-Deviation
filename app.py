import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Gloss & Color Specialist", layout="wide")
st.title("📊 Hệ thống Đối chiếu Chất lượng Lab & Line")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # Mapping các cột cố định
        col_mapping = {
            '生產日期': 'Ngay_SX', 
            '製造批號': 'Batch_Lot', 
            '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_BACK_BLANCH': 'G_Back_N', 'SOUTH_BACK_BLANCH': 'G_Back_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'SOUTH_TOP_DELTA_L': 'dL_S', 'SOUTH_TOP_DELTA_A': 'da_S', 'SOUTH_TOP_DELTA_B': 'db_S'
        }
        
        # TÌM KIẾM THÔNG MINH CHO LSL/USL (Dựa trên từ khóa 下限/上限)
        for col in df.columns:
            if '下限' in col and '光澤' in col:
                col_mapping[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col:
                col_mapping[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_mapping)
        
        # Kiểm tra nếu vẫn thiếu cột (do file gốc không có) thì tạo cột 0 để tránh crash
        if 'Gloss_LSL' not in df.columns: df['Gloss_LSL'] = 0
        if 'Gloss_USL' not in df.columns: df['Gloss_USL'] = 0
        
        # Chuyển đổi kiểu số
        cols_num = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 
                    'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'dL_S', 'da_S', 'db_S',
                    'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Định dạng ngày (bỏ giờ)
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        
        # TÍNH TOÁN TRUNG BÌNH
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['ΔL'] = df[['dL_N', 'dL_S']].mean(axis=1)
        df['Δa'] = df[['da_N', 'da_S']].mean(axis=1)
        df['Δb'] = df[['db_N', 'db_S']].mean(axis=1)
        df['Gloss_Line_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['Gloss_Line_Back'] = df[['G_Back_N', 'G_Back_S']].mean(axis=1)
        
        # KIỂM TRA TRẠNG THÁI (PASS/FAIL)
        df['Status'] = '✅ PASS'
        fail_cond = (df['Gloss_Lab'] < df['Gloss_LSL']) | (df['Gloss_Lab'] > df['Gloss_USL']) | (df['ΔE'] > 1.0)
        df.loc[fail_cond, 'Status'] = '❌ FAIL'
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi cấu trúc: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty: st.stop()

# --- 3. BỘ LỌC SIDEBAR ---
st.sidebar.header("🔍 Lọc dữ liệu")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn:", list_ma_son)
df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. TỔNG HỢP BATCH ---
df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max',
    'Gloss_Lab': 'mean',
    'Gloss_Line_Top': 'mean',
    'Gloss_Line_Back': 'mean',
    'ΔE': 'mean', 'ΔL': 'mean', 'Δa': 'mean', 'Δb': 'mean',
    'Gloss_LSL': 'first', 'Gloss_USL': 'first'
}).sort_values(by='Ngay_SX')

# --- 5. HIỂN THỊ ---
tab1, tab2 = st.tabs(["📋 KIỂM TRA CHI TIẾT", "📈 XU HƯỚNG BATCH"])

with tab1:
    st.subheader(f"So sánh kết quả đo: {ma_son_selected}")
    
    # Lấy thông số LSL/USL của mã sơn hiện tại để hiển thị tiêu đề
    g_lsl = df_filtered['Gloss_LSL'].iloc[0] if not df_filtered.empty else 0
    g_usl = df_filtered['Gloss_USL'].iloc[0] if not df_filtered.empty else 0
    st.markdown(f"**Tiêu chuẩn Gloss:** `{g_lsl}` - `{g_usl}` | **Tiêu chuẩn ΔE:** `≤ 1.0`")

    # Bảng hiển thị
    display_cols = ['Ngay_SX', 'Batch_Lot', 'Status', 'Gloss_Lab', 'Gloss_Line_Top', 'Gloss_Line_Back', 'Gloss_LSL', 'Gloss_USL', 'ΔE', 'ΔL', 'Δa', 'Δb']
    
    def style_fail(row):
        return ['background-color: #ffebee' if row['Status'] == '❌ FAIL' else '' for _ in row]

    st.dataframe(
        df_filtered[display_cols].style.apply(style_fail, axis=1).format({
            'Gloss_Lab': '{:.1f}', 'Gloss_Line_Top': '{:.2f}', 'Gloss_Line_Back': '{:.2f}',
            'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 'Δa': '{:.3f}', 'Δb': '{:.3f}',
            'Gloss_LSL': '{:.1f}', 'Gloss_USL': '{:.1f}'
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("Biểu đồ Độ bóng Lab vs Line")
    if not df_batch.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_data = df_batch.copy()
        plot_data['Batch_Lot'] = plot_data['Batch_Lot'].astype(str)

        sns.lineplot(data=plot_data, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab (光澤)', linewidth=3, color='black')
        sns.lineplot(data=plot_data, x='Batch_Lot', y='Gloss_Line_Top', marker='s', label='Line Top Avg', alpha=0.6)
        sns.lineplot(data=plot_data, x='Batch_Lot', y='Gloss_Line_Back', marker='^', label='Line Back Avg', alpha=0.6)
        
        if pd.notna(g_lsl):
            ax.axhline(g_lsl, color='red', linestyle='--', label='Limit')
            ax.axhline(g_usl, color='red', linestyle='--')
                
        plt.xticks(rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Bảng tổng hợp Delta (Δ) theo Batch")
    st.dataframe(
        df_batch[['Batch_Lot', 'Ngay_SX', 'ΔE', 'ΔL', 'Δa', 'Δb']].style.format({
            'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 'Δa': '{:.3f}', 'Δb': '{:.3f}'
        }), 
        use_container_width=True
    )

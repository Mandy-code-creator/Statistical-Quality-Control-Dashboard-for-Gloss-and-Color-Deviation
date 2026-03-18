import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Line & Lab Analysis", layout="wide")
st.title("📊 Hệ thống Đối chiếu Độ bóng Lab & Line")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên các cột kỹ thuật
        df = df.rename(columns={
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_BACK_BLANCH': 'G_Back_N', 'SOUTH_BACK_BLANCH': 'G_Back_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            '光澤60度反射(下限)': 'LSL', '光澤60度反射(上限)': 'USL'
        })
        
        # Chuyển đổi kiểu số
        cols_num = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'LSL', 'USL']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Định dạng ngày (bỏ giờ)
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        
        # TÍNH TOÁN TRUNG BÌNH CỦA LINE (TOP & BACK)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['Gloss_Line_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['Gloss_Line_Back'] = df[['G_Back_N', 'G_Back_S']].mean(axis=1)
        
        # KIỂM TRA ĐẠT/KHÔNG ĐẠT (Dựa trên kết quả Lab là chính)
        df['Status'] = '✅ PASS'
        fail_cond = (df['Gloss_Lab'] < df['LSL']) | (df['Gloss_Lab'] > df['USL']) | (df['ΔE'] > 1.0)
        df.loc[fail_cond, 'Status'] = '❌ FAIL'
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.stop()

# --- 3. SIDEBAR ---
st.sidebar.header("🔍 Lọc dữ liệu")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn:", list_ma_son)
df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. TỔNG HỢP THEO BATCH ---
df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max',
    'Gloss_Lab': 'mean',
    'Gloss_Line_Top': 'mean',
    'Gloss_Line_Back': 'mean',
    'ΔE': 'mean',
    'LSL': 'first', 'USL': 'first'
}).sort_values(by='Ngay_SX')

# --- 5. HIỂN THỊ ---
tab1, tab2 = st.tabs(["📋 KIỂM TRA CHI TIẾT (LAB vs LINE)", "📈 BIỂU ĐỒ XU HƯỚNG"])

with tab1:
    st.subheader(f"So sánh kết quả đo cho mã: {ma_son_selected}")
    
    # Bảng Input chi tiết
    st.markdown("**Bảng đối chiếu từng cuộn:**")
    display_cols = ['Ngay_SX', 'Batch_Lot', 'Status', 'Gloss_Lab', 'Gloss_Line_Top', 'Gloss_Line_Back', 'LSL', 'USL', 'ΔE']
    
    def style_fail(row):
        return ['background-color: #ffebee' if row['Status'] == '❌ FAIL' else '' for _ in row]

    st.dataframe(
        df_filtered[display_cols].style.apply(style_fail, axis=1).format({
            'Gloss_Lab': '{:.1f}', 'Gloss_Line_Top': '{:.2f}', 'Gloss_Line_Back': '{:.2f}',
            'ΔE': '{:.3f}', 'LSL': '{:.1f}', 'USL': '{:.1f}'
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("Phân tích tương quan Độ bóng")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Vẽ cả 3 đường để so sánh: Lab, Line Top và Line Back
    sns.lineplot(data=df_batch, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab (光澤)', linewidth=3, color='black')
    sns.lineplot(data=df_batch, x='Batch_Lot', y='Gloss_Line_Top', marker='s', label='Line Top Avg', alpha=0.7)
    sns.lineplot(data=df_batch, x='Batch_Lot', y='Gloss_Line_Back', marker='^', label='Line Back Avg', alpha=0.7)
    
    # Vẽ biên LSL/USL
    if not df_batch.empty:
        l, u = df_batch['LSL'].iloc[0], df_batch['USL'].iloc[0]
        if pd.notna(l):
            ax.axhline(l, color='red', linestyle='--', label='Limit')
            ax.axhline(u, color='red', linestyle='--')
            
    plt.xticks(rotation=45)
    ax.set_title("So sánh Độ bóng Lab vs Line qua các Batch")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("**Bảng tổng hợp trung bình Batch:**")
    st.dataframe(df_batch, use_container_width=True)

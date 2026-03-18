import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="QA Lab Input Checker", layout="wide")
st.title("📊 Hệ thống Kiểm soát Kết quả Lab & Chất lượng Sơn")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=10)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên các cột (Lưu ý: 光澤 là cột Input của Lab)
        df = df.rename(columns={
            '生產日期': 'Ngay_SX', 
            '製造批號': 'Batch_Lot', 
            '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab_Input', 
            'NORTH_TOP_DELTA_E': 'dE_N', 'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'SOUTH_TOP_DELTA_E': 'dE_S', 'SOUTH_TOP_DELTA_L': 'dL_S', 'SOUTH_TOP_DELTA_A': 'da_S', 'SOUTH_TOP_DELTA_B': 'db_S',
            '光澤60度反射(下限)': 'LSL', '光澤60度反射(上限)': 'USL'
        })
        
        # Chuyển đổi kiểu số cho các cột đo lường
        cols_numeric = ['Gloss_Lab_Input', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'dL_S', 'da_S', 'db_S', 'LSL', 'USL']
        for col in cols_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Định dạng ngày (bỏ phần giờ 00:00:00)
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        
        # TÍNH TOÁN CÁC CHỈ SỐ DELTA TRUNG BÌNH (N+S)/2
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['ΔL'] = df[['dL_N', 'dL_S']].mean(axis=1)
        df['Δa'] = df[['da_N', 'da_S']].mean(axis=1)
        df['Δb'] = df[['db_N', 'db_S']].mean(axis=1)
        
        # KIỂM TRA ĐẠT/KHÔNG ĐẠT (Dựa trên Gloss Input của Lab và ΔE)
        df['Check_Result'] = '✅ PASS'
        # Điều kiện Fail: Gloss nằm ngoài LSL-USL HOẶC ΔE > 1.0
        fail_condition = (df['Gloss_Lab_Input'] < df['LSL']) | (df['Gloss_Lab_Input'] > df['USL']) | (df['ΔE'] > 1.0)
        df.loc[fail_condition, 'Check_Result'] = '❌ FAIL'
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.stop()

# --- 3. BỘ LỌC SIDEBAR ---
st.sidebar.header("🔍 Tra cứu Mã Sơn")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn (塗料編號):", list_ma_son)

df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. HÀM TÔ MÀU DÒNG LỖI ---
def highlight_failed(row):
    return ['background-color: #ffebee' if row['Check_Result'] == '❌ FAIL' else '' for _ in row]

# --- 5. HIỂN THỊ ---
tab1, tab2 = st.tabs(["📋 KIỂM TRA LAB INPUT", "📉 TỔNG HỢP & XU HƯỚNG BATCH"])

with tab1:
    st.subheader(f"Xác nhận dữ liệu đầu vào cho mã: {ma_son_selected}")
    
    # Hiển thị tiêu chuẩn kỹ thuật hiện tại
    lsl_cur = df_filtered['LSL'].iloc[0] if not df_filtered.empty else 0
    usl_cur = df_filtered['USL'].iloc[0] if not df_filtered.empty else 0
    st.info(f"Tiêu chuẩn kỹ thuật: Gloss ({lsl_cur} - {usl_cur}) | ΔE (≤ 1.0)")

    # Bảng hiển thị dữ liệu Input
    # Cột Gloss_Lab_Input chính là giá trị từ cột "光澤"
    cols_to_show = ['Ngay_SX', 'Batch_Lot', 'Check_Result', 'Gloss_Lab_Input', 'LSL', 'USL', 'ΔE', 'ΔL', 'Δa', 'Δb']
    
    st.dataframe(
        df_filtered[cols_to_show].style.apply(highlight_failed, axis=1).format({
            'Gloss_Lab_Input': '{:.1f}', 'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 
            'Δa': '{:.3f}', 'Δb': '{:.3f}', 'LSL': '{:.1f}', 'USL': '{:.1f}'
        }),
        use_container_width=True
    )

with tab2:
    # Gộp Batch: Tính trung bình các giá trị Input của Lab theo số Batch
    df_batch = df_filtered.groupby('Batch_Lot', as_index=False).agg({
        'Ngay_SX': 'max',
        'Gloss_Lab_Input': 'mean',
        'ΔE': 'mean', 'ΔL': 'mean', 'Δa': 'mean', 'Δb': 'mean',
        'LSL': 'first', 'USL': 'first'
    }).sort_values(by='Ngay_SX')

    st.subheader("Dữ liệu trung bình theo từng Lô (Batch Summary)")
    st.dataframe(
        df_batch.style.format({
            'Gloss_Lab_Input': '{:.2f}', 'ΔE': '{:.3f}', 'ΔL': '{:.3f}', 'Δa': '{:.3f}', 'Δb': '{:.3f}'
        }),
        use_container_width=True
    )

    # Biểu đồ xu hướng Gloss của Lab
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_batch, x='Batch_Lot', y='Gloss_Lab_Input', marker='s', color='tab:blue', label='Gloss Lab Avg')
    if pd.notna(lsl_cur):
        ax.axhline(lsl_cur, color='red', linestyle='--', label='LSL')
        ax.axhline(usl_cur, color='red', linestyle='--', label='USL')
    
    plt.xticks(rotation=45)
    ax.set_title(f"Biến động Độ bóng Lab (光澤) theo Batch - {ma_son_selected}")
    ax.legend()
    st.pyplot(fig)

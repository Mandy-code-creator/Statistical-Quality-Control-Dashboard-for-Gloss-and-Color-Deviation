import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel Quality Dashboard", layout="wide")
st.title("📊 Hệ thống Phân tích Chất lượng Sơn (Theo Mã Sơn & Batch)")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_data():
    # Link Google Sheets của bạn
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Đổi tên các cột kỹ thuật để code chạy mượt
        df = df.rename(columns={
            '生產日期': 'Ngay_SX',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_BLANCH': 'Gloss_N', 
            'SOUTH_TOP_BLANCH': 'Gloss_S',
            'NORTH_TOP_DELTA_E': 'dE_N',
            'SOUTH_TOP_DELTA_E': 'dE_S',
            '光澤60度反射(下限)': 'LSL',
            '光澤60度反射(上限)': 'USL'
        })
        
        # Ép kiểu dữ liệu số để tránh lỗi tính toán
        cols_to_fix = ['Gloss_N', 'Gloss_S', 'dE_N', 'dE_S', 'LSL', 'USL']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Chuyển đổi cột Ngày về dạng chuẩn Datetime
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce')
        
        # BƯỚC 1: Tính trung bình mỗi cuộn (N + S) / 2
        df['Gloss_Coil_Avg'] = df[['Gloss_N', 'Gloss_S']].mean(axis=1)
        df['dE_Coil_Avg'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối Google Sheets: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.warning("Dữ liệu đang trống hoặc link Sheets có vấn đề.")
    st.stop()

# --- 3. BỘ LỌC SIDEBAR (CHỈ THEO MÃ SƠN) ---
st.sidebar.header("🔍 Cấu hình lọc")
list_ma_son = sorted(df_raw['Ma_Son'].dropna().unique().tolist())
ma_son_selected = st.sidebar.selectbox("🎯 Chọn Mã Sơn (塗料編號):", list_ma_son)

# Lọc dữ liệu thô theo mã sơn đã chọn
df_filtered = df_raw[df_raw['Ma_Son'] == ma_son_selected].copy()

# --- 4. XỬ LÝ GỘP BATCH (LOGIC MỚI) ---
# Tại đây, chúng ta CHỈ gộp theo Batch_Lot. 
# Nếu Batch giống nhau nhưng khác Ngày, chúng sẽ vẫn được gom lại làm một.
df_batch_summary = df_filtered.groupby('Batch_Lot', as_index=False).agg({
    'Ngay_SX': 'max',            # Hiển thị ngày gần nhất của lô đó
    'Ma_Son': 'first',
    'Gloss_Coil_Avg': 'mean',    # Tính trung bình lần 2 cho toàn bộ lô
    'dE_Coil_Avg': 'mean',       # Tính trung bình lần 2 cho dE
    'LSL': 'first',
    'USL': 'first'
}).rename(columns={'Gloss_Coil_Avg': 'Gloss_Batch_Avg', 'dE_Coil_Avg': 'dE_Batch_Avg'})

# Sắp xếp lại theo thời gian để vẽ biểu đồ cho đúng
df_batch_summary = df_batch_summary.sort_values(by='Ngay_SX')

# --- 5. HIỂN THỊ KẾT QUẢ ---
tab1, tab2 = st.tabs(["📋 Dữ liệu Tổng hợp", "📉 Biểu đồ Xu hướng"])

with tab1:
    st.subheader(f"Kết quả phân tích mã: {ma_son_selected}")
    
    col1, col2 = st.columns(2)
    col1.metric("Tổng số cuộn (Coils)", len(df_filtered))
    col2.metric("Số lượng Lô (Batches)", len(df_batch_summary))
    
    st.markdown("---")
    st.markdown("**Bảng tổng hợp theo từng Batch (Đã gộp trùng):**")
    # Hiển thị bảng và làm đẹp định dạng số
    st.dataframe(
        df_batch_summary.style.format({
            'Gloss_Batch_Avg': '{:.2f}',
            'dE_Batch_Avg': '{:.3f}',
            'LSL': '{:.1f}',
            'USL': '{:.1f}'
        }), 
        use_container_width=True
    )

with tab2:
    if not df_batch_summary.empty:
        st.subheader(f"Biểu đồ X-Bar cho mã {ma_son_selected}")
        
        # Khởi tạo khung vẽ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Biểu đồ Delta E
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='dE_Batch_Avg', marker='o', ax=ax1, color='tab:red', linewidth=2)
        ax1.axhline(1.0, color='black', linestyle='--', label='Ngưỡng dE = 1.0')
        ax1.set_title("Biến động dE trung bình theo Batch", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Delta E")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Biểu đồ Gloss
        sns.lineplot(data=df_batch_summary, x='Batch_Lot', y='Gloss_Batch_Avg', marker='s', ax=ax2, color='tab:blue', linewidth=2)
        
        # Vẽ đường giới hạn LSL/USL nếu có dữ liệu
        lsl_val = df_batch_summary['LSL'].iloc[0]
        usl_val = df_batch_summary['USL'].iloc[0]
        if pd.notna(lsl_val):
            ax2.axhline(lsl_val, color='orange', linestyle='-', label=f'LSL ({lsl_val})')
            ax2.axhline(usl_val, color='orange', linestyle='-', label=f'USL ({usl_val})')
            
        ax2.set_title("Biến động Độ bóng trung bình theo Batch", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Gloss (60°)")
        ax2.set_xlabel("Mã số Batch (Batch Lot)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Xoay chữ trục X cho dễ đọc
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Thiết lập cấu hình trang
st.set_page_config(page_title="SQC Dashboard", layout="wide")
st.title("Statistical Quality Control Dashboard for Gloss and Color Deviation")
st.markdown("---")

# 2. Các từ điển giải mã (Mapping Dictionaries)
supplier_map = {
    'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao',
    'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar',
    'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun',
    'F': 'KCC (New)', 'D': 'DNT', 'P': 'KCC (Posco)' 
}

resin_map = {
    '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC',
    '5': 'PVDF', '6': 'SMP', '7': 'AC', '8': 'WB',
    '9': 'IP', 'A': 'PVB', 'B': 'PVF', 'G': 'PET'
}

color_map = {
    '0': 'Clear', '1': 'Red', 'R': 'Red', 'O': 'Orange', '2': 'Orange',
    '3': 'Yellow', 'Y': 'Yellow', '4': 'Green', 'G': 'Green',
    '5': 'Blue', 'L': 'Blue', 'V': 'Violet', '6': 'Violet',
    'N': 'Brown', '7': 'Brown', 'T': 'White', 'H': 'White', 'W': 'White', '8': 'White',
    'A': 'Gray', 'C': 'Gray', '9': 'Gray', 'B': 'Black', 'S': 'Silver', 'M': 'Metallic',
    'D': 'Dark'
}

# 3. Hàm tải dữ liệu trực tiếp từ Google Sheet
@st.cache_data(ttl=600) # Cập nhật lại dữ liệu mỗi 10 phút (600s)
def load_data():
    # Chuyển đổi link Google Sheet sang dạng xuất CSV
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    
    # Đọc dữ liệu
    df = pd.read_csv(sheet_url)
    
    # Đổi tên các cột tiếng Trung sang tiếng Việt/Anh để dễ lập trình 
    df = df.rename(columns={
        '生產日期': 'Ngay_San_Xuat',
        '塗料編號': 'Ma_Son',
        '光澤': 'Do_Bong',
        'NORTH_TOP_DELTA_E': 'Delta_E_North',
        'SOUTH_TOP_DELTA_E': 'Delta_E_South'
    })
    
    # Làm sạch dữ liệu: Bỏ các dòng thiếu Mã sơn hoặc Độ bóng
    df = df.dropna(subset=['Ma_Son', 'Do_Bong'])
    
    # Ép kiểu dữ liệu (chuyển chữ thành số)
    df['Do_Bong'] = pd.to_numeric(df['Do_Bong'], errors='coerce')
    df['Delta_E_North'] = pd.to_numeric(df['Delta_E_North'], errors='coerce')
    df['Delta_E_South'] = pd.to_numeric(df['Delta_E_South'], errors='coerce')
    
    # Bóc tách cấu trúc Mã sơn
    df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map).fillna('Khác')
    df['Loai_Nhua'] = df['Ma_Son'].str[2].map(resin_map).fillna('Khác')
    df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map).fillna('Khác')
    
    # Lấy giá trị Delta E trung bình giữa hai mép cuộn (North và South)
    df['Delta_E_Trung_Binh'] = df[['Delta_E_North', 'Delta_E_South']].mean(axis=1)
    
    return df

# Tải dữ liệu
df = load_data()

# 4. Khu vực Thanh công cụ bộ lọc (Sidebar)
st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")

# Chọn Màu sắc
danh_sach_mau = sorted(df['Mau_Sac'].astype(str).unique().tolist())
mau_chon = st.sidebar.selectbox("Chọn Màu Sắc (Color):", ["Tất cả"] + danh_sach_mau)

# Chọn Loại Nhựa
danh_sach_nhua = sorted(df['Loai_Nhua'].astype(str).unique().tolist())
nhua_chon = st.sidebar.selectbox("Chọn Hệ Nhựa (Resin):", ["Tất cả"] + danh_sach_nhua)

# Áp dụng bộ lọc
df_filtered = df.copy()
if mau_chon != "Tất cả":
    df_filtered = df_filtered[df_filtered['Mau_Sac'] == mau_chon]
if nhua_chon != "Tất cả":
    df_filtered = df_filtered[df_filtered['Loai_Nhua'] == nhua_chon]

# 5. Khu vực hiển thị Dashboard
st.subheader(f"📊 Thống kê cho Màu: {mau_chon} | Nhựa: {nhua_chon}")

if not df_filtered.empty:
    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Tổng số lượng cuộn (Coils)", value=len(df_filtered))
    col2.metric(label="Độ bóng trung bình (Gloss)", value=f"{df_filtered['Do_Bong'].mean():.1f}")
    col3.metric(label="Độ lệch chuẩn Gloss (SD)", value=f"{df_filtered['Do_Bong'].std():.2f}")
    col4.metric(label="Độ lệch màu Delta E (TB)", value=f"{df_filtered['Delta_E_Trung_Binh'].mean():.2f}")

    st.markdown("---")

    # --- Charts ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Biểu đồ 1: Phân phối Gloss
    sns.boxplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_filtered, ax=ax1, palette="Set2")
    sns.stripplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_filtered, color='black', alpha=0.4, jitter=True, ax=ax1)
    ax1.set_title("Biến động Độ Bóng (Gloss Variation)", fontweight='bold')
    ax1.set_xlabel("Nhà Cung Cấp (Supplier)")
    ax1.set_ylabel("Gloss")
    ax1.tick_params(axis='x', rotation=45)

    # Biểu đồ 2: Phân phối Color Deviation (Delta E)
    sns.boxplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_filtered, ax=ax2, palette="Set1")
    sns.stripplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_filtered, color='black', alpha=0.4, jitter=True, ax=ax2)
    ax2.set_title("Biến động Độ Lệch Màu (Delta E Variation)", fontweight='bold')
    ax2.set_xlabel("Nhà Cung Cấp (Supplier)")
    ax2.set_ylabel("Delta E (dE)")
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Target dE < 1.0') # Giả lập Target dE thông dụng là 1.0
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    # --- Bảng Dữ Liệu Chi Tiết ---
    st.markdown("### Dữ liệu thô chi tiết (Raw Data Extract)")
    # Chỉ hiển thị các cột quan trọng đã xử lý
    hien_thi_cols = ['Ngay_San_Xuat', 'Ma_Son', 'Nha_Cung_Cap', 'Loai_Nhua', 'Mau_Sac', 'Do_Bong', 'Delta_E_North', 'Delta_E_South']
    st.dataframe(df_filtered[hien_thi_cols].reset_index(drop=True), use_container_width=True)

else:
    st.warning("Không tìm thấy dữ liệu nào phù hợp với bộ lọc hiện tại. Vui lòng chọn màu/nhựa khác.")

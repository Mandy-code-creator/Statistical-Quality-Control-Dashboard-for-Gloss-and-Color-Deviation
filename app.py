import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập cấu hình trang
st.set_page_config(page_title="SQC Dashboard", layout="wide")

# Tiêu đề dự án
st.title("Statistical Quality Control Dashboard for Gloss and Color Deviation")
st.markdown("---")

# 1. Các từ điển ánh xạ (Mapping Dictionaries)
supplier_map = {
    'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao',
    'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar',
    'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
}

resin_map = {
    '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC',
    '5': 'PVDF', '6': 'SMP', '7': 'AC', '8': 'WB',
    '9': 'IP', 'A': 'PVB', 'B': 'PVF'
}

color_map = {
    '0': 'Clear', '1': 'Red', 'R': 'Red', 'O': 'Orange', '2': 'Orange',
    '3': 'Yellow', 'Y': 'Yellow', '4': 'Green', 'G': 'Green',
    '5': 'Blue', 'L': 'Blue', 'V': 'Violet', '6': 'Violet',
    'N': 'Brown', '7': 'Brown', 'T': 'White', 'H': 'White', 'W': 'White', '8': 'White',
    'A': 'Gray', 'C': 'Gray', '9': 'Gray', 'B': 'Black', 'S': 'Silver', 'M': 'Metallic'
}

# 2. Hàm tải và xử lý dữ liệu (Tạm thời dùng dữ liệu giả lập, sẽ thay bằng Google Sheets sau)
@st.cache_data
def load_data():
    # Dữ liệu mẫu
    data = {
        'Ngay_San_Xuat': ['2026-03-15', '2026-03-16', '2026-03-17', '2026-03-18', '2026-03-18'],
        'Ma_Son': ['PJ6CD3WZS', 'PT2CD18ZS', 'PB5XY2TZS', 'PL5XY3WZS', 'PT2CD1RZS'],
        'Do_Bong': [85.2, 86.5, 84.1, 85.8, 35.5]
    }
    df = pd.DataFrame(data)
    
    # Bóc tách dữ liệu
    df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map)
    df['Loai_Nhua'] = df['Ma_Son'].str[2].map(resin_map)
    df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map)
    return df

df = load_data()

# 3. Khu vực Sidebar (Thanh công cụ bên trái)
st.sidebar.header("Bộ lọc Dữ liệu (Filters)")

# Tạo danh sách các màu và nhựa có trong tập dữ liệu để lọc
danh_sach_mau = df['Mau_Sac'].dropna().unique().tolist()
mau_chon = st.sidebar.selectbox("Chọn Màu Sắc để phân tích Gloss:", ["Tất cả"] + danh_sach_mau)

# Lọc dữ liệu theo lựa chọn
if mau_chon != "Tất cả":
    df_filtered = df[df['Mau_Sac'] == mau_chon]
else:
    df_filtered = df

# 4. Khu vực Nội dung chính (Main Content)
st.subheader(f"Phân tích Gloss - Màu: {mau_chon}")

# Hiển thị các chỉ số tổng quan (KPI)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Tổng số cuộn", value=len(df_filtered))
with col2:
    gloss_tb = df_filtered['Do_Bong'].mean()
    st.metric(label="Độ bóng Trung bình", value=f"{gloss_tb:.1f}" if pd.notna(gloss_tb) else "N/A")
with col3:
    gloss_std = df_filtered['Do_Bong'].std()
    st.metric(label="Độ lệch chuẩn (SD)", value=f"{gloss_std:.2f}" if pd.notna(gloss_std) else "0.00")

# Vẽ biểu đồ Boxplot so sánh các nhà cung cấp
st.markdown("### Phân phối Độ bóng theo Nhà cung cấp")
if not df_filtered.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_filtered, ax=ax, palette="Set2")
    sns.stripplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_filtered, color='black', alpha=0.5, jitter=True, ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Nhà cung cấp")
    plt.ylabel("Độ bóng (Gloss)")
    st.pyplot(fig)
else:
    st.warning("Không có dữ liệu cho bộ lọc này.")

# Hiển thị bảng dữ liệu chi tiết
st.markdown("### Bảng dữ liệu chi tiết")
st.dataframe(df_filtered)

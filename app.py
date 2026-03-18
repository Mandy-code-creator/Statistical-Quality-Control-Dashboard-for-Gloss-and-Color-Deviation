import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP CẤU HÌNH TRANG ---
st.set_page_config(page_title="SQC Dashboard - Data Check", layout="wide")
st.title("📊 Statistical Quality Control Dashboard")
st.markdown("Hệ thống Phân tích & Kiểm tra Dữ liệu Tôn mạ màu.")
st.markdown("---")

# --- 2. TỪ ĐIỂN GIẢI MÃ ---
supplier_map = {'S':'Yungchi','T':'AKZO NOBEL','B':'Beckers','C':'Nan Pao','U':'Quali Poly','N':'Nippon','K':'Kansai','V':'Valspar','J':'Valspar (SW)','L':'KCC','R':'Noroo','Q':'Paoqun','F':'KCC (New)','D':'DNT','P':'KCC (Posco)'}
resin_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF','G':'PET'}
color_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','3':'Yellow','Y':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic','D':'Dark'}

# --- 3. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=60) # Giảm thời gian cache để cập nhật dữ liệu nhanh hơn khi bạn sửa Sheet
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df_raw = pd.read_csv(sheet_url)
        
        # Đổi tên cột để dễ xử lý code
        df = df_raw.rename(columns={
            '生產日期': 'Ngay_San_Xuat',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_BLANCH': 'Gloss_North', 
            'SOUTH_TOP_BLANCH': 'Gloss_South',
            'NORTH_TOP_DELTA_E': 'dE_North',
            'SOUTH_TOP_DELTA_E': 'dE_South',
            '光澤60度反射(下限)': 'Gloss_LSL',
            '光澤60度反射(上限)': 'Gloss_USL'
        })
        
        # Chuyển đổi kiểu dữ liệu số
        cols_num = ['Gloss_North', 'Gloss_South', 'dE_North', 'dE_South', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Tính toán giá trị trung bình theo yêu cầu của bạn (Double Average Step 1)
        df['dE_Avg_Coil'] = df[['dE_North', 'dE_South']].mean(axis=1)
        df['Gloss_Avg_Coil'] = df[['Gloss_North', 'Gloss_South']].mean(axis=1)
        
        # Giải mã thông tin từ Mã Sơn
        df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map).fillna('Khác')
        df['Loai_Nhua'] = df['Ma_Son'].str[2].map(resin_map).fillna('Khác')
        df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map).fillna('Khác')
        
        return df, df_raw
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, df_raw = load_data()

if df.empty:
    st.warning("Không thể tải dữ liệu. Vui lòng kiểm tra lại link Google Sheets.")
else:
    # --- 4. CHIA TABS ---
    tab0, tab1, tab2 = st.tabs(["📋 DỮ LIỆU TỔNG HỢP", "🏢 SO SÁNH HÃNG", "📉 KIỂM SOÁT LÔ (LOT)"])

    # ==========================================
    # TAB 0: XUẤT DATA TỔNG HỢP (KIỂM TRA LỖI)
    # ==========================================
    with tab0:
        st.subheader("Bảng dữ liệu chi tiết (Raw Data & Processed)")
        st.info("Mandy hãy kiểm tra bảng này xem các giá trị North/South và Mã Sơn đã khớp với thực tế chưa.")
        
        # Cho phép người dùng chọn xem toàn bộ hoặc chỉ 1 số cột quan trọng
        view_mode = st.radio("Chế độ xem:", ["Dữ liệu đã xử lý (Gọn)", "Dữ liệu gốc từ Google Sheets"], horizontal=True)
        
        if view_mode == "Dữ liệu đã xử lý (Gọn)":
            st.dataframe(df[['Ngay_San_Xuat', 'Batch_Lot', 'Ma_Son', 'Gloss_North', 'Gloss_South', 'Gloss_Avg_Coil', 'dE_Avg_Coil', 'Nha_Cung_Cap']], use_container_width=True)
        else:
            st.dataframe(df_raw, use_container_width=True)

    # ==========================================
    # TAB 1 & 2 (Giữ nguyên logic lọc thông minh của bạn)
    # ==========================================
    # (Phần Sidebar lọc dữ liệu)
    st.sidebar.header("🔍 Bộ lọc")
    mau_list = sorted(df['Mau_Sac'].unique().tolist())
    mau_chon = st.sidebar.selectbox("Màu sắc:", ["Tất cả"] + mau_list)
    
    df_filter = df.copy()
    if mau_chon != "Tất cả":
        df_filter = df_filter[df_filter['Mau_Sac'] == mau_chon]
        
    with tab1:
        st.write(f"Đang phân tích {len(df_filter)} bản ghi.")
        # [Code vẽ Boxplot tương tự như trước...]
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.boxplot(x='Nha_Cung_Cap', y='Gloss_Avg_Coil', data=df_filter, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        # BƯỚC TRUNG BÌNH 2: TRUNG BÌNH THEO BATCH
        df_batch = df_filter.groupby(['Ngay_San_Xuat', 'Batch_Lot'], as_index=False).agg({
            'Gloss_Avg_Coil': 'mean',
            'dE_Avg_Coil': 'mean',
            'Gloss_LSL': 'first',
            'Gloss_USL': 'first'
        })
        st.markdown("#### Biểu đồ Xu hướng theo Batch (Sau khi lấy trung bình lần 2)")
        # [Code vẽ Line chart tương tự như trước...]
        st.dataframe(df_batch)

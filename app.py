import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP CẤU HÌNH TRANG ---
st.set_page_config(page_title="SQC Dashboard", layout="wide")
st.title("📊 Statistical Quality Control Dashboard")
st.markdown("Hệ thống Phân tích Độ bóng (Gloss) và Độ lệch màu (Delta E) Tôn mạ màu.")
st.markdown("---")

# --- 2. TỪ ĐIỂN GIẢI MÃ (MAPPING DICTIONARIES) ---
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

# --- 3. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=600)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # Đổi tên cột
        df = df.rename(columns={
            '生產日期': 'Ngay_San_Xuat',
            '塗料編號': 'Ma_Son',
            '光澤': 'Do_Bong',
            'NORTH_TOP_DELTA_E': 'Delta_E_North',
            'SOUTH_TOP_DELTA_E': 'Delta_E_South'
        })
        
        # Làm sạch cơ bản
        df = df.dropna(subset=['Ma_Son', 'Do_Bong', 'Ngay_San_Xuat'])
        df['Do_Bong'] = pd.to_numeric(df['Do_Bong'], errors='coerce')
        df['Delta_E_North'] = pd.to_numeric(df['Delta_E_North'], errors='coerce')
        df['Delta_E_South'] = pd.to_numeric(df['Delta_E_South'], errors='coerce')
        
        # Bóc tách mã sơn
        df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map).fillna('Khác')
        df['Loai_Nhua'] = df['Ma_Son'].str[2].map(resin_map).fillna('Khác')
        df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map).fillna('Khác')
        
        # Tính toán
        df['Delta_E_Trung_Binh'] = df[['Delta_E_North', 'Delta_E_South']].mean(axis=1)
        df = df.dropna(subset=['Do_Bong'])
        
        # Sắp xếp theo ngày sản xuất để vẽ biểu đồ Trend chuẩn xác
        df['Ngay_San_Xuat'] = pd.to_datetime(df['Ngay_San_Xuat'], errors='coerce')
        df = df.sort_values(by='Ngay_San_Xuat')
        df['Ngay_San_Xuat_Str'] = df['Ngay_San_Xuat'].dt.strftime('%Y-%m-%d') # Format lại chuỗi để hiển thị
        
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Dữ liệu trống hoặc không kết nối được Google Sheet.")
else:
    # --- 4. SIDEBAR - BỘ LỌC CHUNG ---
    st.sidebar.header("🔍 Cài đặt Phân tích")
    
    # Lọc Màu sắc và Nhựa áp dụng cho TOÀN BỘ Dashboard
    danh_sach_mau = sorted([str(m) for m in df['Mau_Sac'].unique() if m != 'Khác'])
    mau_chon = st.sidebar.selectbox("🎨 Chọn Màu Sắc (Color):", ["Tất cả"] + danh_sach_mau)
    
    danh_sach_nhua = sorted([str(n) for n in df['Loai_Nhua'].unique() if n != 'Khác'])
    nhua_chon = st.sidebar.selectbox("🧪 Chọn Hệ Nhựa (Resin):", ["Tất cả"] + danh_sach_nhua)
    
    # Lọc DataFrame chung
    df_main = df.copy()
    if mau_chon != "Tất cả":
        df_main = df_main[df_main['Mau_Sac'] == mau_chon]
    if nhua_chon != "Tất cả":
        df_main = df_main[df_main['Loai_Nhua'] == nhua_chon]

    if df_main.empty:
        st.info("Không có dữ liệu cho hệ màu/nhựa này.")
    else:
        # --- 5. CHIA GIAO DIỆN THÀNH 2 TABS ---
        tab1, tab2 = st.tabs(["🏢 So sánh Nhà Cung Cấp (Vĩ mô)", "📉 Kiểm soát theo Lô Sản Xuất (Vi mô)"])
        
        # ==========================================
        # TAB 1: SO SÁNH NHÀ CUNG CẤP (BENCHMARKING)
        # ==========================================
        with tab1:
            st.subheader(f"So sánh năng lực các hãng sơn - Màu: {mau_chon} | Nhựa: {nhua_chon}")
            
            # KPI Cards
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số cuộn phân tích", len(df_main))
            c2.metric("Số lượng Nhà cung cấp", df_main['Nha_Cung_Cap'].nunique())
            c3.metric("Độ lệch màu (dE) trung bình chung", f"{df_main['Delta_E_Trung_Binh'].mean():.2f}")
            
            st.markdown("---")
            
            # Biểu đồ Boxplot
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.boxplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_main, ax=ax1, hue='Nha_Cung_Cap', palette="Set2", legend=False)
            sns.stripplot(x='Nha_Cung_Cap', y='Do_Bong', data=df_main, color='black', alpha=0.3, jitter=True, ax=ax1)
            ax1.set_title("Biến động Độ Bóng (Gloss Variation)", fontweight='bold')
            ax1.set_xlabel("Nhà Cung Cấp")
            ax1.set_ylabel("Độ bóng (Gloss)")
            ax1.tick_params(axis='x', rotation=45)
            
            sns.boxplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_main, ax=ax2, hue='Nha_Cung_Cap', palette="Set1", legend=False)
            sns.stripplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_main, color='black', alpha=0.3, jitter=True, ax=ax2)
            ax2.set_title("Biến động Độ Lệch Màu (Delta E Variation)", fontweight='bold')
            ax2.set_xlabel("Nhà Cung Cấp")
            ax2.set_ylabel("Delta E (dE)")
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Target dE < 1.0')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            st.pyplot(fig1)
            
            # Bảng tóm tắt thống kê
            st.markdown("#### Bảng thống kê theo Nhà cung cấp")
            bang_thong_ke = df_main.groupby('Nha_Cung_Cap').agg(
                So_Cuon=('Do_Bong', 'count'),
                Gloss_TB=('Do_Bong', 'mean'),
                Gloss_SD=('Do_Bong', 'std'),
                dE_TB=('Delta_E_Trung_Binh', 'mean'),
                dE_Max=('Delta_E_Trung_Binh', 'max')
            ).reset_index()
            st.dataframe(bang_thong_ke.style.format({
                'Gloss_TB': '{:.1f}', 'Gloss_SD': '{:.2f}', 'dE_TB': '{:.2f}', 'dE_Max': '{:.2f}'
            }))

        # ==========================================
        # TAB 2: KIỂM SOÁT THEO LÔ (BATCH/LOT CONTROL)
        # ==========================================
        with tab2:
            st.subheader("Theo dõi xu hướng chất lượng qua từng đợt sản xuất (Trend Analysis)")
            
            # Chọn 1 nhà cung cấp cụ thể để xem chi tiết
            danh_sach_ncc_tab2 = sorted(df_main['Nha_Cung_Cap'].unique().tolist())
            ncc_chon = st.selectbox("🏭 Chọn Hãng Sơn để phân tích xu hướng lô:", danh_sach_ncc_tab2)
            
            df_lot = df_main[df_main['Nha_Cung_Cap'] == ncc_chon]
            
            if not df_lot.empty:
                st.markdown(f"**Đang hiển thị:** Màu {mau_chon} | Nhựa {nhua_chon} | Hãng **{ncc_chon}**")
                
                # Biểu đồ xu hướng (Run Chart / Line Chart)
                fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                
                # Biểu đồ Đường dE (Quan trọng hơn nên vẽ trước)
                sns.lineplot(x='Ngay_San_Xuat_Str', y='Delta_E_Trung_Binh', data=df_lot, ax=ax3, marker='o', color='crimson', errorbar=None)
                sns.scatterplot(x='Ngay_San_Xuat_Str', y='Delta_E_Trung_Binh', data=df_lot, ax=ax3, color='black', alpha=0.5)
                ax3.set_title("Biểu đồ Kiểm soát Độ Lệch Màu (dE Run Chart)", fontweight='bold')
                ax3.set_ylabel("Delta E (dE)")
                ax3.axhline(y=1.0, color='red', linestyle='--', label='Giới hạn chuẩn (UCL) = 1.0')
                ax3.legend()
                ax3.grid(True, linestyle=':', alpha=0.6)
                
                # Biểu đồ Đường Gloss
                sns.lineplot(x='Ngay_San_Xuat_Str', y='Do_Bong', data=df_lot, ax=ax4, marker='s', color='teal', errorbar=None)
                sns.scatterplot(x='Ngay_San_Xuat_Str', y='Do_Bong', data=df_lot, ax=ax4, color='black', alpha=0.5)
                
                # Tính toán Target Gloss (Giá trị trung bình của toàn bộ các lô)
                target_gloss = df_lot['Do_Bong'].mean()
                ax4.axhline(y=target_gloss, color='green', linestyle='-', label=f'Target (Mean) = {target_gloss:.1f}')
                
                ax4.set_title("Biểu đồ Xu hướng Độ Bóng (Gloss Run Chart)", fontweight='bold')
                ax4.set_ylabel("Độ bóng (Gloss)")
                ax4.set_xlabel("Batch / Ngày Sản Xuất")
                ax4.legend()
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, linestyle=':', alpha=0.6)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Dữ liệu chi tiết của Hãng đang chọn
                st.markdown("#### Chi tiết các cuộn thuộc đợt sản xuất")
                cols_to_show = ['Ngay_San_Xuat_Str', 'Ma_Son', 'Do_Bong', 'Delta_E_North', 'Delta_E_South', 'Delta_E_Trung_Binh']
                st.dataframe(df_lot[cols_to_show].reset_index(drop=True))

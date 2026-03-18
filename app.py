import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP CẤU HÌNH TRANG ---
st.set_page_config(page_title="SQC Dashboard", layout="wide")
st.title("📊 Statistical Quality Control Dashboard")
st.markdown("Hệ thống Phân tích Độ bóng (Gloss) và Độ lệch màu (Delta E) Tôn mạ màu.")
st.markdown("---")

# --- 2. TỪ ĐIỂN GIẢI MÃ ---
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
        df = df.rename(columns={
            '生產日期': 'Ngay_San_Xuat',
            '製造批號': 'Batch_Lot',
            '塗料編號': 'Ma_Son',
            'NORTH_TOP_BLANCH': 'Gloss_North_Top', 
            'SOUTH_TOP_BLANCH': 'Gloss_South_Top',
            'NORTH_TOP_DELTA_E': 'Delta_E_North',
            'SOUTH_TOP_DELTA_E': 'Delta_E_South',
            '光澤60度反射(下限)': 'Gloss_LSL',
            '光澤60度反射(上限)': 'Gloss_USL'
        })
        
        df = df.dropna(subset=['Ma_Son', 'Batch_Lot'])
        cols_to_numeric = ['Gloss_North_Top', 'Gloss_South_Top', 'Delta_E_North', 'Delta_E_South', 'Gloss_LSL', 'Gloss_USL']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map).fillna('Khác')
        df['Loai_Nhua'] = df['Ma_Son'].str[2].map(resin_map).fillna('Khác')
        df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map).fillna('Khác')
        
        if 'Delta_E_North' in df.columns and 'Delta_E_South' in df.columns:
            df['Delta_E_Trung_Binh'] = df[['Delta_E_North', 'Delta_E_South']].mean(axis=1)
        
        if 'Gloss_North_Top' in df.columns and 'Gloss_South_Top' in df.columns:
            df['Gloss_Trung_Binh'] = df[['Gloss_North_Top', 'Gloss_South_Top']].mean(axis=1)
            df = df.dropna(subset=['Gloss_Trung_Binh'])
        
        df['Ngay_San_Xuat'] = pd.to_datetime(df['Ngay_San_Xuat'], errors='coerce')
        df = df.sort_values(by=['Ngay_San_Xuat', 'Batch_Lot'])
        df['Batch_Lot'] = df['Batch_Lot'].astype(str)
        df['Ngay_San_Xuat_Str'] = df['Ngay_San_Xuat'].dt.strftime('%Y-%m-%d')
        
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Dữ liệu trống hoặc thiếu các cột chuẩn.")
else:
    # --- 4. SIDEBAR - BỘ LỌC ĐA TẦNG ---
    st.sidebar.header("🔍 Cài đặt Phân tích")
    
    # Tầng 1: Khoanh vùng bằng Nhóm Màu & Nhựa
    danh_sach_mau = sorted([str(m) for m in df['Mau_Sac'].unique() if m != 'Khác'])
    mau_chon = st.sidebar.selectbox("🎨 Phân loại Nhóm Màu:", ["Tất cả"] + danh_sach_mau)
    
    danh_sach_nhua = sorted([str(n) for n in df['Loai_Nhua'].unique() if n != 'Khác'])
    nhua_chon = st.sidebar.selectbox("🧪 Phân loại Hệ Nhựa:", ["Tất cả"] + danh_sach_nhua)
    
    # Lọc tạm thời để thu gọn danh sách Mã Sơn
    df_temp = df.copy()
    if mau_chon != "Tất cả":
        df_temp = df_temp[df_temp['Mau_Sac'] == mau_chon]
    if nhua_chon != "Tất cả":
        df_temp = df_temp[df_temp['Loai_Nhua'] == nhua_chon]
        
    # Tầng 2: Chọn đích danh Mã Sơn (Cốt lõi để soi Lot)
    danh_sach_ma_son = sorted(df_temp['Ma_Son'].unique().tolist())
    ma_son_chon = st.sidebar.selectbox("🎯 Chọn Mã Sơn (塗料編號):", ["Tất cả"] + danh_sach_ma_son)
    
    # Áp dụng bộ lọc chính
    df_main = df_temp.copy()
    if ma_son_chon != "Tất cả":
        df_main = df_main[df_main['Ma_Son'] == ma_son_chon]

    if df_main.empty:
        st.info("Không có dữ liệu cho cấu hình này.")
    else:
        # --- 5. CHIA TABS ---
        tab1, tab2 = st.tabs(["🏢 So sánh Nhà Cung Cấp (Vĩ mô)", "📉 Kiểm soát theo Lô Sản Xuất (Vi mô)"])
        
        # ==========================================
        # TAB 1: SO SÁNH NHÀ CUNG CẤP
        # ==========================================
        with tab1:
            if ma_son_chon != "Tất cả":
                st.info(f"💡 Bạn đang lọc đích danh Mã Sơn **{ma_son_chon}**. Biểu đồ so sánh sẽ chỉ hiển thị hãng sản xuất mã này. Để so sánh nhiều hãng, vui lòng đổi Mã Sơn thành 'Tất cả' ở thanh công cụ bên trái.")
                
            st.subheader(f"So sánh năng lực - Màu: {mau_chon} | Nhựa: {nhua_chon}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số cuộn phân tích", len(df_main))
            c2.metric("Số lượng Nhà cung cấp", df_main['Nha_Cung_Cap'].nunique())
            
            df_main['Dat_Chuan_Gloss'] = (df_main['Gloss_Trung_Binh'] >= df_main['Gloss_LSL']) & (df_main['Gloss_Trung_Binh'] <= df_main['Gloss_USL'])
            ty_le_dat = (df_main['Dat_Chuan_Gloss'].sum() / len(df_main)) * 100
            c3.metric("Tỷ lệ đạt chuẩn Độ bóng màng sơn (%)", f"{ty_le_dat:.1f}%")
            
            st.markdown("---")
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.boxplot(x='Nha_Cung_Cap', y='Gloss_Trung_Binh', data=df_main, ax=ax1, hue='Nha_Cung_Cap', palette="Set2", legend=False)
            sns.stripplot(x='Nha_Cung_Cap', y='Gloss_Trung_Binh', data=df_main, color='black', alpha=0.3, jitter=True, ax=ax1)
            ax1.set_title("Biến động Độ Bóng Màng Sơn (Topcoat Gloss)", fontweight='bold')
            ax1.set_xlabel("Nhà Cung Cấp")
            ax1.set_ylabel("Độ bóng Thành phẩm")
            ax1.tick_params(axis='x', rotation=45)
            
            sns.boxplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_main, ax=ax2, hue='Nha_Cung_Cap', palette="Set1", legend=False)
            sns.stripplot(x='Nha_Cung_Cap', y='Delta_E_Trung_Binh', data=df_main, color='black', alpha=0.3, jitter=True, ax=ax2)
            ax2.set_title("Biến động Độ Lệch Màu (Delta E)", fontweight='bold')
            ax2.set_xlabel("Nhà Cung Cấp")
            ax2.set_ylabel("Delta E (dE)")
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Target dE < 1.0')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)

        # ==========================================
        # TAB 2: KIỂM SOÁT THEO LÔ (BATCH/LOT CONTROL)
        # ==========================================
        with tab2:
            st.subheader("Theo dõi xu hướng chất lượng trung bình theo từng Lot Sơn (X-Bar Chart)")
            
            danh_sach_ncc_tab2 = sorted(df_main['Nha_Cung_Cap'].unique().tolist())
            
            # Logic thông minh: Nếu chỉ có 1 hãng (do đã chọn Mã Sơn cụ thể), tự động chọn hãng đó!
            if len(danh_sach_ncc_tab2) == 1:
                ncc_chon = danh_sach_ncc_tab2[0]
                st.markdown(f"🏭 Hệ thống nhận diện Hãng sản xuất: **{ncc_chon}**")
            else:
                ncc_chon = st.selectbox("🏭 Chọn Hãng Sơn để phân tích các Lô:", danh_sach_ncc_tab2)
                
            df_lot = df_main[df_main['Nha_Cung_Cap'] == ncc_chon]
            
            if not df_lot.empty:
                # TRUNG BÌNH KÉP: Tính trung bình của tất cả cuộn trong cùng 1 Lot
                df_batch_agg = df_lot.groupby(['Ngay_San_Xuat_Str', 'Batch_Lot'], as_index=False).agg(
                    So_Luong_Cuon=('Ma_Son', 'count'),
                    Gloss_Batch_TB=('Gloss_Trung_Binh', 'mean'),
                    dE_Batch_TB=('Delta_E_Trung_Binh', 'mean'),
                    Gloss_LSL=('Gloss_LSL', 'first'),
                    Gloss_USL=('Gloss_USL', 'first')
                )
                
                df_batch_agg['Label_Truc_X'] = df_batch_agg['Ngay_San_Xuat_Str'] + "\n(" + df_batch_agg['Batch_Lot'] + ")"
                
                # Cảnh báo tự động
                loi_gloss = df_batch_agg[(df_batch_agg['Gloss_Batch_TB'] < df_batch_agg['Gloss_LSL']) | (df_batch_agg['Gloss_Batch_TB'] > df_batch_agg['Gloss_USL'])]
                loi_de = df_batch_agg[df_batch_agg['dE_Batch_TB'] > 1.0]
                
                if not loi_gloss.empty or not loi_de.empty:
                    st.error("🚨 **CẢNH BÁO CHẤT LƯỢNG:** Phát hiện Lot sơn vi phạm tiêu chuẩn kiểm soát!")
                    if not loi_gloss.empty:
                        st.write(f"- **Lỗi Độ Bóng (Gloss):** Các Lot {', '.join(loi_gloss['Batch_Lot'].tolist())}")
                    if not loi_de.empty:
                        st.write(f"- **Lỗi Lệch Màu (dE > 1.0):** Các Lot {', '.join(loi_de['Batch_Lot'].tolist())}")
                else:
                    st.success("✅ Tuyệt vời! Tất cả các Lot sơn đang hiển thị đều đạt chuẩn kiểm soát.")
                
                # Vẽ Biểu Đồ
                fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
                
                sns.lineplot(x='Label_Truc_X', y='dE_Batch_TB', data=df_batch_agg, ax=ax3, marker='o', color='crimson', label='dE Trung Bình Lô')
                ax3.set_title("Biểu đồ Kiểm soát dE (Trung bình Lô Sơn)", fontweight='bold')
                ax3.set_ylabel("Delta E (dE)")
                ax3.axhline(y=1.0, color='red', linestyle='--', label='Cảnh báo dE = 1.0')
                ax3.legend(loc='upper right')
                ax3.grid(True, linestyle=':', alpha=0.6)
                
                sns.lineplot(x='Label_Truc_X', y='Gloss_Batch_TB', data=df_batch_agg, ax=ax4, marker='s', color='teal', label='Gloss Trung Bình Lô')
                sns.lineplot(x='Label_Truc_X', y='Gloss_USL', data=df_batch_agg, ax=ax4, color='darkorange', linestyle='--', label='Giới hạn trên (USL)')
                sns.lineplot(x='Label_Truc_X', y='Gloss_LSL', data=df_batch_agg, ax=ax4, color='darkorange', linestyle='--', label='Giới hạn dưới (LSL)')
                
                ax4.set_title("Biểu đồ Kiểm soát Độ Bóng (Trung bình Lô Sơn)", fontweight='bold')
                ax4.set_ylabel("Độ bóng (Gloss)")
                ax4.set_xlabel("Ngày Sản Xuất (Mã Lô)")
                ax4.legend(loc='upper right')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, linestyle=':', alpha=0.6)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Bảng Dữ Liệu
                st.markdown("#### Bảng Dữ Liệu Tổng Hợp Theo Lot Sơn")
                def highlight_batch_errors(row):
                    styles = [''] * len(row)
                    try:
                        if pd.notna(row['Gloss_Batch_TB']) and pd.notna(row['Gloss_LSL']) and pd.notna(row['Gloss_USL']):
                            if row['Gloss_Batch_TB'] < row['Gloss_LSL'] or row['Gloss_Batch_TB'] > row['Gloss_USL']:
                                styles[3] = 'background-color: #ffcccc; color: red; font-weight: bold;'
                        if pd.notna(row['dE_Batch_TB']) and row['dE_Batch_TB'] > 1.0:
                            styles[4] = 'background-color: #ffcccc; color: red; font-weight: bold;'
                    except: pass
                    return styles

                df_display = df_batch_agg[['Ngay_San_Xuat_Str', 'Batch_Lot', 'So_Luong_Cuon', 'Gloss_Batch_TB', 'dE_Batch_TB', 'Gloss_LSL', 'Gloss_USL']]
                st.dataframe(df_display.style.apply(highlight_batch_errors, axis=1).format({
                    'Gloss_Batch_TB': '{:.1f}', 'dE_Batch_TB': '{:.2f}', 'Gloss_LSL': '{:.1f}', 'Gloss_USL': '{:.1f}'
                }), width='stretch')

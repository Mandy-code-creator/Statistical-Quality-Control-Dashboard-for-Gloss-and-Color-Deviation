import streamlit as st
import pandas as pd
import numpy as np

# --- 1. THIẾT LẬP CẤU HÌNH ---
st.set_page_config(page_title="Steel QA - Summary Data Report", layout="wide")

@st.cache_data(ttl=10)
def get_original_summary():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Khôi phục Mapping cột gốc
        col_map = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab', 'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S'
        }
        # Tự động tìm cột LSL/USL Gloss
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'Gloss_USL'
        
        df = df.rename(columns=col_map)
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper()

        # --- ÁP DỤNG LOGIC MÃ SƠN MỚI NHẤT CỦA MANDY ---
        # Ký tự 2: Nhà cung cấp
        v_map = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        # Ký tự 3: Hệ nhựa
        r_map = {
            '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC', '5': 'PVDF', 
            '6': 'SMP', '7': 'AC', '8': 'WB', '9': 'IP', 'A': 'PVB', 'B': 'PVF'
        }
        
        df['Vendor'] = df['Ma_Son_Str'].str[1].map(v_map)
        df['Resin'] = df['Ma_Son_Str'].str[2].map(r_map)
        df['Color_Code'] = df['Ma_Son_Str'].str[-4:] # 4 ký tự cuối

        # Chuyển đổi số liệu
        for c in ['Gloss_Lab', 'Gloss_LSL', 'Gloss_USL', 'dE_N', 'dE_S']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        # Logic Pass/Fail cho Gloss
        df['Is_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
        
        return df.dropna(subset=['Vendor'])
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df_raw = get_original_summary()

# --- 2. HIỂN THỊ BẢNG SUMMARY DATA ---
st.header("📊 Summary Data: Phân tích năng lực Nhà cung cấp")

if not df_raw.empty:
    # Nhóm dữ liệu theo đúng phân tầng Kỹ thuật: Nhựa -> Màu -> Vendor
    summary = df_raw.groupby(['Resin', 'Color_Code', 'Vendor']).agg({
        'Batch_Lot': 'count',
        'Gloss_Lab': ['mean', 'std', 'min', 'max'],
        'Gloss_LSL': 'mean',
        'Gloss_USL': 'mean',
        'Is_Pass': 'mean',
        'ΔE': 'mean'
    }).reset_index()

    # Làm sạch tiêu đề cột cho chuyên nghiệp
    summary.columns = [
        'Hệ Nhựa', 'Mã Màu (4 cuối)', 'Nhà Cung Cấp', 'Số Lô (n)', 
        'Gloss TB', 'Độ lệch (Std)', 'Min thực tế', 'Max thực tế', 
        'Spec LSL', 'Spec USL', '% Đạt Spec', 'ΔE TB'
    ]

    # Định dạng % đạt
    summary['% Đạt Spec'] = (summary['% Đạt Spec'] * 100).round(1).astype(str) + '%'

    # Hiển thị bảng dữ liệu
    st.dataframe(
        summary.style.format({
            'Gloss TB': '{:.1f}',
            'Độ lệch (Std)': '{:.2f}',
            'Min thực tế': '{:.1f}',
            'Max thực tế': '{:.1f}',
            'Spec LSL': '{:.0f}',
            'Spec USL': '{:.0f}',
            'ΔE TB': '{:.2f}'
        }).background_gradient(cmap='RdYlGn', subset=['ΔE TB'], low=1, high=0) # ΔE thấp là xanh
          .background_gradient(cmap='RdYlGn_r', subset=['Độ lệch (Std)']), # Std thấp là xanh
        use_container_width=True,
        hide_index=True
    )

    st.info("💡 **Ghi chú:** Bảng đã tự động nhóm các lô hàng có cùng Hệ nhựa và Mã màu để bạn so sánh năng lực giữa các Nhà cung cấp một cách công bằng nhất.")
else:
    st.warning("Đang chờ dữ liệu từ Google Sheets...")

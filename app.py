import streamlit as st
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Steel QA - Summary Report", layout="wide")

@st.cache_data(ttl=10)
def load_summary_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        # Mapping cột theo file gốc của Mandy
        col_map = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab', 'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S'
        }
        # Tìm cột giới hạn Gloss
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'USL'
        
        df = df.rename(columns=col_map)
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper()

        # --- GIẢI MÃ THEO QUY TẮC MỚI NHẤT ---
        v_map = {
            'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao', 
            'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar', 
            'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
        }
        r_map = {
            '1': 'PU', '2': 'PE', '3': 'EPOXY', '4': 'PVC', '5': 'PVDF', 
            '6': 'SMP', '7': 'AC', '8': 'WB', '9': 'IP', 'A': 'PVB', 'B': 'PVF'
        }
        
        df['Vendor'] = df['Ma_Son_Str'].str[1].map(v_map)
        df['Resin'] = df['Ma_Son_Str'].str[2].map(r_map)
        df['Color_4'] = df['Ma_Son_Str'].str[-4:] # Mã màu 4 số cuối

        # Chuyển đổi kiểu số
        cols_num = ['Gloss_Lab', 'LSL', 'USL', 'dE_N', 'dE_S']
        for col in cols_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        # Tính Pass/Fail dựa trên từng dòng dữ liệu thực tế
        df['Is_Pass'] = (df['Gloss_Lab'] >= df['LSL']) & (df['Gloss_Lab'] <= df['USL'])
        
        return df.dropna(subset=['Vendor', 'Resin'])
    except Exception as e:
        st.error(f"⚠️ Lỗi dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_summary_data()

# --- 2. HIỂN THỊ BẢNG SUMMARY DATA ---
st.title("📊 Summary Data: Kiểm soát chất lượng Nhà cung cấp")

if not df_raw.empty:
    # Nhóm dữ liệu để tính toán Summary
    summary = df_raw.groupby(['Resin', 'Color_4', 'Vendor']).agg({
        'Batch_Lot': 'count',
        'Gloss_Lab': ['mean', 'std', 'min', 'max'],
        'LSL': 'mean',
        'USL': 'mean',
        'Is_Pass': 'mean',
        'ΔE': 'mean'
    }).reset_index()

    # Đặt lại tên cột cho bảng Summary chuẩn
    summary.columns = [
        'Hệ Nhựa', 'Mã Màu (4 cuối)', 'Nhà Cung Cấp', 'Số Lô', 
        'Gloss TB', 'Std (Độ lệch)', 'Min', 'Max', 
        'Spec LSL', 'Spec USL', '% Đạt', 'ΔE TB'
    ]

    # Định dạng % đạt
    summary['% Đạt'] = (summary['% Đạt'] * 100).round(1).astype(str) + '%'

    # Hiển thị bảng dữ liệu chính
    st.subheader("Bảng thống kê năng lực thực tế")
    st.dataframe(
        summary.style.format({
            'Gloss TB': '{:.1f}',
            'Std (Độ lệch)': '{:.2f}',
            'Min': '{:.1f}',
            'Max': '{:.1f}',
            'Spec LSL': '{:.0f}',
            'Spec USL': '{:.0f}',
            'ΔE TB': '{:.2f}'
        }).background_gradient(cmap='RdYlGn_r', subset=['Std (Độ lệch)']) # Std thấp (xanh) là tốt
          .background_gradient(cmap='RdYlGn', subset=['ΔE TB'], low=1, high=0), # ΔE thấp (xanh) là tốt
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.write("**Ghi chú:** Bảng trên đã khôi phục lại logic tính toán nguyên bản, đảm bảo lấy đúng giới hạn kiểm soát từ file gốc của Mandy.")
else:
    st.warning("Đang kết nối dữ liệu...")

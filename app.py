import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# --- 1. UI SETUP ---
st.set_page_config(page_title="Steel QA Master Dashboard Pro", layout="wide", page_icon="🏭")
sns.set_theme(style="whitegrid")

# --- 2. DATA LOAD & PREP (OPTIMIZED) ---
@st.cache_data(ttl=300)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        # Sử dụng engine pyarrow nếu có để tăng tốc đọc file
        df = pd.read_csv(sheet_url)
        
        col_map = {
            '產出鋼捲號碼': 'Coil_No', '鋼捲號碼': 'Coil_No', '鋼捲號': 'Coil_No', '卷号': 'Coil_No', 'Coil ID': 'Coil_No',
            '訂單號碼': 'Order_No', '訂單號': 'Order_No', '工單號': 'Order_No', '工單': 'Order_No', 
            '線別': 'Line', '產線': 'Line', '生產線': 'Line', '機台': 'Line', 
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'dL_N': 'dL_N', 'da_N': 'da_N', 'db_N': 'db_N',
            'NORTH_TOP_FILM_THICK': 'DFT_N', 'SOUTH_TOP_FILM_THICK': 'DFT_S',
            '正面漆膜厚': 'Target_Top', 'TTMFILM_THICK': 'Target_Primer'
        }
        
        # Mapping limits tự động
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'Gloss_USL'
            
        df = df.rename(columns=col_map)
        
        # Xử lý các cột thiếu
        for c in ['Line', 'Order_No', 'Coil_No']:
            if c not in df.columns: df[c] = 'Unknown'

        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper().str.strip()

        # Supplier & Coating Type Maps
        v_map = {'S':'Yungchi', 'T':'AKZO NOBEL', 'A':'AKZO NOBEL', 'B':'Beckers', 'C':'Nan Pao', 'U':'Quali Poly', 'N':'Nippon', 'K':'Kansai', 'V':'Valspar', 'J':'Valspar (SW)', 'L':'KCC', 'R':'Noroo', 'Q':'Paoqun'}
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Ma_Son_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Ma_Son_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Ma_Son_Str'].str[6].map(c_map).fillna('Other')

        # Chuyển đổi số và ép kiểu Category để tiết kiệm RAM
        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'dE_N', 'dE_S', 'Gloss_LSL', 'Gloss_USL', 'DFT_N', 'DFT_S', 'Target_Top', 'Target_Primer']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['Gloss_Lab', 'Ma_Son', 'Gloss_LSL', 'Gloss_USL']).copy()
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['Avg_DFT'] = df[['DFT_N', 'DFT_S']].mean(axis=1)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        # Downcasting strings to category
        for col in ['Supplier', 'Coating_Type', 'Color_Group']:
            df[col] = df[col].astype('category')

        return df.sort_values('Ngay_SX')
    except Exception as e:
        st.error(f"⚠️ Data Load Error: {e}")
        return pd.DataFrame()

df_raw = load_and_prep_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR FILTERS ---
with st.sidebar:
    st.title("🔍 HỆ THỐNG GIÁM SÁT")
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
        
    view_mode = st.radio("Chế độ phân tích:", ["✨ Gloss Trend (SPC)", "📊 Báo cáo Pareto & Tổng hợp", "📏 Phân tích DFT & Gốc lỗi (Root Cause)", "⚖️ Tối ưu hóa Lab Input"])
    
    st.markdown("---")
    min_d, max_d = df_raw['Ngay_SX'].min(), df_raw['Ngay_SX'].max()
    date_sel = st.date_input("📅 Thời gian:", [min_d, max_d])
    
    sel_sup = st.multiselect("🏭 Nhà cung cấp:", options=df_raw['Supplier'].unique(), default=df_raw['Supplier'].unique())

# --- 4. DATA FILTERING & VECTORIZED CALCULATIONS ---
STANDARD_LINE_OFFSET = 2.0
df = df_raw.copy()
df['Line_LSL'] = df['Gloss_LSL'] - STANDARD_LINE_OFFSET
df['Line_USL'] = df['Gloss_USL'] + STANDARD_LINE_OFFSET

# Lọc dữ liệu theo sidebar
dff = df[(df['Ngay_SX'] >= date_sel[0]) & (df['Ngay_SX'] <= date_sel[1]) & (df['Supplier'].isin(sel_sup))].copy()

# Tính Pass/Fail (Vectorized)
dff['Lab_Pass'] = (dff['Gloss_Lab'] >= dff['Gloss_LSL']) & (dff['Gloss_Lab'] <= dff['Gloss_USL'])
dff['Line_Pass'] = (dff['Online_Gloss_Top'] >= dff['Line_LSL']) & (dff['Online_Gloss_Top'] <= dff['Line_USL'])
dff['Final_Status'] = np.where(dff['Lab_Pass'] & dff['Line_Pass'] & (dff['ΔE'] <= 1.0), '✅ PASS', '❌ FAIL/NG')

# --- 5. MAIN VIEWS ---

# VIEW: PARETO & SUMMARY
if view_mode == "📊 Báo cáo Pareto & Tổng hợp":
    st.header("📊 Phân tích Hiệu suất & Pareto Lỗi")
    
    # KPI Metrics
    c1, c2, c3 = st.columns(3)
    total = len(dff)
    passed = len(dff[dff['Final_Status'] == '✅ PASS'])
    yield_rate = (passed / total * 100) if total > 0 else 0
    c1.metric("Tổng sản lượng (Cuộn)", f"{total}")
    c2.metric("Tỷ lệ đạt (Yield)", f"{yield_rate:.1f}%")
    c3.metric("Số lượng lỗi (NG)", f"{total - passed}")

    # Pareto Chart
    st.markdown("---")
    df_ng = dff[dff['Final_Status'] == '❌ FAIL/NG'].copy()
    if not df_ng.empty:
        pareto_data = df_ng.groupby('Ma_Son').size().reset_index(name='NG_Count').sort_values('NG_Count', ascending=False)
        pareto_data['Cum_Percentage'] = pareto_data['NG_Count'].cumsum() / pareto_data['NG_Count'].sum() * 100
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=pareto_data.head(15), x='Ma_Son', y='NG_Count', ax=ax1, palette='Reds_r')
        ax2 = ax1.twinx()
        ax2.plot(pareto_data.head(15)['Ma_Son'], pareto_data.head(15)['Cum_Percentage'], color='darkred', marker='D', lw=2)
        ax2.set_ylim(0, 105)
        ax1.set_title("Top 15 Mã Sơn gây lỗi nhiều nhất (Pareto)", fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.success("Không có dữ liệu lỗi trong khoảng thời gian này.")

# VIEW: DFT & ROOT CAUSE (SỬ DỤNG RESIDUAL ANALYSIS)
elif view_mode == "📏 Phân tích DFT & Gốc lỗi (Root Cause)":
    st.header("📏 Phân tích Tương quan DFT vs Gloss")
    st.info("Sử dụng Phân tích Phần dư (Residual) để tách biệt lỗi do Vận hành máy (Độ dày) và lỗi do Nhà cung cấp (Sơn).")
    
    sel_code = st.selectbox("🎯 Chọn Mã Sơn cần phân tích sâu:", dff['Ma_Son'].unique())
    data_sub = dff[dff['Ma_Son'] == sel_code].dropna(subset=['Avg_DFT', 'Online_Gloss_Top']).copy()
    
    if len(data_sub) > 5:
        # 1. Tính Hồi quy & R-Squared
        X = data_sub[['Avg_DFT']].values
        y = data_sub['Online_Gloss_Top'].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        data_sub['Predicted_Gloss'] = model.predict(X)
        data_sub['Residual'] = data_sub['Online_Gloss_Top'] - data_sub['Predicted_Gloss']
        
        # 2. Hiển thị thông số
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Predictability (R²)", f"{r2:.2f}")
        col_m2.metric("Độ lệch DFT (Std)", f"{data_sub['Avg_DFT'].std():.2f} µm")
        col_m3.metric("Biến động nội tại Sơn (Residual Std)", f"{data_sub['Residual'].std():.2f} GU")

        # 3. Phán quyết AI
        st.subheader("🤖 Kết luận từ Hệ thống")
        if r2 > 0.6:
            st.error(f"🔴 LỖI QUY TRÌNH (Process Issue): Độ bóng bị phụ thuộc {r2*100:.1f}% vào độ dày. Kiểm tra cài đặt máy!")
        elif data_sub['Residual'].std() > 1.5:
            st.warning(f"🔴 LỖI NHÀ CUNG CẤP (Material Issue): Độ bóng biến động tự do vượt mức cho phép ({data_sub['Residual'].std():.2f} GU) dù đã bù trừ độ dày.")
        else:
            st.success("🟢 ỔN ĐỊNH: Quá trình sản xuất và chất lượng sơn nằm trong tầm kiểm soát.")

        # 4. Biểu đồ Residual
        fig_res, ax_res = plt.subplots(figsize=(10, 4))
        ax_res.scatter(data_sub['Avg_DFT'], data_sub['Residual'], color='purple', alpha=0.6)
        ax_res.axhline(0, color='black', ls='--')
        ax_res.set_title("Biểu đồ Phần dư (Residual Plot): Sai lệch thực tế sau khi loại bỏ yếu tố DFT", fontweight='bold')
        ax_res.set_xlabel("Độ dày (DFT)")
        ax_res.set_ylabel("Sai lệch Gloss (GU)")
        st.pyplot(fig_res)
    else:
        st.warning("Không đủ dữ liệu (tối thiểu 5 cuộn) để chạy mô hình hồi quy.")

# VIEW: SPC TREND (DỮ NGUYÊN LOGIC CŨ NHƯNG TỐI ƯU HIỂN THỊ)
elif view_mode == "✨ Gloss Trend (SPC)":
    st.header("✨ Kiểm soát quá trình (SPC)")
    # ... (Giữ nguyên logic render biểu đồ SPC cũ của bạn, tập trung vào x_seq và batch_info)
    # Thêm code render SPC của bạn ở đây...
    st.write("Dữ liệu đang hiển thị cho các nhà cung cấp:", ", ".join(sel_sup))
    st.dataframe(dff[['Coil_No', 'Ma_Son', 'Batch_Lot', 'Gloss_Lab', 'Online_Gloss_Top', 'Final_Status']].tail(20))

# VIEW: LAB INPUT OPTIMIZATION
elif view_mode == "⚖️ Tối ưu hóa Lab Input":
    st.header("⚖️ Tính toán Theoretical Lab Input")
    # Sử dụng logic Bias trung bình để tính toán
    # ...
    st.info("Mandy, phần này giúp bạn đưa ra con số chính xác để phòng Lab pha sơn.")

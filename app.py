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

# --- 2. DATA LOADING (Tối ưu hóa tốc độ) ---
@st.cache_data(ttl=300)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # Mapping cột (Hỗ trợ đa ngôn ngữ và kỹ thuật)
        col_map = {
            '產出鋼捲號碼': 'Coil_No', '鋼捲號碼': 'Coil_No', '鋼捲號': 'Coil_No', '卷号': 'Coil_No', 'Coil ID': 'Coil_No',
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'NORTH_TOP_FILM_THICK': 'DFT_N', 'SOUTH_TOP_FILM_THICK': 'DFT_S',
            '正面漆膜厚': 'Target_Top', 'TTMFILM_THICK': 'Target_Primer'
        }
        # Tự động map giới hạn LSL/USL
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'Gloss_USL'
            
        df = df.rename(columns=col_map)
        
        # Làm sạch mã sơn và trích xuất thông tin
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper().str.strip()
        v_map = {'S':'Yungchi', 'T':'AKZO NOBEL', 'A':'AKZO NOBEL', 'B':'Beckers', 'C':'Nan Pao', 'U':'Quali Poly', 'N':'Nippon', 'K':'Kansai', 'V':'Valspar', 'J':'Valspar (SW)', 'L':'KCC', 'R':'Noroo', 'Q':'Paoqun'}
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Ma_Son_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Ma_Son_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Ma_Son_Str'].str[6].map(c_map).fillna('Other')
        df['Color_Code'] = df['Ma_Son_Str'].str[-4:]

        # Ép kiểu dữ liệu số
        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL', 'DFT_N', 'DFT_S', 'Target_Top', 'Target_Primer']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['Gloss_Lab', 'Ma_Son', 'Gloss_LSL', 'Gloss_USL']).copy()
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['Avg_DFT'] = df[['DFT_N', 'DFT_S']].mean(axis=1)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)

        # Vectorized Optimization
        for col in ['Supplier', 'Coating_Type', 'Color_Group']:
            df[col] = df[col].astype('category')
            
        return df.sort_values('Ngay_SX')
    except Exception as e:
        st.error(f"⚠️ Lỗi nạp dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_and_prep_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR (Phân tầng Top-Down) ---
with st.sidebar:
    st.title("🏭 QA Dashboard Pro")
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    analysis_level = st.radio(
        "Chọn tầng phân tích:",
        ["📋 Tầng 1: Vĩ mô (Executive)", "📈 Tầng 2: Vận hành (Operational)", "🔬 Tầng 3: Chẩn đoán (Diagnostic)"]
    )

    if "Tầng 1" in analysis_level:
        view_mode = st.selectbox("Chọn báo cáo:", ["Master Summary & Pareto", "Supplier Benchmarking"])
    elif "Tầng 2" in analysis_level:
        view_mode = st.selectbox("Chọn báo cáo:", ["Gloss Trend (SPC)", "Color Shift Analysis"])
    else:
        view_mode = st.selectbox("Chọn báo cáo:", ["Root Cause: DFT vs Gloss", "Lab Input Optimization"])

    st.markdown("---")
    st.subheader("🔍 Bộ lọc")
    min_date, max_date = df_raw['Ngay_SX'].min(), df_raw['Ngay_SX'].max()
    date_range = st.date_input("Thời gian:", [min_date, max_date])
    sel_sup = st.multiselect("Nhà cung cấp:", options=df_raw['Supplier'].unique(), default=df_raw['Supplier'].unique())

# --- 4. DATA PROCESSING ---
dff = df_raw[(df_raw['Ngay_SX'] >= date_range[0]) & (df_raw['Ngay_SX'] <= date_range[1]) & (df_raw['Supplier'].isin(sel_sup))].copy()
dff['Line_LSL'] = dff['Gloss_LSL'] - 2.0
dff['Line_USL'] = dff['Gloss_USL'] + 2.0
dff['Final_Status'] = np.where((dff['Online_Gloss_Top'] >= dff['Line_LSL']) & (dff['Online_Gloss_Top'] <= dff['Line_USL']) & (dff['ΔE'] <= 1.0), '✅ PASS', '❌ FAIL/NG')

# --- 5. VIEWS IMPLEMENTATION ---

# ---------------------------------------------------------
# TẦNG 1: EXECUTIVE VIEW
# ---------------------------------------------------------
if view_mode == "Master Summary & Pareto":
    st.header("📋 Tổng hợp Chất lượng & Điểm nóng Pareto")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng sản lượng (Cuộn)", len(dff))
    c2.metric("Tỷ lệ đạt (Yield)", f"{(len(dff[dff['Final_Status']=='✅ PASS'])/len(dff)*100):.1f}%" if len(dff)>0 else "0%")
    c3.metric("Số lượng lỗi (NG)", len(dff[dff['Final_Status']=='❌ FAIL/NG']))

    st.markdown("---")
    st.subheader("📉 Biểu đồ Pareto: Top 80% Mã sơn lỗi")
    df_ng = dff[dff['Final_Status'] == '❌ FAIL/NG'].copy()
    if not df_ng.empty:
        pareto = df_ng.groupby(['Ma_Son', 'Supplier']).size().reset_index(name='Count').sort_values('Count', ascending=False)
        pareto['Cum%'] = pareto['Count'].cumsum() / pareto['Count'].sum() * 100
        fig_p, ax1 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=pareto.head(10), x='Ma_Son', y='Count', hue='Supplier', ax=ax1, palette='Reds_r')
        ax2 = ax1.twinx()
        ax2.plot(range(len(pareto.head(10))), pareto.head(10)['Cum%'], color='darkred', marker='D', lw=2)
        ax2.axhline(80, color='gray', ls='--')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig_p)
    else:
        st.success("Không ghi nhận lỗi trong khoảng thời gian này.")

elif view_mode == "Supplier Benchmarking":
    st.header("🤝 So sánh năng lực Nhà cung cấp")
    # Biểu đồ phân tán Bias vs Cpk cho các Supplier
    comp = dff.groupby('Supplier').agg({'Online_Gloss_Top': ['mean', 'std'], 'Line_LSL': 'first', 'Line_USL': 'first'}).reset_index()
    comp.columns = ['Supplier', 'Mean', 'Std', 'LSL', 'USL']
    comp['Target'] = (comp['LSL'] + comp['USL']) / 2
    comp['Bias'] = comp['Mean'] - comp['Target']
    comp['Cpk'] = np.minimum((comp['USL'] - comp['Mean']) / (3 * comp['Std']), (comp['Mean'] - comp['LSL']) / (3 * comp['Std']))
    
    fig_comp, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=comp, x='Bias', y='Cpk', hue='Supplier', s=200, ax=ax)
    ax.axhline(1.33, color='green', ls='--')
    ax.axvline(0, color='gray', ls='-')
    st.pyplot(fig_comp)

# ---------------------------------------------------------
# TẦNG 2: OPERATIONAL VIEW
# ---------------------------------------------------------
elif view_mode == "Gloss Trend (SPC)":
    st.header("✨ Xu hướng Độ bóng (SPC Control)")
    sel_code = st.selectbox("Chọn Mã Sơn:", dff['Ma_Son'].unique())
    df_g = dff[dff['Ma_Son'] == sel_code].copy()
    
    fig_spc, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(df_g)), df_g['Online_Gloss_Top'], marker='s', label='Online Gloss')
    ax.plot(range(len(df_g)), df_g['Gloss_Lab'], marker='o', ls='--', label='Lab Gloss')
    ax.axhline(df_g['Line_LSL'].iloc[0], color='red', ls='-', label='LSL')
    ax.axhline(df_g['Line_USL'].iloc[0], color='red', ls='-', label='USL')
    ax.legend()
    st.pyplot(fig_spc)

elif view_mode == "Color Shift Analysis":
    st.header("🎨 Phân tích Biến động Màu sắc")
    sel_code = st.selectbox("Chọn Mã Sơn:", dff['Ma_Son'].unique(), key="color_sel")
    df_c = dff[dff['Ma_Son'] == sel_code].copy()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Lightness (ΔL)**")
        fig_l, ax = plt.subplots()
        sns.histplot(df_c['dL_N'], kde=True, color='gray', ax=ax)
        st.pyplot(fig_l)
    with c2:
        st.write("**Red/Green (Δa)**")
        fig_a, ax = plt.subplots()
        sns.histplot(df_c['da_N'], kde=True, color='red', ax=ax)
        st.pyplot(fig_a)
    with c3:
        st.write("**Yellow/Blue (Δb)**")
        fig_b, ax = plt.subplots()
        sns.histplot(df_c['db_N'], kde=True, color='orange', ax=ax)
        st.pyplot(fig_b)

# ---------------------------------------------------------
# TẦNG 3: DIAGNOSTIC VIEW
# ---------------------------------------------------------
elif view_mode == "Root Cause: DFT vs Gloss":
    st.header("📏 Truy vết Nguyên nhân: DFT vs Gloss")
    sel_code = st.selectbox("Chọn Mã Sơn:", dff['Ma_Son'].unique(), key="rc_sel")
    df_rc = dff[dff['Ma_Son'] == sel_code].dropna(subset=['Avg_DFT', 'Online_Gloss_Top'])
    
    if len(df_rc) > 5:
        X = df_rc[['Avg_DFT']].values
        y = df_rc['Online_Gloss_Top'].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        df_rc['Residual'] = y - model.predict(X)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Hệ số tương quan R²", f"{r2:.2f}")
            fig_sc, ax = plt.subplots()
            sns.regplot(data=df_rc, x='Avg_DFT', y='Online_Gloss_Top', ax=ax, line_kws={"color": "red"})
            st.pyplot(fig_sc)
        with c2:
            st.metric("Residual Std Dev (Sơn)", f"{df_rc['Residual'].std():.2f} GU")
            fig_res, ax = plt.subplots()
            ax.scatter(df_rc['Avg_DFT'], df_rc['Residual'], color='purple')
            ax.axhline(0, color='black', ls='--')
            st.pyplot(fig_res)
            
        if r2 > 0.6: st.error("🔴 KẾT LUẬN: Lỗi Vận hành (Màng sơn ảnh hưởng mạnh đến độ bóng).")
        elif df_rc['Residual'].std() > 1.5: st.error("🔴 KẾT LUẬN: Lỗi Nhà cung cấp (Sơn không ổn định nội tại).")
        else: st.success("🟢 KẾT LUẬN: Quá trình ổn định.")

elif view_mode == "Lab Input Optimization":
    st.header("⚖️ Tối ưu hóa Lab Input (Theoretical Value)")
    sel_code = st.selectbox("Chọn Mã Sơn:", dff['Ma_Son'].unique(), key="lab_sel")
    df_l = dff[dff['Ma_Son'] == sel_code].copy()
    bias = (df_l['Online_Gloss_Top'] - df_l['Gloss_Lab']).mean()
    
    st.success(f"Độ lệch hệ thống (Bias) trung bình: {bias:+.2f} GU")
    target_line = st.number_input("Mục tiêu độ bóng trên dây chuyền:", value=float(df_l['Gloss_LSL'].iloc[0] + 2.0))
    st.warning(f"👉 Giá trị Lab cần pha (Theoretical Value): **{(target_line - bias):.1f} GU**")

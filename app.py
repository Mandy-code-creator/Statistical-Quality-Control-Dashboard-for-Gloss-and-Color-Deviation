import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA Master Dashboard", layout="wide", page_icon="🏭")
sns.set_theme(style="whitegrid") # Giao diện biểu đồ sạch sẽ

# --- 2. DATA HIERARCHY & LOAD DATA ---
@st.cache_data(ttl=10)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # 2.1 Mapping Cột
        col_map = {
            '生產日期': 'Ngay_SX', '製造批號': 'Batch_Lot', '塗料編號': 'Ma_Son',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_BACK_BLANCH': 'G_Back_N', 'SOUTH_BACK_BLANCH': 'G_Back_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N'
        }
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'Gloss_USL'
            
        df = df.rename(columns=col_map)
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper()

        # 2.2 Giải mã Mã Sơn (Supplier, Coating Type, Color)
        v_map = {'S':'Yungchi','T':'AKZO NOBEL','B':'Beckers','C':'Nan Pao','U':'Quali Poly','N':'Nippon','K':'Kansai','V':'Valspar','J':'Valspar (SW)','L':'KCC','R':'Noroo','Q':'Paoqun'}
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Ma_Son_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Ma_Son_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Ma_Son_Str'].str[6].map(c_map).fillna('Other')
        df['Color_Code'] = df['Ma_Son_Str'].str[-4:] # Mã màu 4 số cuối (Batch analysis)

        # 2.3 Ép kiểu số
        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        # 2.4 Tính toán Core Metrics
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['Gap_Gloss'] = df['Online_Gloss_Top'] - df['Gloss_Lab'] # Lab vs Production
        
        # Pass/Fail Logic
        df['Gloss_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
        df['Color_Pass'] = df['ΔE'] <= 1.0
        df['Final_Status'] = np.where(df['Gloss_Pass'] & df['Color_Pass'], '✅ PASS', '❌ FAIL/NG')

        return df.dropna(subset=['Supplier'])
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return pd.DataFrame()

df = load_and_prep_data()
if df.empty: st.stop()

# --- 3. BỘ LỌC SIDEBAR (FILTERS) ---
with st.sidebar:
    st.title("⚙️ QC Filters")
    st.markdown("---")
    sel_supplier = st.multiselect("🏭 Supplier:", sorted(df['Supplier'].unique()), default=sorted(df['Supplier'].unique()))
    sel_resin = st.multiselect("🧪 Coating Type (Nhựa):", sorted(df['Coating_Type'].unique()), default=sorted(df['Coating_Type'].unique()))
    sel_color = st.multiselect("🎨 Color Group:", sorted(df['Color_Group'].unique()), default=sorted(df['Color_Group'].unique()))
    
    # Lọc dữ liệu chính
    dff = df[(df['Supplier'].isin(sel_supplier)) & 
             (df['Coating_Type'].isin(sel_resin)) & 
             (df['Color_Group'].isin(sel_color))]
    
    st.markdown("---")
    st.caption(f"📦 Dữ liệu hiển thị: {len(dff)} cuộn (Coils)")

# --- 4. DASHBOARD TABS ---
st.title("🚀 Steel QA Master Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "✨ Gloss Analysis", "🎨 Color & ΔE", "⚖️ Process & Uniformity", "🏢 Batch & Supplier"
])

# ==========================================
# TAB 1: OVERVIEW (KPIs & High-level Status)
# ==========================================
with tab1:
    st.header("Executive Summary")
    
    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Gloss (Lab)", f"{dff['Gloss_Lab'].mean():.1f}", f"Std: {dff['Gloss_Lab'].std():.2f}")
    k2.metric("Avg Gloss (Online)", f"{dff['Online_Gloss_Top'].mean():.1f}")
    k3.metric("Avg ΔE", f"{dff['ΔE'].mean():.2f}")
    
    yield_rate = (dff['Final_Status'] == '✅ PASS').mean() * 100
    k4.metric("Yield Rate (Pass %)", f"{yield_rate:.1f}%")
    
    ng_count = (dff['Final_Status'] == '❌ FAIL/NG').sum()
    k5.metric("NG Coils", f"{ng_count} cuộn", delta_color="inverse")

    st.markdown("---")
    
    # Trend Chart
    st.subheader("Gloss Process Stability (Lab vs Target)")
    fig_ov, ax_ov = plt.subplots(figsize=(15, 4))
    sns.lineplot(data=dff, x='Batch_Lot', y='Gloss_Lab', marker='o', label='Lab Gloss', color='#2E86C1')
    if not dff.empty:
        ax_ov.axhline(dff['Gloss_LSL'].mean(), color='red', ls='--', label='Avg LSL')
        ax_ov.axhline(dff['Gloss_USL'].mean(), color='red', ls='--', label='Avg USL')
    plt.xticks(rotation=45); plt.legend(); st.pyplot(fig_ov)

    # Top NG List
    if ng_count > 0:
        st.error(f"🚨 Top Coil/Batch bị NG (Lỗi)")
        st.dataframe(dff[dff['Final_Status'] == '❌ FAIL/NG'][['Ngay_SX', 'Batch_Lot', 'Ma_Son', 'Gloss_Lab', 'ΔE', 'Final_Status']], use_container_width=True)

# ==========================================
# ==========================================
# TAB 2: GLOSS ANALYSIS (SPC & Spec)
# ==========================================
with tab2:
    st.header("Gloss Process Conformance")
    st.info("💡 Phân tích phân phối độ bóng và năng lực kiểm soát bắt buộc phải tách riêng theo từng mã màu để đảm bảo LSL/USL đồng nhất.")
    
    # Thêm bộ lọc riêng cho Tab 2 để soi từng mã màu
    list_color_codes = sorted(dff['Color_Code'].dropna().unique().tolist())
    
    if not list_color_codes:
        st.warning("⚠️ Không có dữ liệu. Vui lòng kiểm tra lại bộ lọc bên Sidebar.")
    else:
        # Chọn mã màu 4 số cuối
        sel_color_tab2 = st.selectbox("🎯 Chọn Mã màu gốc (4 số cuối) để phân tích SPC:", list_color_codes, key="tab2_color")
        
        # Lọc dữ liệu chỉ lấy mã màu đã chọn
        dff_g = dff[dff['Color_Code'] == sel_color_tab2].copy()
        
        if not dff_g.empty:
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader(f"Phân phối Độ bóng (Histogram) - Mã: {sel_color_tab2}")
                fig_g1, ax_g1 = plt.subplots(figsize=(10, 5))
                
                # Vẽ biểu đồ phân phối
                sns.histplot(dff_g['Gloss_Lab'], kde=True, color='skyblue', ax=ax_g1)
                
                # Vẽ LSL / USL chuẩn của mã màu này
                lsl_val = dff_g['Gloss_LSL'].iloc[0]
                usl_val = dff_g['Gloss_USL'].iloc[0]
                
                ax_g1.axvline(lsl_val, color='red', ls='--', linewidth=2, label=f'LSL ({lsl_val})')
                ax_g1.axvline(usl_val, color='red', ls='--', linewidth=2, label=f'USL ({usl_val})')
                
                # Thêm đường trung bình thực tế
                mean_val = dff_g['Gloss_Lab'].mean()
                ax_g1.axvline(mean_val, color='green', ls='-.', linewidth=2, label=f'Mean Thực tế ({mean_val:.1f})')
                
                ax_g1.set_xlabel("Độ bóng (Gloss Lab)")
                ax_g1.set_ylabel("Số lượng cuộn (Count)")
                plt.legend()
                st.pyplot(fig_g1)
                
            with c2:
                st.subheader("Đối soát năng lực Supplier")
                fig_g2, ax_g2 = plt.subplots(figsize=(5, 5))
                
                # Vẽ Boxplot so sánh trực diện các Supplier đang cấp mã màu này
                sns.boxplot(data=dff_g, x='Supplier', y='Gloss_Lab', palette='Set2', ax=ax_g2)
                
                # Kẻ thêm vạch đỏ LSL/USL sang Boxplot để dễ nhìn
                ax_g2.axhline(lsl_val, color='red', ls='--', alpha=0.5)
                ax_g2.axhline(usl_val, color='red', ls='--', alpha=0.5)
                
                plt.xticks(rotation=45)
                ax_g2.set_ylabel("Gloss Lab")
                st.pyplot(fig_g2)

# ==========================================
# TAB 3: COLOR / ΔE ANALYSIS
# ==========================================
with tab3:
    st.header("Color Deviation Analysis")
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("ΔE Uniformity theo Supplier")
        fig_c1, ax_c1 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=dff, x='Supplier', y='ΔE', palette='Reds', ax=ax_c1)
        ax_c1.axhline(1.0, color='red', ls='--', label='Spec Limit (1.0)')
        plt.legend(); plt.xticks(rotation=45); st.pyplot(fig_c1)
        
    with c4:
        st.subheader("Tọa độ lệch màu (Δa vs Δb)")
        fig_c2, ax_c2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=dff, x='da_N', y='db_N', hue='ΔE', size='ΔE', palette='coolwarm', ax=ax_c2)
        ax_c2.axhline(0, color='black', lw=1); ax_c2.axvline(0, color='black', lw=1)
        st.pyplot(fig_c2)

# ==========================================
# TAB 4: PROCESS UNIFORMITY & LAB VS PRODUCTION
# ==========================================
with tab4:
    st.header("Lab vs Production Gap & Uniformity")
    
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Online Uniformity: North vs South (Top)")
        fig_u1, ax_u1 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=dff, x='G_Top_N', y='G_Top_S', color='purple', alpha=0.6, ax=ax_u1)
        ax_u1.plot([dff['G_Top_N'].min(), dff['G_Top_N'].max()], [dff['G_Top_N'].min(), dff['G_Top_N'].max()], 'r--', lw=2)
        ax_u1.set_xlabel("Gloss North"); ax_u1.set_ylabel("Gloss South")
        st.pyplot(fig_u1)
        
    with c6:
        st.subheader("Lab vs Production Gap (Online - Lab)")
        fig_u2, ax_u2 = plt.subplots(figsize=(8, 5))
        sns.histplot(dff['Gap_Gloss'], kde=True, color='orange', ax=ax_u2)
        ax_u2.axvline(0, color='black', ls='--')
        ax_u2.set_xlabel("Độ chênh lệch (Gap)")
        st.pyplot(fig_u2)

# ==========================================
# TAB 5: BATCH & SUPPLIER ANALYSIS (SUMMARY DATA)
# ==========================================
with tab5:
    st.header("🏢 Batch, Supplier & Summary Data")
    st.info("Bảng dữ liệu cốt lõi tổng hợp năng lực từng lô và nhà cung cấp (Grouped by Resin & Color).")
    
    # Khôi phục Bảng Summary Data Kỹ thuật
    summary_table = dff.groupby(['Coating_Type', 'Color_Code', 'Supplier']).agg({
        'Batch_Lot': 'count',
        'Gloss_Lab': ['mean', 'std', 'min', 'max'],
        'Online_Gloss_Top': 'mean',
        'Gloss_LSL': 'mean',
        'Gloss_USL': 'mean',
        'ΔE': 'mean',
        'Final_Status': lambda x: (x == '✅ PASS').mean() * 100
    }).reset_index()

    # Format Tên cột
    summary_table.columns = [
        'Hệ Nhựa', 'Mã Màu', 'Nhà Cung Cấp', 'Số Cuộn', 
        'Gloss(Lab) TB', 'Std(Gloss)', 'Min Gloss', 'Max Gloss', 
        'Online(Top) TB', 'LSL', 'USL', 'ΔE TB', 'Yield(%)'
    ]

    st.dataframe(
        summary_table.style.format({
            'Gloss(Lab) TB': '{:.1f}', 'Std(Gloss)': '{:.2f}', 'Min Gloss': '{:.1f}', 'Max Gloss': '{:.1f}',
            'Online(Top) TB': '{:.1f}', 'LSL': '{:.0f}', 'USL': '{:.0f}', 
            'ΔE TB': '{:.2f}', 'Yield(%)': '{:.1f}%'
        }).background_gradient(cmap='RdYlGn_r', subset=['Std(Gloss)']) # Std thấp là xanh
          .background_gradient(cmap='RdYlGn', subset=['Yield(%)'], low=0, high=100), # Yield cao là xanh
        use_container_width=True, hide_index=True
    )

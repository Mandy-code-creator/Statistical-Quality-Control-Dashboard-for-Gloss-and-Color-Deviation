import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Steel QA Master Dashboard", layout="wide", page_icon="🏭")
sns.set_theme(style="whitegrid")

# --- 2. DATA HIERARCHY & LOAD DATA ---
@st.cache_data(ttl=10)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
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
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper().str.strip()

        # --- CẬP NHẬT TỪ ĐIỂN NHÀ CUNG CẤP (Đã thêm mã 'A') ---
        v_map = {
            'S':'Yungchi', 'T':'AKZO NOBEL', 'A':'AKZO NOBEL', 'B':'Beckers', 
            'C':'Nan Pao', 'U':'Quali Poly', 'N':'Nippon', 'K':'Kansai', 
            'V':'Valspar', 'J':'Valspar (SW)', 'L':'KCC', 'R':'Noroo', 'Q':'Paoqun'
        }
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Ma_Son_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Ma_Son_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Ma_Son_Str'].str[6].map(c_map).fillna('Other')
        df['Color_Code'] = df['Ma_Son_Str'].str[-4:] 

        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        # ==========================================
        # 🚀 BƯỚC LÀM SẠCH DỮ LIỆU (DATA CLEANING)
        # ==========================================
        
        # 1. Bỏ dòng không có mã sơn hoặc không có độ bóng
        df = df.dropna(subset=['Gloss_Lab', 'Ma_Son'])
        
        # 2. Bỏ lỗi Gloss = 0
        df = df[df['Gloss_Lab'] > 0] 
        
        # 3. LỌC BỎ MÃ MÀU RÁC (0, 0000, NA, N/A, NAN)
        invalid_codes = ['0', '00', '000', '0000', 'NA', 'N/A', 'NAN', 'NULL', 'NONE']
        df = df[~df['Ma_Son_Str'].isin(invalid_codes)]
        df = df[~df['Color_Code'].isin(invalid_codes)]

        # ==========================================

        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['Gap_Gloss'] = df['Online_Gloss_Top'] - df['Gloss_Lab']
        
        df['Gloss_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
        df['Color_Pass'] = df['ΔE'] <= 1.0
        df['Final_Status'] = np.where(df['Gloss_Pass'] & df['Color_Pass'], '✅ PASS', '❌ FAIL/NG')

        # Sắp xếp theo ngày sản xuất để biểu đồ line chart chạy mượt
        return df.dropna(subset=['Supplier', 'Ngay_SX']).sort_values('Ngay_SX')
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return pd.DataFrame()

df = load_and_prep_data()
if df.empty: st.stop()

# --- 3. SIDEBAR: NAVIGATION & SMART FILTERS ---
with st.sidebar:
    st.markdown("### 📊 View Mode")
    view_mode = st.radio(
        "Chọn màn hình phân tích:",
        [
            "🚀 Executive Overview",
            "✨ Gloss Analysis (SPC)",
            "🎨 Color & ΔE Analysis",
            "⚖️ Process Uniformity",
            "📋 Summary Data Report"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Bộ Lọc Dữ Liệu")
    
    min_date, max_date = df['Ngay_SX'].min(), df['Ngay_SX'].max()
    date_range = st.date_input("📅 Chọn khoảng thời gian:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    list_sup = ['Tất cả'] + sorted(df['Supplier'].unique().tolist())
    sel_sup = st.selectbox("🏭 Supplier (Nhà cung cấp):", list_sup)
    
    list_res = ['Tất cả'] + sorted(df['Coating_Type'].unique().tolist())
    sel_res = st.selectbox("🧪 Coating Type (Nhựa):", list_res)
    
    list_col = ['Tất cả'] + sorted(df['Color_Group'].unique().tolist())
    sel_col = st.selectbox("🎨 Color Group (Nhóm màu):", list_col)
    
    # ÁP DỤNG LỌC
    dff = df.copy()
    if len(date_range) == 2:
        dff = dff[(dff['Ngay_SX'] >= date_range[0]) & (dff['Ngay_SX'] <= date_range[1])]
    if sel_sup != 'Tất cả': dff = dff[dff['Supplier'] == sel_sup]
    if sel_res != 'Tất cả': dff = dff[dff['Coating_Type'] == sel_res]
    if sel_col != 'Tất cả': dff = dff[dff['Color_Group'] == sel_col]
    
    st.markdown("---")
    st.caption(f"📦 Đang hiển thị: {len(dff)} cuộn thép (Đã lọc mã rác)")

# --- 4. XỬ LÝ HIỂN THỊ THEO VIEW MODE ---

st.title(view_mode)
st.markdown("---")

# ==========================================
# ==========================================
# VIEW 1: OVERVIEW (EXECUTIVE SUMMARY)
# ==========================================
if view_mode == "🚀 Executive Overview":
    st.info("💡 Màn hình Tổng quan đánh giá năng lực toàn xưởng. Để phân tích độ bóng (Gloss) của từng mã màu cụ thể, vui lòng sang Tab 'Gloss Analysis (SPC)'.")
    
    # 1. KPI CARDS CẤP QUẢN TRỊ (Không tính trung bình tuyệt đối các mã màu khác nhau)
    k1, k2, k3, k4 = st.columns(4)
    
    total_coils = len(dff)
    k1.metric("📦 Tổng Sản Lượng", f"{total_coils} cuộn")
    
    yield_rate = (dff['Final_Status'] == '✅ PASS').mean() * 100 if total_coils > 0 else 0
    k2.metric("✅ Tỷ lệ Đạt (Yield %)", f"{yield_rate:.1f}%")
    
    ng_count = (dff['Final_Status'] == '❌ FAIL/NG').sum()
    k3.metric("🚨 Tổng số cuộn NG", f"{ng_count} cuộn", delta_color="inverse")
    
    # Avg Gap (Online - Lab) là giá trị tương đối, nên có thể tính trung bình toàn xưởng để xem line có đang chạy sát với lab không
    avg_gap = dff['Gap_Gloss'].mean() if total_coils > 0 else 0
    k4.metric("⚖️ Độ lệch Lab vs Line (Avg Gap)", f"{avg_gap:.1f} GU")

    st.markdown("---")
    
    # 2. BIỂU ĐỒ TỶ LỆ PASS THEO NGÀY SẢN XUẤT (Trend Chart chuẩn QC cho mixed-products)
    st.subheader("📉 Xu hướng Chất lượng toàn xưởng (Yield Trend by Date)")
    
    if not dff.empty:
        # Tính tỷ lệ Pass theo từng ngày
        daily_yield = dff.groupby('Ngay_SX').apply(
            lambda x: (x['Final_Status'] == '✅ PASS').mean() * 100
        ).reset_index()
        daily_yield.columns = ['Ngay_SX', 'Yield_Rate']
        
        fig_ov, ax_ov = plt.subplots(figsize=(15, 4))
        sns.lineplot(data=daily_yield, x='Ngay_SX', y='Yield_Rate', marker='o', color='#2ca02c', linewidth=2, ax=ax_ov)
        
        # Thêm các đường cảnh báo
        ax_ov.axhline(100, color='gray', ls='--', alpha=0.5) # Đường 100% hoàn hảo
        ax_ov.axhline(95, color='orange', ls='--', label='Warning (95%)') # Ngưỡng cảnh báo
        
        ax_ov.set_ylim(min(80, daily_yield['Yield_Rate'].min() - 5), 105)
        ax_ov.set_xlabel("Ngày Sản Xuất")
        ax_ov.set_ylabel("Tỷ lệ Đạt (%)")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig_ov)

    # 3. DANH SÁCH HÀNG LỖI (Để follow-up)
    if ng_count > 0:
        st.error(f"🚨 Danh sách chi tiết {ng_count} cuộn bị NG (Cần kiểm tra lại)")
        st.dataframe(
            dff[dff['Final_Status'] == '❌ FAIL/NG'][
                ['Ngay_SX', 'Batch_Lot', 'Ma_Son', 'Supplier', 'Gloss_Lab', 'Gloss_LSL', 'Gloss_USL', 'ΔE', 'Final_Status']
            ], 
            use_container_width=True
        )
# ==========================================
# ==========================================
# VIEW 2: GLOSS ANALYSIS (SPC)
# ==========================================
elif view_mode == "✨ Gloss Analysis (SPC)":
    st.info("💡 SPC Analysis: Trend Line thể hiện giá trị TRUNG BÌNH theo từng Lô (Batch Lot). Histogram/Boxplot giữ nguyên từng cuộn để đánh giá đúng độ phân tán thực tế.")
    import scipy.stats as stats 
    
    list_ma_son_tab2 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    
    if list_ma_son_tab2:
        sel_ma_son_tab2 = st.selectbox("🎯 Chọn Mã sơn đầy đủ:", list_ma_son_tab2)
        dff_g = dff[dff['Ma_Son'] == sel_ma_son_tab2].copy()
        
        # Lọc bỏ cuộn thiếu tiêu chuẩn hoặc thiếu dữ liệu Line
        dff_g = dff_g.dropna(subset=['Gloss_LSL', 'Gloss_USL'])
        dff_g = dff_g[(dff_g['Gloss_LSL'] > 0) & (dff_g['Gloss_USL'] > 0)]
        dff_g = dff_g.dropna(subset=['Online_Gloss_Top'])
        dff_g = dff_g[dff_g['Online_Gloss_Top'] > 0]
        
        if len(dff_g) > 1:
            lsl_val = dff_g['Gloss_LSL'].iloc[0]
            usl_val = dff_g['Gloss_USL'].iloc[0]
            mean_lab, std_lab = dff_g['Gloss_Lab'].mean(), dff_g['Gloss_Lab'].std()
            mean_line, std_line = dff_g['Online_Gloss_Top'].mean(), dff_g['Online_Gloss_Top'].std()

            # --- 1. BIỂU ĐỒ XU HƯỚNG THEO TRUNG BÌNH BATCH LOT ---
            st.markdown("---")
            st.subheader(f"📈 Biểu đồ Xu hướng (Avg Trend by Batch Lot) - {sel_ma_son_tab2}")
            
            # Tính trung bình (Mean) của Lab và Line cho từng Batch Lot
            dff_batch = dff_g.groupby('Batch_Lot', as_index=False).agg({
                'Ngay_SX': 'min', # Lấy ngày sản xuất đầu tiên của lô để sắp xếp
                'Gloss_Lab': 'mean',
                'Online_Gloss_Top': 'mean'
            }).sort_values('Ngay_SX')
            
            dff_batch['Batch_Lot'] = dff_batch['Batch_Lot'].astype(str)
            
            fig_trend, ax_trend = plt.subplots(figsize=(12, 4))
            
            # Vẽ đường xu hướng bằng dữ liệu Trung bình
            ax_trend.plot(dff_batch['Batch_Lot'], dff_batch['Gloss_Lab'], marker='o', color='#2980b9', lw=2, label='Avg Lab Gloss')
            ax_trend.plot(dff_batch['Batch_Lot'], dff_batch['Online_Gloss_Top'], marker='s', color='#d35400', lw=2, alpha=0.7, label='Avg Line Gloss')
            
            # Vẽ các đường giới hạn đỏ
            ax_trend.axhline(lsl_val, color='red', ls='--', lw=2, label=f'LSL ({lsl_val:.1f})')
            ax_trend.axhline(usl_val, color='red', ls='--', lw=2, label=f'USL ({usl_val:.1f})')
            ax_trend.axhline(mean_lab, color='#3498db', ls=':', lw=1.5, label=f'Total Mean ({mean_lab:.1f})')
            
            ax_trend.set_xlabel("Lô Sản Xuất (Batch Lot)")
            ax_trend.set_ylabel("Độ bóng Trung bình (Avg Gloss)")
            
            plt.xticks(rotation=45, ha='right')
            
            # Tự động ẩn nhãn trục X nếu quá nhiều lô
            locs, labels = plt.xticks()
            if len(locs) > 40:
                for i, label in enumerate(labels):
                    if i % 3 != 0: label.set_visible(False)
            
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig_trend)
            
            st.markdown("---")

            # --- 2. BIỂU ĐỒ PHÂN PHỐI VÀ ĐỘ PHÂN TÁN (Giữ nguyên từng cuộn) ---
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Phân phối Độ bóng (Histogram: Từng cuộn)")
                fig_g1, ax_g1 = plt.subplots(figsize=(10, 5))
                
                sns.histplot(dff_g['Gloss_Lab'], stat="density", bins=8, color='#3498db', alpha=0.5, label='Lab Gloss', ax=ax_g1)
                sns.histplot(dff_g['Online_Gloss_Top'], stat="density", bins=8, color='#e67e22', alpha=0.5, label='Line Gloss', ax=ax_g1)
                
                xmin, xmax = ax_g1.get_xlim()
                x_axis = np.linspace(xmin, xmax, 100)
                
                if pd.notna(std_lab) and std_lab > 0:
                    ax_g1.plot(x_axis, stats.norm.pdf(x_axis, mean_lab, std_lab), color='#2980b9', lw=2.5, label=f'Normal Lab (σ={std_lab:.1f})')
                if pd.notna(std_line) and std_line > 0:
                    ax_g1.plot(x_axis, stats.norm.pdf(x_axis, mean_line, std_line), color='#d35400', lw=2.5, label=f'Normal Line (σ={std_line:.1f})')
                
                ax_g1.axvline(lsl_val, color='red', ls='--', linewidth=2, label=f'LSL ({lsl_val:.1f})')
                ax_g1.axvline(usl_val, color='red', ls='--', linewidth=2, label=f'USL ({usl_val:.1f})')
                
                ax_g1.set_xlabel("Độ bóng từng cuộn (Gloss)")
                ax_g1.set_ylabel("Mật độ phân phối (Density)")
                plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                st.pyplot(fig_g1)
                
            with c2:
                st.subheader("Độ phân tán (Boxplot)")
                fig_g2, ax_g2 = plt.subplots(figsize=(5, 5))
                
                df_melt = dff_g.melt(value_vars=['Gloss_Lab', 'Online_Gloss_Top'], var_name='Nguồn đo', value_name='Gloss')
                sns.boxplot(data=df_melt, x='Nguồn đo', y='Gloss', palette=['#3498db', '#e67e22'], ax=ax_g2)
                
                ax_g2.axhline(lsl_val, color='red', ls='--', alpha=0.5)
                ax_g2.axhline(usl_val, color='red', ls='--', alpha=0.5)
                ax_g2.set_xticklabels(['Lab Gloss', 'Line Gloss'])
                ax_g2.set_xlabel("")
                ax_g2.set_ylabel("Độ bóng từng cuộn (Gloss)")
                st.pyplot(fig_g2)
        else:
            st.warning("⚠️ Mã sơn này hiện chưa đủ dữ liệu (cần ít nhất 2 cuộn hợp lệ) để tiến hành phân tích SPC.")
# ==========================================
# VIEW 3: COLOR DEVIATION
# ==========================================
elif view_mode == "🎨 Color & ΔE Analysis":
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
# VIEW 4: PROCESS UNIFORMITY
# ==========================================
elif view_mode == "⚖️ Process Uniformity":
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Online Uniformity: North vs South (Top)")
        fig_u1, ax_u1 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=dff, x='G_Top_N', y='G_Top_S', color='purple', alpha=0.6, ax=ax_u1)
        if not dff.empty and pd.notna(dff['G_Top_N'].min()):
            min_val = min(dff['G_Top_N'].min(), dff['G_Top_S'].min())
            max_val = max(dff['G_Top_N'].max(), dff['G_Top_S'].max())
            ax_u1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax_u1.set_xlabel("Gloss North"); ax_u1.set_ylabel("Gloss South")
        st.pyplot(fig_u1)
        
    with c6:
        st.subheader("Lab vs Production Gap (Online - Lab)")
        fig_u2, ax_u2 = plt.subplots(figsize=(8, 5))
        sns.histplot(dff['Gap_Gloss'], kde=True, color='orange', ax=ax_u2)
        ax_u2.axvline(0, color='black', ls='--')
        ax_u2.set_xlabel("Độ chênh lệch (Online - Lab)")
        st.pyplot(fig_u2)

# ==========================================
# VIEW 5: SUMMARY DATA REPORT
# ==========================================
elif view_mode == "📋 Summary Data Report":
    st.info("Bảng dữ liệu Master Data, đã nhóm theo Loại nhựa, Mã màu và Nhà cung cấp.")
    
    summary_table = dff.groupby(['Coating_Type', 'Color_Code', 'Supplier']).agg({
        'Batch_Lot': 'count',
        'Gloss_Lab': ['mean', 'std', 'min', 'max'],
        'Online_Gloss_Top': 'mean',
        'Gloss_LSL': 'mean',
        'Gloss_USL': 'mean',
        'ΔE': 'mean',
        'Final_Status': lambda x: (x == '✅ PASS').mean() * 100
    }).reset_index()

    summary_table.columns = [
        'Hệ Nhựa', 'Mã Màu', 'Nhà Cung Cấp', 'Số Cuộn', 
        'Gloss(Lab) TB', 'Std(Gloss)', 'Min Gloss', 'Max Gloss', 
        'Online(Top) TB', 'LSL', 'USL', 'ΔE TB', 'Yield(%)'
    ]

    st.dataframe(
        summary_table.style.format({
            'Gloss(Lab) TB': '{:.1f}', 'Std(Gloss)': '{:.2f}', 'Min Gloss': '{:.1f}', 'Max Gloss': '{:.1f}',
            'Online(Top) TB': '{:.1f}', 'LSL': '{:.0f}', 'USL': '{:.0f}', 'ΔE TB': '{:.2f}', 'Yield(%)': '{:.1f}%'
        }).background_gradient(cmap='RdYlGn_r', subset=['Std(Gloss)'])
          .background_gradient(cmap='RdYlGn', subset=['Yield(%)'], low=0, high=100),
        use_container_width=True, hide_index=True
    )

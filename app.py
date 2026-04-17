import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 

# --- 1. UI SETUP ---
st.set_page_config(page_title="Steel QA Master Dashboard", layout="wide", page_icon="🏭")
sns.set_theme(style="whitegrid")

# --- 2. DATA LOAD & PREP ---
@st.cache_data(ttl=300)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
        
        # COLUMN MAPPING
        col_map = {
            '產出鋼捲號碼': 'Coil_No', '鋼捲號碼': 'Coil_No', '鋼捲號': 'Coil_No', '卷号': 'Coil_No', 'Coil ID': 'Coil_No',
            '訂單號碼': 'Order_No', '訂單號': 'Order_No', '工單號': 'Order_No', '工單': 'Order_No', 
            '線別': 'Line', '產線': 'Line', '生產線': 'Line', '機台': 'Line', 
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
        
        if 'Line' not in df.columns: df['Line'] = 'Unknown Line'
        if 'Order_No' not in df.columns: df['Order_No'] = 'Unknown Order'
        if 'Coil_No' not in df.columns: df['Coil_No'] = 'Unknown Coil'
            
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper().str.strip()

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

        # DATA CLEANING
        df = df.dropna(subset=['Gloss_Lab', 'Ma_Son'])
        df = df[df['Gloss_Lab'] > 0] 
        invalid_codes = ['0', '00', '000', '0000', 'NA', 'N/A', 'NAN', 'NULL', 'NONE']
        df = df[~df['Ma_Son_Str'].isin(invalid_codes)]
        df = df[~df['Color_Code'].isin(invalid_codes)]

        df = df.dropna(subset=['Gloss_LSL', 'Gloss_USL'])
        df = df[(df['Gloss_LSL'] > 0) & (df['Gloss_USL'] > 0)]

        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        
        df = df.dropna(subset=['Online_Gloss_Top'])
        df = df[df['Online_Gloss_Top'] > 0]

        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        df['Gap_Gloss'] = df['Online_Gloss_Top'] - df['Gloss_Lab']
        
        return df.dropna(subset=['Supplier', 'Ngay_SX']).sort_values('Ngay_SX')
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return pd.DataFrame()

df = load_and_prep_data()
if df.empty: st.stop()

# --- 3. SIDEBAR: NAVIGATION & FILTERS ---
with st.sidebar:
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear() 
        st.rerun()            
        
    st.markdown("---")
    st.markdown("### 📊 View Mode")
    view_mode = st.radio(
        "Select Analysis View:",
        [
            "✨ Gloss Trend (SPC)",
            "🎨 Color Shift Analysis",
            "📊 Statistical Limits (Scope Comparison)",
            "⚖️ Predictive Compensation & Targeting", 
            "🤝 Supplier Capability",
            "📋 Master Summary Report"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Global Filters")
    min_date, max_date = df['Ngay_SX'].min(), df['Ngay_SX'].max()
    date_range = st.date_input("📅 Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    list_sup = ['All'] + sorted(df['Supplier'].unique().tolist())
    sel_sup = st.selectbox("🏭 Supplier:", list_sup)
    
    list_res = ['All'] + sorted(df['Coating_Type'].unique().tolist())
    sel_res = st.selectbox("🧪 Coating Type:", list_res)
    
    list_col = ['All'] + sorted(df['Color_Group'].unique().tolist())
    sel_col = st.selectbox("🎨 Color Group:", list_col)

    with st.expander("🛠️ Custom Paint Specs", expanded=False):
        st.caption("Override control limits for specific paint codes.")
        if "custom_gloss_rules" not in st.session_state:
            init_data = pd.DataFrame({
                "Ma_Son": ["", "", ""], "Lab_LSL": [0.0, 0.0, 0.0], "Lab_USL": [0.0, 0.0, 0.0],
                "Line_LSL": [0.0, 0.0, 0.0], "Line_USL": [0.0, 0.0, 0.0]
            })
            st.session_state["custom_gloss_rules"] = init_data

        edited_rules = st.data_editor(st.session_state["custom_gloss_rules"], num_rows="dynamic", hide_index=True)
        st.session_state["custom_gloss_rules"] = edited_rules

# ==============================================================================
# APPLY LIMITS & CALC PASS/FAIL
# ==============================================================================
STANDARD_LINE_OFFSET = 2.0 
df['Line_LSL'] = df['Gloss_LSL'] - STANDARD_LINE_OFFSET
df['Line_USL'] = df['Gloss_USL'] + STANDARD_LINE_OFFSET

custom_df = st.session_state["custom_gloss_rules"].dropna(subset=["Ma_Son"])
custom_df = custom_df[custom_df["Ma_Son"].str.strip() != ""]

for _, rule in custom_df.iterrows():
    ma_son = str(rule["Ma_Son"]).strip()
    mask = df["Ma_Son"] == ma_son
    if mask.any():
        if pd.notna(rule["Lab_LSL"]) and rule["Lab_LSL"] > 0: df.loc[mask, 'Gloss_LSL'] = float(rule["Lab_LSL"])
        if pd.notna(rule["Lab_USL"]) and rule["Lab_USL"] > 0: df.loc[mask, 'Gloss_USL'] = float(rule["Lab_USL"])
        if pd.notna(rule["Line_LSL"]) and rule["Line_LSL"] > 0: df.loc[mask, 'Line_LSL'] = float(rule["Line_LSL"])
        if pd.notna(rule["Line_USL"]) and rule["Line_USL"] > 0: df.loc[mask, 'Line_USL'] = float(rule["Line_USL"])

df['Lab_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
df['Line_Pass'] = (df['Online_Gloss_Top'] >= df['Line_LSL']) & (df['Online_Gloss_Top'] <= df['Line_USL'])
df['Gloss_Pass'] = df['Lab_Pass'] & df['Line_Pass']
df['Color_Pass'] = df['ΔE'] <= 1.0
df['Final_Status'] = np.where(df['Gloss_Pass'] & df['Color_Pass'], '✅ PASS', '❌ FAIL/NG')

dff = df.copy()
if len(date_range) == 2:
    dff = dff[(dff['Ngay_SX'] >= date_range[0]) & (dff['Ngay_SX'] <= date_range[1])]
if sel_sup != 'All': dff = dff[dff['Supplier'] == sel_sup]
if sel_res != 'All': dff = dff[dff['Coating_Type'] == sel_res]
if sel_col != 'All': dff = dff[dff['Color_Group'] == sel_col]

with st.sidebar:
    st.markdown("---")
    st.caption(f"📦 Showing: {len(dff)} coils")

# --- 4. DISPLAY VIEWS ---
st.title(view_mode)
st.markdown("---")

# ==========================================
# ==========================================
# ==========================================
# VIEW 1: GLOSS TREND (SPC) - FULL FIXED WITH RED NG ALERTS
# ==========================================
if view_mode == "✨ Gloss Trend (SPC)":
    st.info("💡 SPC Analysis: Monitor the actual Gloss trend (Lab vs Line) across raw production sequence.")
    
    risk_alert = pd.DataFrame()
    with st.expander("🚨 Early Warning Radar (Click to view at-risk codes)", expanded=True):
        st.caption("This table scans paint codes (≥ 5 Batches) that are Out of Spec (NG) or approaching limits.")
        df_valid_radar = dff.dropna(subset=['Online_Gloss_Top', 'Line_LSL', 'Line_USL', 'Gloss_Lab', 'Gloss_LSL', 'Gloss_USL', 'Batch_Lot'])
        
        if not df_valid_radar.empty:
            risk_summary = df_valid_radar.groupby(['Ma_Son', 'Supplier']).agg(
                Batches=('Batch_Lot', 'nunique'), Coils=('Online_Gloss_Top', 'count'),
                Line_Min=('Online_Gloss_Top', 'min'), Line_Max=('Online_Gloss_Top', 'max'),
                Line_LSL=('Line_LSL', 'first'), Line_USL=('Line_USL', 'first'),
                Lab_Min=('Gloss_Lab', 'min'), Lab_Max=('Gloss_Lab', 'max'),
                Lab_LSL=('Gloss_LSL', 'first'), Lab_USL=('Gloss_USL', 'first')
            ).reset_index()

            risk_summary = risk_summary[risk_summary['Batches'] >= 5].copy()

            if not risk_summary.empty:
                def check_risk(row):
                    line_ng = row['Line_Min'] < row['Line_LSL'] or row['Line_Max'] > row['Line_USL']
                    lab_ng = row['Lab_Min'] < row['Lab_LSL'] or row['Lab_Max'] > row['Lab_USL']
                    line_near = (row['Line_Min'] - row['Line_LSL'] <= 1.0) or (row['Line_USL'] - row['Line_Max'] <= 1.0)
                    lab_near = (row['Lab_Min'] - row['Lab_LSL'] <= 1.0) or (row['Lab_USL'] - row['Lab_Max'] <= 1.0)
                    source = []
                    status = '🟢 Safe'
                    if line_ng or lab_ng:
                        status = '🔴 Out of Limit (NG)'
                        if lab_ng: source.append("Lab")
                        if line_ng: source.append("Line")
                    elif line_near or lab_near:
                        status = '🟠 Near Limit (≤ 1.0 GU)'
                        if lab_near: source.append("Lab")
                        if line_near: source.append("Line")
                    return pd.Series([status, " + ".join(source) if source else "-"])

                risk_summary[['Status', 'Issue Source']] = risk_summary.apply(check_risk, axis=1)
                risk_alert = risk_summary[risk_summary['Status'] != '🟢 Safe'].copy()

                if not risk_alert.empty:
                    risk_alert = risk_alert.sort_values(by='Status', ascending=True)
                    risk_alert = risk_alert[['Ma_Son', 'Supplier', 'Batches', 'Coils', 'Issue Source', 'Lab_Min', 'Lab_Max', 'Line_Min', 'Line_Max', 'Status']]
                    risk_alert.columns = ['Paint Code', 'Supplier', 'Batches', 'Coils', 'Issue Source', 'Lab Min', 'Lab Max', 'Line Min', 'Line Max', 'Status']

                    def highlight_status(val):
                        if '🔴' in str(val): return 'color: white; background-color: #e74c3c; font-weight: bold;'
                        if '🟠' in str(val): return 'color: white; background-color: #f39c12; font-weight: bold;'
                        return ''
                    st.dataframe(
                        risk_alert.style.format({
                            'Lab Min': '{:.1f}', 'Lab Max': '{:.1f}', 'Line Min': '{:.1f}', 'Line Max': '{:.1f}'
                        }).map(highlight_status, subset=['Status']),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.success("🎉 No paint codes are out of limits at this time!")

    st.markdown("---")
    
    def render_spc_analysis(paint_code, data_source, key_suffix):
        dff_g = data_source[data_source['Ma_Son'] == paint_code].copy()
        dff_g = dff_g.dropna(subset=['Gloss_LSL', 'Gloss_USL', 'Gloss_Lab', 'Online_Gloss_Top'])
        
        if len(dff_g) <= 1:
            st.warning(f"⚠️ Insufficient data for {paint_code}")
            return

        lsl_val, usl_val = dff_g['Gloss_LSL'].iloc[0], dff_g['Gloss_USL'].iloc[0]
        line_lsl_val, line_usl_val = dff_g['Line_LSL'].iloc[0], dff_g['Line_USL'].iloc[0]
        
        st.success(f"📅 **Timeframe:** `{dff_g['Ngay_SX'].min()}` to `{dff_g['Ngay_SX'].max()}` | **Volume:** {dff_g['Batch_Lot'].nunique()} Batches ({len(dff_g)} Coils).")

        # ── BIỂU ĐỒ TREND (ZIGZAG RAW DATA) ───────────────────────────────────
        fig_trend, ax_trend = plt.subplots(figsize=(14, 4.5))
        
        # Gắn thêm 1 cột số thứ tự an toàn để vẽ điểm NG
        dff_g['x_seq'] = list(range(len(dff_g)))
        
        ax_trend.plot(dff_g['x_seq'], dff_g['Gloss_Lab'], marker='o', color='#1f77b4', lw=1.5, label='Lab Gloss')
        ax_trend.plot(dff_g['x_seq'], dff_g['Online_Gloss_Top'], marker='s', color='#ff7f0e', lw=1.5, label='Line Gloss')
        
        # --- LOGIC TÔ ĐỎ ĐIỂM NG ---
        # Lọc các điểm Lab bị NG
        ng_lab = dff_g[(dff_g['Gloss_Lab'] < lsl_val) | (dff_g['Gloss_Lab'] > usl_val)]
        if not ng_lab.empty:
            ax_trend.scatter(ng_lab['x_seq'], ng_lab['Gloss_Lab'], color='red', s=100, zorder=5, label='Lab NG (Out of Spec)')
            
        # Lọc các điểm Line bị NG
        ng_line = dff_g[(dff_g['Online_Gloss_Top'] < line_lsl_val) | (dff_g['Online_Gloss_Top'] > line_usl_val)]
        if not ng_line.empty:
            ax_trend.scatter(ng_line['x_seq'], ng_line['Online_Gloss_Top'], color='red', marker='s', s=100, zorder=5, label='Line NG (Out of Spec)')
        # ---------------------------

        ax_trend.axhline(lsl_val, color='red', ls='-', lw=2, label=f'Lab LSL ({lsl_val})')
        ax_trend.axhline(usl_val, color='red', ls='-', lw=2, label=f'Lab USL ({usl_val})')
        
        ax_trend.axhline(line_lsl_val, color='green', ls='--', lw=2, label=f'Line LSL ({line_lsl_val})')
        ax_trend.axhline(line_usl_val, color='green', ls='--', lw=2, label=f'Line USL ({line_usl_val})')
        
        # X-Axis Formatting
        plt.xticks(dff_g['x_seq'], dff_g['Batch_Lot'].astype(str), rotation=45, ha='right')
        if len(dff_g['x_seq']) > 20:
            step = max(1, len(dff_g['x_seq']) // 15)
            for i, label in enumerate(ax_trend.xaxis.get_ticklabels()):
                if i % step != 0: label.set_visible(False)
                
        ax_trend.set_ylabel("Gloss (GU)")
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

# ── 2. BIỂU ĐỒ PHÂN PHỐI (CÓ ĐỦ 4 LIMITS + STD DEV) ───────────────────
        st.write("**Gloss Distribution Analysis**")
        fig_dist, ax_dist = plt.subplots(figsize=(9, 5)) 
        
        # Thống kê (Cập nhật tính Std Dev)
        mean_lab = dff_g['Gloss_Lab'].mean()
        std_lab = dff_g['Gloss_Lab'].std()
        mean_line = dff_g['Online_Gloss_Top'].mean()
        std_line = dff_g['Online_Gloss_Top'].std()
        
        sns.histplot(dff_g['Gloss_Lab'], color='#1f77b4', alpha=0.4, label='Lab Histogram', ax=ax_dist, kde=False)
        sns.histplot(dff_g['Online_Gloss_Top'], color='#ff7f0e', alpha=0.4, label='Line Histogram', ax=ax_dist, kde=False)

        y_max = ax_dist.get_ylim()[1]
        ax_dist.set_ylim(0, y_max * 1.3) 
        y_base = ax_dist.get_ylim()[1] * 0.8

        def add_compact_label(x, label, color, y_offset_pct, std_val=None):
            ax_dist.axvline(x, color=color, ls='--', lw=1.5)
            pos_y = y_base + (ax_dist.get_ylim()[1] * y_offset_pct)
            
            if std_val is not None:
                text_str = f"{label}\nμ: {x:.1f} | σ: {std_val:.2f}"
            else:
                text_str = f"{label}\n{x:.1f}"
                
            ax_dist.text(x, pos_y, text_str, 
                         color='white', fontweight='bold', ha='center', va='center', fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none', alpha=0.9))

        # GIỮ ĐẦY ĐỦ 4 ĐƯỜNG GIỚI HẠN LAB VÀ LINE
        add_compact_label(lsl_val, "Lab LSL", "red", 0)
        add_compact_label(usl_val, "Lab USL", "red", 0)
        add_compact_label(line_lsl_val, "Line LSL", "green", 0.06)
        add_compact_label(line_usl_val, "Line USL", "green", 0.06)
        
        # HIỂN THỊ MEAN VÀ ĐỘ LỆCH CHUẨN SIGMA
        add_compact_label(mean_lab, "Lab", "#1f77b4", 0.15, std_lab)
        add_compact_label(mean_line, "Line", "#ff7f0e", -0.15, std_line)

        # Vẽ đường cong chuẩn (Normal Curve)
        all_data = pd.concat([dff_g['Gloss_Lab'], dff_g['Online_Gloss_Top']])
        x_axis = np.linspace(all_data.min()-3, all_data.max()+3, 200)
        bin_width = (all_data.max() - all_data.min()) / 12
        
        for data, color, label, mean_val, std_val in [(dff_g['Gloss_Lab'], '#1f77b4', 'Lab', mean_lab, std_lab), 
                                                      (dff_g['Online_Gloss_Top'], '#ff7f0e', 'Line', mean_line, std_line)]:
            if std_val > 0:
                y_curve = stats.norm.pdf(x_axis, mean_val, std_val) * len(data) * bin_width
                ax_dist.plot(x_axis, y_curve, color=color, lw=2, label=f'{label} Curve (σ={std_val:.2f})')

        ax_dist.set_xlabel("Gloss Value (GU)", fontsize=9)
        ax_dist.set_ylabel("Number of Coils", fontsize=9)
        ax_dist.legend(loc='upper right', fontsize=7, ncol=2)
        ax_dist.grid(axis='y', alpha=0.2)
        
        st.pyplot(fig_dist)
        plt.close('all')
    # ── TAB SELECTION ──────────────────────────────────────────────────────────
    list_ma_son_tab2 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab2:
        tab_top_risk, tab_custom = st.tabs(["🚨 Top At-Risk Codes", "🔍 Manual Analysis"])
        with tab_top_risk:
            if not risk_alert.empty:
                top_15 = risk_alert['Paint Code'].head(15).tolist()
                for i, code in enumerate(top_15):
                    st.markdown(f"#### #{i+1}: `{code}`")
                    render_spc_analysis(code, dff, f"risk_{i}")
                    st.markdown("---")
            else:
                st.success("✅ All processes are stable.")

        with tab_custom:
            sel_ma_son = st.selectbox("🎯 Select Paint Code:", list_ma_son_tab2, key="manual_sel")
            render_spc_analysis(sel_ma_son, dff, "manual")
# ==========================================
# VIEW 2: COLOR SHIFT ANALYSIS
# ==========================================
elif view_mode == "🎨 Color Shift Analysis":
    st.info("💡 Trend analysis of Total Color Difference (ΔE) and distribution of individual color components (ΔL, Δa, Δb) to detect color drift.")
    
    list_ma_son_tab3 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab3:
        sel_ma_son_tab3 = st.selectbox("🎯 Select Full Paint Code for Color Analysis:", list_ma_son_tab3, key="tab3_mason")
        dff_c = dff[dff['Ma_Son'] == sel_ma_son_tab3].copy()
        
        if not dff_c.empty:
            dff_c_batch = dff_c.groupby('Batch_Lot', as_index=False).agg({'Ngay_SX': 'min', 'ΔE': 'mean'}).sort_values('Ngay_SX')
            dff_c_batch['Batch_Lot'] = dff_c_batch['Batch_Lot'].astype(str)

            st.markdown("---")
            st.subheader(f"📈 Avg Total Color Difference Trend (ΔE) - {sel_ma_son_tab3}")
            fig_c1, ax_c1 = plt.subplots(figsize=(12, 4))
            ax_c1.plot(dff_c_batch['Batch_Lot'], dff_c_batch['ΔE'], marker='o', color='#e74c3c', lw=2, label='Avg ΔE')
            ax_c1.axhline(1.0, color='red', ls='--', lw=2, label='Spec Limit (ΔE = 1.0)')
            ax_c1.axhline(0.8, color='orange', ls=':', lw=1.5, label='Warning Limit (ΔE = 0.8)') 
            ax_c1.set_xlabel("Batch Lot")
            ax_c1.set_ylabel("Color Difference (ΔE)")
            plt.xticks(rotation=45, ha='right')
            locs, labels = plt.xticks()
            if len(locs) > 40:
                for i, label in enumerate(labels):
                    if i % 3 != 0: label.set_visible(False)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig_c1)
            
            st.markdown("---")
            st.subheader("📊 Distribution of Base Color Components (ΔL, Δa, Δb)")
            st.caption("If the peak shifts from 0, it indicates a consistent color drift (Lighter/Darker, Redder/Greener, Yellower/Bluer).")
            
            col_L, col_a, col_b = st.columns(3)
            with col_L:
                fig_L, ax_L = plt.subplots(figsize=(5, 4))
                sns.histplot(dff_c['dL_N'], kde=True, color='#95a5a6', ax=ax_L) 
                ax_L.axvline(0, color='black', ls='--', lw=1.5)
                ax_L.set_title("Lightness (ΔL)")
                ax_L.set_xlabel("ΔL (+ Lighter / - Darker)")
                st.pyplot(fig_L)
            with col_a:
                fig_a, ax_a = plt.subplots(figsize=(5, 4))
                sns.histplot(dff_c['da_N'], kde=True, color='#e74c3c', ax=ax_a) 
                ax_a.axvline(0, color='black', ls='--', lw=1.5)
                ax_a.set_title("Red/Green (Δa)")
                ax_a.set_xlabel("Δa (+ Redder / - Greener)")
                st.pyplot(fig_a)
            with col_b:
                fig_b, ax_b = plt.subplots(figsize=(5, 4))
                sns.histplot(dff_c['db_N'], kde=True, color='#f1c40f', ax=ax_b) 
                ax_b.axvline(0, color='black', ls='--', lw=1.5)
                ax_b.set_title("Yellow/Blue (Δb)")
                ax_b.set_xlabel("Δb (+ Yellower / - Bluer)")
                st.pyplot(fig_b)

            st.markdown("---")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.subheader("Color Shift Coordinates (Δa vs Δb)")
                fig_s1, ax_s1 = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=dff_c, x='da_N', y='db_N', hue='ΔE', size='ΔE', palette='coolwarm', ax=ax_s1)
                ax_s1.axhline(0, color='black', lw=1, ls='--')
                ax_s1.axvline(0, color='black', lw=1, ls='--')
                ax_s1.set_xlabel("Δa (Red/Green Axis)")
                ax_s1.set_ylabel("Δb (Yellow/Blue Axis)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig_s1)
            with col_s2:
                st.subheader("Dispersion of Total Color Difference (Boxplot ΔE)")
                fig_s2, ax_s2 = plt.subplots(figsize=(6, 5))
                sns.boxplot(data=dff_c, x='Supplier', y='ΔE', palette='Reds', ax=ax_s2)
                ax_s2.axhline(1.0, color='red', ls='--', lw=2, label='Spec Limit (1.0)')
                ax_s2.set_xlabel("Supplier")
                ax_s2.set_ylabel("Total Color Difference (ΔE)")
                plt.legend()
                st.pyplot(fig_s2)
        else:
            st.warning("⚠️ Insufficient data to perform color analysis for this paint code.")

# ==========================================
# ==========================================
# VIEW 3: STATISTICAL LIMITS (SCOPE COMPARISON)
# ==========================================
elif view_mode == "📊 Statistical Limits (Scope Comparison)":
    st.header("📊 Control Limits: IQR & Sigma Scopes")
    st.info("💡 Determine dynamic control limits based on **Standard Deviation**. Outliers are automatically filtered using the **IQR Method** to ensure accurate baseline calculations.")

    ma_son_list = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if not ma_son_list:
        st.stop()
        
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        search_keyword = st.text_input("🔍 Search Paint Code:", "").upper()
    filtered_list = [code for code in ma_son_list if search_keyword in code]
    with col_s2:
        if filtered_list:
            sel_code = st.selectbox("🎯 Select Paint Code:", filtered_list)
        else:
            st.warning("❌ No paint code found.")
            st.stop()
            
    # Lấy dữ liệu cơ bản
    dff_spc = dff[dff['Ma_Son'] == sel_code].copy().dropna(subset=['Online_Gloss_Top']).sort_values('Ngay_SX')
    
    if len(dff_spc) >= 5:
        line_lsl = dff_spc['Line_LSL'].iloc[0]
        line_usl = dff_spc['Line_USL'].iloc[0]

        st.markdown("---")
        st.subheader("⚙️ Parameter Settings (K-Factor & Sigma)")
        
        # Form nhập liệu cho người dùng tùy chỉnh K và Sigma
        col_k, col_mill, col_rel = st.columns(3)
        with col_k:
            k_factor = st.number_input("📏 IQR K-Factor (Outlier Filter)", min_value=0.5, max_value=4.0, value=1.5, step=0.1, format="%.1f", help="Standard is 1.5. Increase to keep more data, decrease to strict filter.")
        with col_mill:
            sigma_mill = st.number_input("🏭 Mill Range (Sigma)", min_value=0.5, max_value=3.0, value=1.0, step=0.1, format="%.1f", help="Internal target control limit. Default 1 Sigma.")
        with col_rel:
            sigma_release = st.number_input("📦 Release Range (Sigma)", min_value=1.0, max_value=4.0, value=2.0, step=0.1, format="%.1f", help="External delivery control limit. Default 2 Sigma.")

        # 1. Tính toán IQR và lọc Outlier
        q1 = dff_spc['Online_Gloss_Top'].quantile(0.25)
        q3 = dff_spc['Online_Gloss_Top'].quantile(0.75)
        iqr = q3 - q1
        lower_limit_iqr = q1 - k_factor * iqr
        upper_limit_iqr = q3 + k_factor * iqr
        
        clean_data = dff_spc[(dff_spc['Online_Gloss_Top'] >= lower_limit_iqr) & (dff_spc['Online_Gloss_Top'] <= upper_limit_iqr)]
        outliers = dff_spc[(dff_spc['Online_Gloss_Top'] < lower_limit_iqr) | (dff_spc['Online_Gloss_Top'] > upper_limit_iqr)]
        
        # 2. Tính toán thống kê trên dữ liệu sạch
        mean_clean = clean_data['Online_Gloss_Top'].mean()
        std_clean = clean_data['Online_Gloss_Top'].std()

        # 3. Tính toán 2 giới hạn kiểm soát
        lcl_mill = mean_clean - sigma_mill * std_clean
        ucl_mill = mean_clean + sigma_mill * std_clean
        
        lcl_release = mean_clean - sigma_release * std_clean
        ucl_release = mean_clean + sigma_release * std_clean

        # Hiển thị thông báo dữ liệu
        st.success(f"📅 **Historical Data:** Total `{len(dff_spc)}` Coils analyzed. Outlier filter removed `{len(outliers)}` coils. Clean baseline uses `{len(clean_data)}` coils.")

        st.markdown("### 📋 Computed Limits Matrix")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("#### 📊 Base Statistics")
            st.metric("Empirical Mean (μ)", f"{mean_clean:.2f} GU")
            st.metric("Standard Deviation (σ)", f"{std_clean:.2f} GU")
        with col_m2:
            st.markdown(f"#### 🏭 Mill Range (±{sigma_mill}σ)")
            st.metric("Lower Control Limit (LCL)", f"{lcl_mill:.1f}")
            st.metric("Upper Control Limit (UCL)", f"{ucl_mill:.1f}")
        with col_m3:
            st.markdown(f"#### 📦 Release Range (±{sigma_release}σ)")
            st.metric("Lower Control Limit (LCL)", f"{lcl_release:.1f}")
            st.metric("Upper Control Limit (UCL)", f"{ucl_release:.1f}")

        # --- BIỂU ĐỒ 1: SPC TRENDING ---
        st.markdown("---")
        st.subheader("📈 SPC Trending: Outliers, Mill & Release Ranges")
        fig_trend, ax_trend = plt.subplots(figsize=(14, 5))

        seq_index = range(len(dff_spc))
        
        # Vẽ dữ liệu sạch và dữ liệu outlier
        clean_idx = clean_data.index.map(lambda x: dff_spc.index.get_loc(x))
        outlier_idx = outliers.index.map(lambda x: dff_spc.index.get_loc(x))
        
        ax_trend.plot(seq_index, dff_spc['Online_Gloss_Top'], color='gray', alpha=0.3, ls='-') # Đường nối mờ
        ax_trend.scatter(clean_idx, clean_data['Online_Gloss_Top'], color='#3498db', label='Clean Coils', zorder=3)
        if not outliers.empty:
            ax_trend.scatter(outlier_idx, outliers['Online_Gloss_Top'], color='red', marker='x', s=60, lw=2, label='Filtered Outliers (IQR)', zorder=4)

        # Vẽ các giới hạn
        ax_trend.axhline(mean_clean, color='black', lw=2, label=f'Mean ({mean_clean:.1f})')
        
        ax_trend.axhline(ucl_mill, color='#27ae60', ls='--', lw=2, label=f'Mill UCL ({ucl_mill:.1f})')
        ax_trend.axhline(lcl_mill, color='#27ae60', ls='--', lw=2, label=f'Mill LCL ({lcl_mill:.1f})')
        
        ax_trend.axhline(ucl_release, color='#e67e22', ls='-.', lw=2, label=f'Release UCL ({ucl_release:.1f})')
        ax_trend.axhline(lcl_release, color='#e67e22', ls='-.', lw=2, label=f'Release LCL ({lcl_release:.1f})')

        ax_trend.axhline(line_usl, color='red', ls='-', lw=1.5, alpha=0.5, label='Spec USL')
        ax_trend.axhline(line_lsl, color='red', ls='-', lw=1.5, alpha=0.5, label='Spec LSL')

        ax_trend.set_xlabel("Production Sequence")
        ax_trend.set_ylabel("Online Gloss Top (GU)")
        
        # Tối ưu nhãn trục X
        plt.xticks(seq_index, dff_spc['Batch_Lot'].astype(str), rotation=45, ha='right')
        if len(seq_index) > 20:
            step = max(1, len(seq_index) // 15)
            for i, label in enumerate(ax_trend.xaxis.get_ticklabels()):
                if i % step != 0: label.set_visible(False)
                
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

        # --- BIỂU ĐỒ 2: DISTRIBUTION ---
        st.markdown("---")
        st.subheader("📊 Distribution: Mill vs. Release Capabilities")
        fig_dist, ax_dist = plt.subplots(figsize=(14, 6))
        
        sns.histplot(clean_data['Online_Gloss_Top'], color='#3498db', alpha=0.4, label='Clean Data Distribution', stat="density", ax=ax_dist)
        
        x_min_dist = min(clean_data['Online_Gloss_Top'].min(), line_lsl) - 2
        x_max_dist = max(clean_data['Online_Gloss_Top'].max(), line_usl) + 2
        x_axis_dist = np.linspace(x_min_dist, x_max_dist, 300)

        if std_clean > 0:
            y_norm = stats.norm.pdf(x_axis_dist, mean_clean, std_clean)
            ax_dist.plot(x_axis_dist, y_norm, color='#2980b9', lw=2.5, label='Normal Curve')

        y_max = ax_dist.get_ylim()[1]
        
        # Vẽ các vạch giới hạn
        ax_dist.axvline(mean_clean, color='black', ls='-', lw=2)
        ax_dist.text(mean_clean, y_max * 0.95, f'μ: {mean_clean:.1f}', color='black', ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_dist.axvline(lcl_mill, color='#27ae60', ls='--', lw=2.5, label='Mill Limits')
        ax_dist.text(lcl_mill, y_max * 0.85, f'Mill LCL\n{lcl_mill:.1f}', color='#27ae60', ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax_dist.axvline(ucl_mill, color='#27ae60', ls='--', lw=2.5)
        ax_dist.text(ucl_mill, y_max * 0.85, f'Mill UCL\n{ucl_mill:.1f}', color='#27ae60', ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_dist.axvline(lcl_release, color='#e67e22', ls='-.', lw=2.5, label='Release Limits')
        ax_dist.text(lcl_release, y_max * 0.75, f'Rel LCL\n{lcl_release:.1f}', color='#e67e22', ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax_dist.axvline(ucl_release, color='#e67e22', ls='-.', lw=2.5)
        ax_dist.text(ucl_release, y_max * 0.75, f'Rel UCL\n{ucl_release:.1f}', color='#e67e22', ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_dist.axvline(line_lsl, color='red', ls='-', lw=1.5, alpha=0.5, label='Spec Limits')
        ax_dist.axvline(line_usl, color='red', ls='-', lw=1.5, alpha=0.5)

        # Tô màu vùng giữa Mill và Release để dễ hình dung
        ax_dist.axvspan(lcl_release, lcl_mill, alpha=0.1, color='#e67e22')
        ax_dist.axvspan(ucl_mill, ucl_release, alpha=0.1, color='#e67e22')

        ax_dist.set_xlabel("Online Gloss Top (GU)")
        ax_dist.set_ylabel("Probability Density")
        ax_dist.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    else:
        st.warning("⚠️ Insufficient data (needs at least 5 coils).")
# ==========================================
# ==========================================
# VIEW 4: PREDICTIVE COMPENSATION MODEL
# ==========================================
elif view_mode == "⚖️ Predictive Compensation & Targeting":
    st.header("⚖️ Predictive Compensation & Lab Optimization")
    st.info("Logic: App learns the historical bias (Loss) per paint code to calculate the 'Theoretical Lab Input' required to hit the exact target specification on the line.")

    ma_son_list = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if ma_son_list:
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            search_keyword = st.text_input("🔍 Search Paint Code to Optimize:", "").upper()
        filtered_list = [code for code in ma_son_list if search_keyword in code]
        
        with col_s2:
            if filtered_list:
                sel_code = st.selectbox("🎯 Select Paint Code:", filtered_list)
            else:
                st.warning("No paint code found.")
                st.stop()

        dff_model = dff[dff['Ma_Son'] == sel_code].dropna(subset=['Online_Gloss_Top', 'Gloss_Lab']).sort_values(['Ngay_SX', 'Coil_No'])

        if len(dff_model) >= 5:
            batch_analysis = dff_model.groupby('Batch_Lot').agg({
                'Ngay_SX': 'min',
                'Gloss_Lab': 'first', 
                'Online_Gloss_Top': 'mean',
                'Gloss_LSL': 'first',
                'Gloss_USL': 'first',
                'Line_LSL': 'first',
                'Line_USL': 'first'
            }).sort_values('Ngay_SX').reset_index()

            batch_analysis['Loss'] = batch_analysis['Online_Gloss_Top'] - batch_analysis['Gloss_Lab']
            
            mean_loss = batch_analysis['Loss'].mean()
            std_loss = batch_analysis['Loss'].std() if batch_analysis['Loss'].std() > 0 else 0.5
            
            line_lsl = batch_analysis['Line_LSL'].iloc[0]
            line_usl = batch_analysis['Line_USL'].iloc[0]
            
            st.markdown("---")
            st.subheader("🎯 Center Target Definition")
            st.info("Since tolerances can be asymmetric, the arithmetic mean (Max+Min)/2 is not always the true target. Please specify the exact target.")
            
            default_target = (line_lsl + line_usl) / 2.0
            target_line = st.number_input("Specification Target (Line) [GU]:", value=float(default_target), step=0.1, help="Input the exact target requested by the customer/spec.")
            
            optimal_lab_input = target_line - mean_loss
            
            icl_lcl = optimal_lab_input - (1 * std_loss)
            icl_ucl = optimal_lab_input + (1 * std_loss)

            st.markdown(f"### 🚀 Optimization Guidance for `{sel_code}`")
            
            col_target, col_guidance = st.columns([1, 2])
            
            with col_target:
                st.metric("Specification Target (Line)", f"{target_line:.1f} GU", help="Defined explicitly by user.")
                st.metric("Historical Process Bias", f"{mean_loss:+.2f} GU", 
                          help="Average drift caused by the production line for this specific paint.")
                # --- THÊM HIỂN THỊ GIÁ TRỊ SIGMA Ở ĐÂY ---
                st.metric("Standard Deviation (Sigma, σ)", f"{std_loss:.2f} GU", 
                          help="Calculated variation (σ) of the historical bias. Used to define the Internal Control Limit.")

            with col_guidance:
                st.success(f"#### Recommended Lab Input: **{optimal_lab_input:.1f} GU**")
                st.write(f"To ensure the final product hits the exact target of **{target_line:.1f} GU** on the line, the laboratory should aim for a pre-production mix of **{optimal_lab_input:.1f} GU** to compensate for the process drift.")
                
                # --- CẬP NHẬT CHÚ THÍCH SIGMA TRONG DẢI ICL ---
                st.warning(f"**Internal Control Limit (ICL): {icl_lcl:.1f} - {icl_ucl:.1f}** *(±1σ, với σ = {std_loss:.2f})*")
                st.caption("Production is only authorized if Lab testing falls within this tightened range (±1σ).")

            st.markdown("---")
            st.subheader("🔔 Bias Distribution Shift (Lab vs. Line)")
            st.caption("Illustrates the systematic offset ('Absolute Bias') between the Assigned Value (Lab) and Achieved Value (Line).")

            fig_bell, ax_bell = plt.subplots(figsize=(12, 5))
            
            mean_lab_hist = batch_analysis['Gloss_Lab'].mean()
            std_lab_hist = batch_analysis['Gloss_Lab'].std() if batch_analysis['Gloss_Lab'].std() > 0 else 0.5
            mean_line_hist = batch_analysis['Online_Gloss_Top'].mean()
            std_line_hist = batch_analysis['Online_Gloss_Top'].std() if batch_analysis['Online_Gloss_Top'].std() > 0 else 0.5

            x_min_bell = min(mean_lab_hist - 4*std_lab_hist, mean_line_hist - 4*std_line_hist)
            x_max_bell = max(mean_lab_hist + 4*std_lab_hist, mean_line_hist + 4*std_line_hist)
            x_axis_bell = np.linspace(x_min_bell, x_max_bell, 300)

            y_lab_bell = stats.norm.pdf(x_axis_bell, mean_lab_hist, std_lab_hist)
            ax_bell.plot(x_axis_bell, y_lab_bell, color='#27ae60', lw=2.5, label=f'Lab Input (Assigned Value)\nMean: {mean_lab_hist:.1f} | σ: {std_lab_hist:.2f}')
            ax_bell.fill_between(x_axis_bell, y_lab_bell, alpha=0.1, color='#27ae60')
            ax_bell.axvline(mean_lab_hist, color='#27ae60', ls='--', lw=1.5)

            y_line_bell = stats.norm.pdf(x_axis_bell, mean_line_hist, std_line_hist)
            ax_bell.plot(x_axis_bell, y_line_bell, color='#c0392b', lw=2.5, label=f'Line Output (Achieved Value)\nMean: {mean_line_hist:.1f} | σ: {std_line_hist:.2f}')
            ax_bell.fill_between(x_axis_bell, y_line_bell, alpha=0.1, color='#c0392b')
            ax_bell.axvline(mean_line_hist, color='#c0392b', ls='--', lw=1.5)

            y_annotate = max(max(y_lab_bell), max(y_line_bell)) * 0.6
            ax_bell.annotate('', xy=(mean_lab_hist, y_annotate), xytext=(mean_line_hist, y_annotate),
                              arrowprops=dict(arrowstyle='<|-|>', color='#f39c12', lw=2.5, mutation_scale=15))
            ax_bell.text((mean_lab_hist + mean_line_hist)/2, y_annotate + 0.02, f'Absolute Bias: {mean_loss:+.2f} GU', 
                          ha='center', va='bottom', color='#f39c12', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            ax_bell.set_xlabel("Gloss Value (GU)")
            ax_bell.set_ylabel("Probability Density")
            ax_bell.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig_bell)
            plt.close(fig_bell)

            st.markdown("---")
            st.subheader("📊 Systematic Drift Pattern (Lab vs. Line)")
            
            fig_model, ax_model = plt.subplots(figsize=(14, 6))
            
            batch_labels = batch_analysis['Batch_Lot'].astype(str)
            ax_model.plot(batch_labels, batch_analysis['Gloss_Lab'], marker='o', ls='--', color='gray', alpha=0.6, label='Actual Lab Input')
            ax_model.plot(batch_labels, batch_analysis['Online_Gloss_Top'], marker='s', color='#2980b9', lw=2, label='Actual Line Output (Avg)')
            
            ax_model.axhline(target_line, color='red', ls='-', lw=2.5, label=f'Line Spec Target ({target_line:.1f})')
            ax_model.axhline(line_usl, color='red', ls='--', lw=1.5, alpha=0.5, label=f'Line USL ({line_usl:.1f})')
            ax_model.axhline(line_lsl, color='red', ls='--', lw=1.5, alpha=0.5, label=f'Line LSL ({line_lsl:.1f})')
            
            ax_model.axhline(optimal_lab_input, color='#27ae60', ls='-', lw=2.5, label=f'Optimal Lab Input ({optimal_lab_input:.1f})')
            
            ax_model.set_ylabel("Gloss Value (GU)")
            ax_model.set_xlabel("Batch Sequence")
            plt.xticks(rotation=45, ha='right')
            locs, labels = plt.xticks()
            if len(locs) > 30:
                step = max(1, len(locs) // 20) 
                for i, label in enumerate(labels):
                    if i % step != 0: label.set_visible(False)
            ax_model.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig_model)
            plt.close(fig_model)

            with st.expander("View Systematic Bias Data Details"):
                st.dataframe(batch_analysis[['Batch_Lot', 'Gloss_Lab', 'Online_Gloss_Top', 'Loss']].tail(10))

        else:
            st.warning("⚠️ Insufficient historical data for this paint code to build a reliable compensation model (Min. 5 coils required).")
# ==========================================
# ==========================================
# VIEW 5: SUPPLIER CAPABILITY
# ==========================================
elif view_mode == "🤝 Supplier Capability":
    st.info("💡 Flexible Mode: Select a specific code for capability comparison (Cpk). Select 'All' to evaluate overall stability of a color group based on Target Deviation (ΔGloss).")
    
    st.markdown("---")
    st.subheader("🚨 Negotiation Radar (Supplier Blacklist)")
    st.caption("Paint codes with **Cpk < 1.0** (unstable gloss) or **ΔE Max > 1.0** (color shift) will be flagged here. Minimum 2 Batches required.")
    
    dff_radar = dff.dropna(subset=['Online_Gloss_Top', 'Supplier', 'Gloss_LSL', 'Gloss_USL', 'Color_Group', 'Color_Code', 'dL_N', 'da_N', 'db_N'])
    dff_radar = dff_radar[(dff_radar['Gloss_LSL'] > 0) & (dff_radar['Gloss_USL'] > 0) & (dff_radar['Online_Gloss_Top'] > 0)]
    
    if not dff_radar.empty:
        # TÍNH TOÁN THEO SỐ MẺ (BATCH)
        radar_summary = dff_radar.groupby(['Color_Group', 'Color_Code', 'Supplier']).agg(
            Batches=('Batch_Lot', 'nunique'),
            Coils=('Online_Gloss_Top', 'count'),
            LSL=('Gloss_LSL', 'first'), USL=('Gloss_USL', 'first'),
            Mean_Line=('Online_Gloss_Top', 'mean'), Std_Line=('Online_Gloss_Top', 'std'), dE_Max=('ΔE', 'max')
        ).reset_index()
        
        # Chỉ đánh giá các nhà cung cấp đã giao từ 2 mẻ (Batch) trở lên
        radar_summary = radar_summary[radar_summary['Batches'] >= 2]
        
        def calc_cpk_radar(row):
            if pd.isna(row['Std_Line']) or row['Std_Line'] == 0: return np.nan
            return min((row['USL'] - row['Mean_Line']) / (3 * row['Std_Line']), (row['Mean_Line'] - row['LSL']) / (3 * row['Std_Line']))
            
        radar_summary['Cpk (Line)'] = radar_summary.apply(calc_cpk_radar, axis=1)
        radar_alert = radar_summary[(radar_summary['Cpk (Line)'] < 1.0) | (radar_summary['dE_Max'] > 1.0)].copy()
        
        if not radar_alert.empty:
            radar_alert = radar_alert.sort_values(by=['Cpk (Line)'], ascending=True)
            radar_alert = radar_alert[['Color_Group', 'Color_Code', 'Supplier', 'Batches', 'Coils', 'LSL', 'USL', 'Mean_Line', 'Std_Line', 'dE_Max', 'Cpk (Line)']]
            radar_alert.columns = ['Color Group', 'Color Code (4 Digits)', 'Supplier', 'Batches', 'Coils', 'LSL', 'USL', 'Mean (Line)', 'Std (Line)', 'Max ΔE', 'Cpk (Line)']
            st.dataframe(
                radar_alert.style.format({
                    'LSL': '{:.0f}', 'USL': '{:.0f}', 'Mean (Line)': '{:.1f}',
                    'Std (Line)': '{:.2f}', 'Max ΔE': '{:.2f}', 'Cpk (Line)': '{:.2f}'
                }).background_gradient(cmap='Reds_r', subset=['Cpk (Line)']).background_gradient(cmap='Reds', subset=['Max ΔE']),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("🎉 Excellent! No paint codes are currently in critical violation of Cpk or Color limits.")
    
    st.markdown("---")
    st.write("### 🔍 Drill-down Analysis")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        list_color_group = sorted(dff['Color_Group'].dropna().unique().tolist())
        if list_color_group:
            sel_color_group = st.selectbox("🎨 Step 1: Select Color Group:", list_color_group)
        else:
            st.warning("No color group data available.")
            sel_color_group = None
            
    with col_f2:
        if sel_color_group:
            dff_nhom = dff[dff['Color_Group'] == sel_color_group].copy()
            list_code_4digits = ['All'] + sorted(dff_nhom['Color_Code'].dropna().unique().tolist())
            sel_code_4digits = st.selectbox("🔢 Step 2: Select Color Code (Last 4 digits):", list_code_4digits)
        else:
            sel_code_4digits = None
            
    if sel_code_4digits:
        if sel_code_4digits == 'All':
            dff_comp = dff_nhom.copy()
            title_suffix = f"Group: {sel_color_group} (All codes)"
            is_mixed = True
        else:
            dff_comp = dff_nhom[dff_nhom['Color_Code'] == sel_code_4digits].copy()
            title_suffix = f"Code: {sel_code_4digits}"
            is_mixed = False
        
        dff_comp = dff_comp.dropna(subset=['Online_Gloss_Top', 'Supplier', 'Gloss_LSL', 'Gloss_USL', 'dL_N', 'da_N', 'db_N'])
        dff_comp = dff_comp[(dff_comp['Gloss_LSL'] > 0) & (dff_comp['Gloss_USL'] > 0) & (dff_comp['Online_Gloss_Top'] > 0)]
        dff_comp['Gloss_Target'] = (dff_comp['Gloss_LSL'] + dff_comp['Gloss_USL']) / 2
        dff_comp['Gloss_Dev'] = dff_comp['Online_Gloss_Top'] - dff_comp['Gloss_Target']
        
        # Đếm số lượng MẺ SƠN duy nhất theo từng nhà cung cấp
        batch_counts = dff_comp.groupby('Supplier')['Batch_Lot'].nunique()
        # Lọc ra những nhà cung cấp có từ 2 Mẻ sơn trở lên
        valid_suppliers = batch_counts[batch_counts >= 2].index
        dff_comp = dff_comp[dff_comp['Supplier'].isin(valid_suppliers)]
        
        if len(dff_comp['Supplier'].unique()) >= 1:
            st.markdown("---")
            c1, c2 = st.columns([2, 2.2]) 
            plot_col = 'Gloss_Dev' if is_mixed else 'Online_Gloss_Top'
            plot_ylabel = "Deviation from Target (ΔGloss)" if is_mixed else "Online Gloss (Line Gloss)"
            
            with c1:
                st.subheader(f"📊 Line Gloss Dispersion ({title_suffix})")
                fig_comp1, ax_comp1 = plt.subplots(figsize=(10, 5))
                num_sups = dff_comp['Supplier'].nunique()
                
                if is_mixed:
                    b_width = 0.5 if num_sups > 1 else 0.3
                    sns.boxplot(data=dff_comp, x='Supplier', y=plot_col, palette='Set2', ax=ax_comp1, showfliers=False, width=b_width)
                    sns.stripplot(data=dff_comp, x='Supplier', y=plot_col, color='black', alpha=0.3, size=3, jitter=True, ax=ax_comp1)
                    ax_comp1.axhline(0, color='green', ls='--', lw=2, label='Target Standard (0)')
                else:
                    b_width = 0.3 if num_sups > 1 else 0.15
                    sns.boxplot(data=dff_comp, x='Supplier', y=plot_col, color='#ecf0f1', ax=ax_comp1, showfliers=False, width=b_width, linewidth=1.5)
                    sns.stripplot(data=dff_comp, x='Supplier', y=plot_col, hue='Supplier', palette='Set1', alpha=0.85, size=7, jitter=0.15, ax=ax_comp1, legend=False)
                    lsl_val = dff_comp['Gloss_LSL'].iloc[0]
                    usl_val = dff_comp['Gloss_USL'].iloc[0]
                    ax_comp1.axhline(lsl_val, color='red', ls='--', lw=2, label=f'LSL ({lsl_val:.0f})')
                    ax_comp1.axhline(usl_val, color='red', ls='--', lw=2, label=f'USL ({usl_val:.0f})')
                    total_mean = dff_comp['Online_Gloss_Top'].mean()
                    ax_comp1.axhline(total_mean, color='gray', ls=':', lw=1.5, label=f'Avg Line ({total_mean:.1f})')
                
                ax_comp1.set_xlabel("Supplier")
                ax_comp1.set_ylabel(plot_ylabel)
                plt.legend()
                st.pyplot(fig_comp1)
                
            with c2:
                if is_mixed:
                    st.subheader("Stability Index Table (Aggregated)")
                    comp_table = dff_comp.groupby('Supplier').agg(
                        Batches=('Batch_Lot', 'nunique'), Coils=('Online_Gloss_Top', 'count'), 
                        Mean_Dev=('Gloss_Dev', 'mean'), Std_Dev=('Gloss_Dev', 'std'), Avg_dE=('ΔE', 'mean')
                    ).reset_index()
                    comp_table = comp_table.sort_values('Std_Dev', ascending=True)
                    comp_table.columns = ['Supplier', 'Batches', 'Coils', 'Avg Target Dev', 'Dispersion (Std)', 'Avg ΔE']
                    st.dataframe(
                        comp_table.style.format({'Avg Target Dev': '{:+.2f}', 'Dispersion (Std)': '{:.2f}', 'Avg ΔE': '{:.2f}'}).background_gradient(cmap='RdYlGn_r', subset=['Dispersion (Std)']), 
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.subheader("Line Capability Table (Online Cpk)")
                    comp_table = dff_comp.groupby('Supplier').agg(
                        Batches=('Batch_Lot', 'nunique'), Coils=('Online_Gloss_Top', 'count'), 
                        LSL=('Gloss_LSL', 'mean'), USL=('Gloss_USL', 'mean'), 
                        Mean_Gloss=('Online_Gloss_Top', 'mean'), Std_Gloss=('Online_Gloss_Top', 'std'), Avg_dE=('ΔE', 'mean')
                    ).reset_index()
                    def calc_cpk(row):
                        if pd.isna(row['Std_Gloss']) or row['Std_Gloss'] == 0: return np.nan
                        return min((row['USL'] - row['Mean_Gloss']) / (3 * row['Std_Gloss']), (row['Mean_Gloss'] - row['LSL']) / (3 * row['Std_Gloss']))
                    comp_table['Cpk (Line)'] = comp_table.apply(calc_cpk, axis=1)
                    comp_table = comp_table.sort_values('Cpk (Line)', ascending=False)
                    comp_table.columns = ['Supplier', 'Batches', 'Coils', 'LSL', 'USL', 'Mean (Line)', 'Std (Line)', 'Avg ΔE', 'Cpk (Line)']
                    st.dataframe(
                        comp_table.style.format({'LSL': '{:.0f}', 'USL': '{:.0f}', 'Mean (Line)': '{:.1f}', 'Std (Line)': '{:.2f}', 'Avg ΔE': '{:.2f}', 'Cpk (Line)': '{:.2f}'}).background_gradient(cmap='RdYlGn', subset=['Cpk (Line)']), 
                        use_container_width=True, hide_index=True
                    )

            st.markdown("---")
            st.subheader("📉 Batch-to-Batch Gloss Variation")
            st.caption("Detailed view of the specific batches with the highest and lowest average gloss. Exposes exact lot numbers for vendor accountability.")
            
            batch_means = dff_comp.groupby(['Supplier', 'Batch_Lot']).agg(Mean_Line=('Online_Gloss_Top', 'mean'), LSL=('Gloss_LSL', 'first'), USL=('Gloss_USL', 'first')).reset_index()
            
            b2b_records = []
            for sup, group in batch_means.groupby('Supplier'):
                if len(group) >= 2:
                    idx_min = group['Mean_Line'].idxmin()
                    idx_max = group['Mean_Line'].idxmax()
                    min_row = group.loc[idx_min]
                    max_row = group.loc[idx_max]
                    b2b_records.append({
                        'Supplier': sup, 'Batches': len(group), 'LSL': min_row['LSL'], 'USL': min_row['USL'],
                        'Min Batch': min_row['Batch_Lot'], 'Min Avg': min_row['Mean_Line'],
                        'Max Batch': max_row['Batch_Lot'], 'Max Avg': max_row['Mean_Line'],
                        'Gap': max_row['Mean_Line'] - min_row['Mean_Line']
                    })
            
            b2b_table = pd.DataFrame(b2b_records)
            
            if not b2b_table.empty:
                b2b_table = b2b_table.sort_values('Gap', ascending=False)
                st.dataframe(
                    b2b_table.style.format({'LSL': '{:.0f}', 'USL': '{:.0f}', 'Min Avg': '{:.1f}', 'Max Avg': '{:.1f}', 'Gap': '{:.1f}'}).background_gradient(cmap='Oranges', subset=['Gap']), 
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough multi-batch data to compare Batch-to-Batch gloss variation.")

            st.markdown("---")
            st.subheader("🎨 Batch Color Drift Detailed Analysis")
            st.caption("Table displays AVERAGE values of color components (ΔL, Δa, Δb) per batch. The **Full Paint Code** column helps Vendors trace exact formulas.")
            
            color_drift = dff_comp.groupby(['Supplier', 'Batch_Lot']).agg(
                Full_Paint_Code=('Ma_Son', 'first'), Ngay_SX=('Ngay_SX', 'min'),
                Mean_dL=('dL_N', 'mean'), Mean_da=('da_N', 'mean'), Mean_db=('db_N', 'mean'), Max_dE=('ΔE', 'max')
            ).reset_index()
            
            color_drift = color_drift.sort_values(by=['Supplier', 'Ngay_SX'])
            color_drift.columns = ['Supplier', 'Batch Lot', 'Full Paint Code', 'Production Date', 'ΔL (Avg)', 'Δa (Avg)', 'Δb (Avg)', 'Max ΔE']
            
            c_drift1, c_drift2 = st.columns(2)
            with c_drift1:
                drift_suppliers = ['All Suppliers'] + sorted(color_drift['Supplier'].unique().tolist())
                sel_drift_sup = st.selectbox("🏭 Select Supplier:", drift_suppliers, key="drift_sup_filter")
            if sel_drift_sup != 'All Suppliers':
                color_drift = color_drift[color_drift['Supplier'] == sel_drift_sup]
            with c_drift2:
                drift_codes = ['All Paint Codes'] + sorted(color_drift['Full Paint Code'].unique().tolist())
                sel_drift_code = st.selectbox("🎯 Select Full Paint Code:", drift_codes, key="drift_code_filter")
            if sel_drift_code != 'All Paint Codes':
                color_drift = color_drift[color_drift['Full Paint Code'] == sel_drift_code]
            
            st.dataframe(
                color_drift.style.format({'ΔL (Avg)': '{:+.2f}', 'Δa (Avg)': '{:+.2f}', 'Δb (Avg)': '{:+.2f}', 'Max ΔE': '{:.2f}'})
                .background_gradient(cmap='bwr', subset=['ΔL (Avg)'], vmin=-0.5, vmax=0.5) 
                .background_gradient(cmap='bwr', subset=['Δa (Avg)'], vmin=-0.3, vmax=0.3) 
                .background_gradient(cmap='bwr', subset=['Δb (Avg)'], vmin=-0.3, vmax=0.3) 
                .background_gradient(cmap='Reds', subset=['Max ΔE'], vmin=0, vmax=1.0),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("⚠️ Insufficient data (needs at least 2 batches per supplier with Line data) to perform comparison.")

# ==========================================
# ==========================================
# VIEW 6: MASTER SUMMARY REPORT
# ==========================================
elif view_mode == "📋 Master Summary Report":
    st.info("Master Data table, grouped by Resin Type, Color Code, and Supplier.")
    
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
        'Resin Type', 'Color Code', 'Supplier', 'Coils', 
        'Avg Gloss(Lab)', 'Std(Gloss)', 'Min Gloss', 'Max Gloss', 
        'Avg Online(Top)', 'LSL', 'USL', 'Avg ΔE', 'Yield (%)'
    ]

    st.dataframe(
        summary_table.style.format({
            'Avg Gloss(Lab)': '{:.1f}', 'Std(Gloss)': '{:.2f}', 'Min Gloss': '{:.1f}', 'Max Gloss': '{:.1f}',
            'Avg Online(Top)': '{:.1f}', 'LSL': '{:.0f}', 'USL': '{:.0f}', 'Avg ΔE': '{:.2f}', 'Yield (%)': '{:.1f}%'
        }).background_gradient(cmap='RdYlGn_r', subset=['Std(Gloss)'])
          .background_gradient(cmap='RdYlGn', subset=['Yield (%)'], low=0, high=100),
        use_container_width=True, hide_index=True
    )

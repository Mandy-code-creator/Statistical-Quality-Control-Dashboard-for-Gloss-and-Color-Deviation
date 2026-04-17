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
            "📊 Statistical Limits (Scope Comparison)",
            "⚖️ Predictive Compensation & Targeting", 
            "🤝 Supplier Capability",
            "🎨 Color Shift Analysis",
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
# VIEW 1: GLOSS TREND (SPC)
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

        # ── BIỂU ĐỒ 1: TREND (ZIGZAG RAW DATA) ───────────────────────────────────
        fig_trend, ax_trend = plt.subplots(figsize=(14, 4.5))
        dff_g['x_seq'] = list(range(len(dff_g)))
        
        ax_trend.plot(dff_g['x_seq'], dff_g['Gloss_Lab'], marker='o', color='#1f77b4', lw=1.5, label='Lab Gloss')
        ax_trend.plot(dff_g['x_seq'], dff_g['Online_Gloss_Top'], marker='s', color='#ff7f0e', lw=1.5, label='Line Gloss')
        
        ng_lab = dff_g[(dff_g['Gloss_Lab'] < lsl_val) | (dff_g['Gloss_Lab'] > usl_val)]
        if not ng_lab.empty:
            ax_trend.scatter(ng_lab['x_seq'], ng_lab['Gloss_Lab'], color='red', s=100, zorder=5, label='Lab NG (Out of Spec)')
            
        ng_line = dff_g[(dff_g['Online_Gloss_Top'] < line_lsl_val) | (dff_g['Online_Gloss_Top'] > line_usl_val)]
        if not ng_line.empty:
            ax_trend.scatter(ng_line['x_seq'], ng_line['Online_Gloss_Top'], color='red', marker='s', s=100, zorder=5, label='Line NG (Out of Spec)')

        ax_trend.axhline(lsl_val, color='red', ls='-', lw=2, label=f'Lab LSL ({lsl_val})')
        ax_trend.axhline(usl_val, color='red', ls='-', lw=2, label=f'Lab USL ({usl_val})')
        ax_trend.axhline(line_lsl_val, color='green', ls='--', lw=2, label=f'Line LSL ({line_lsl_val})')
        ax_trend.axhline(line_usl_val, color='green', ls='--', lw=2, label=f'Line USL ({line_usl_val})')
        
        ax_trend.set_xlabel("Production Sequence (Coils grouped by Batch)")
        
        batch_info = dff_g.groupby('Batch_Lot', sort=False)['x_seq'].agg(['min', 'max', 'mean']).reset_index()

        for val in batch_info['min']:
            if val > 0:
                ax_trend.axvline(x=val - 0.5, color='gray', linestyle=':', lw=1.5, alpha=0.5)

        min_x_distance = max(1, len(dff_g) / 30.0) 
        kept_ticks = []
        kept_labels = []
        last_tick = -999
        
        for idx, row in batch_info.iterrows():
            if (row['mean'] - last_tick) >= min_x_distance:
                kept_ticks.append(row['mean'])
                kept_labels.append(str(row['Batch_Lot']))
                last_tick = row['mean']
                
        if kept_ticks and kept_ticks[-1] != batch_info['mean'].iloc[-1]:
            if (batch_info['mean'].iloc[-1] - kept_ticks[-1]) < min_x_distance:
                kept_ticks.pop() 
                kept_labels.pop()
            kept_ticks.append(batch_info['mean'].iloc[-1])
            kept_labels.append(str(batch_info['Batch_Lot'].iloc[-1]))

        ax_trend.set_xticks(kept_ticks)
        ax_trend.set_xticklabels(kept_labels, rotation=45, ha='right', fontsize=8)
        
        ax_trend.set_ylabel("Gloss (GU)")
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

        # ── BIỂU ĐỒ 2: GLOSS DISTRIBUTION ─────────────────────────
        st.write("**Gloss Distribution Analysis**")
        fig_dist, ax_dist = plt.subplots(figsize=(9, 5)) 
        
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

        add_compact_label(lsl_val, "Lab LSL", "red", 0)
        add_compact_label(usl_val, "Lab USL", "red", 0)
        add_compact_label(line_lsl_val, "Line LSL", "green", 0.06)
        add_compact_label(line_usl_val, "Line USL", "green", 0.06)
        
        add_compact_label(mean_lab, "Lab", "#1f77b4", 0.15, std_lab)
        add_compact_label(mean_line, "Line", "#ff7f0e", -0.15, std_line)

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

    # ── TAB SELECTION ──────────────────────────
    list_ma_son_tab2 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab2:
        tab_top_risk, tab_custom, tab_resin = st.tabs(["🚨 Top At-Risk Codes", "🔍 Manual Analysis", "🧪 Resin Comparison"])
        
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
            
        # --- TAB SO SÁNH NHỰA THEO LOGIC MỚI: CÙNG NHÓM MÀU + CÙNG DẢI SPEC ---
        with tab_resin:
            st.subheader("🧪 Resin Comparison Analysis (Apples-to-Apples)")
            st.caption("Auto-filtered: Comparing different Resin types within the SAME **Color Group** and SAME **Gloss Spec Range**.")
            
            # 1. Gom nhóm theo Nhóm Màu và Dải Spec chuẩn xác (Lab & Line)
            dff_resin_base = dff.dropna(subset=['Color_Group', 'Gloss_LSL', 'Gloss_USL', 'Coating_Type']).copy()
            
            # Tạo chuỗi nhận diện theo logic Mandy: "Black Group (Lab: 20~30 | Line: 18~32)"
            dff_resin_base['Group_Spec'] = (
                dff_resin_base['Color_Group'] + " Group (Lab: " + 
                dff_resin_base['Gloss_LSL'].apply(lambda x: f"{x:g}") + "~" + 
                dff_resin_base['Gloss_USL'].apply(lambda x: f"{x:g}") + " | Line: " +
                (dff_resin_base['Gloss_LSL'] - 2.0).apply(lambda x: f"{x:g}") + "~" +
                (dff_resin_base['Gloss_USL'] + 2.0).apply(lambda x: f"{x:g}") + ")"
            )
            
            # 2. Bộ đếm: Chỉ lấy những Group có từ 2 loại nhựa trở lên
            resin_counts = dff_resin_base.groupby('Group_Spec')['Coating_Type'].nunique()
            valid_counts = resin_counts[resin_counts >= 2]
            
            if not valid_counts.empty:
                display_options = {spec: f"{spec} ➡️ [{count} Resins]" for spec, count in valid_counts.items()}
                sorted_display_list = sorted(display_options.values())
                
                sel_display_text = st.selectbox("🎯 Select Color Group & Standard Spec:", sorted_display_list)
                sel_group_spec = [k for k, v in display_options.items() if v == sel_display_text][0]
                
                # 3. Lọc dữ liệu
                df_resin_subset = dff_resin_base[dff_resin_base['Group_Spec'] == sel_group_spec].copy()
                available_resins = sorted(df_resin_subset['Coating_Type'].unique().tolist())
                
                st.success(f"Comparing **{len(available_resins)}** resin types: {', '.join(available_resins)}")
                
                # --- THIẾT KẾ DUAL-PLOT ---
                fig_resin, (ax_dist, ax_box) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
                palette = sns.color_palette("tab10", len(available_resins))
                
                # Lấy các thông số Spec để vẽ
                lsl_val = df_resin_subset['Gloss_LSL'].iloc[0] - 2.0
                usl_val = df_resin_subset['Gloss_USL'].iloc[0] + 2.0
                target_val = (lsl_val + usl_val) / 2.0
                
                # Kẻ vạch Target & Limits xuyên suốt cả 2 biểu đồ
                for ax in [ax_dist, ax_box]:
                    ax.axvline(target_val, color='black', linestyle='-', lw=2.5, label=f'Target ({target_val:.1f})' if ax == ax_dist else "")
                    ax.axvline(lsl_val, color='red', linestyle='--', lw=1.5, alpha=0.6, label='Line Spec Limits' if ax == ax_dist else "")
                    ax.axvline(usl_val, color='red', linestyle='--', lw=1.5, alpha=0.6)

                for idx, resin in enumerate(available_resins):
                    r_data = df_resin_subset[df_resin_subset['Coating_Type'] == resin]['Online_Gloss_Top'].dropna()
                    
                    if len(r_data) > 1:
                        r_mean = r_data.mean()
                        r_std = r_data.std() if r_data.std() > 0 else 0.1
                        r_min = r_data.min()
                        r_max = r_data.max()
                        
                        # 1. Vẽ Normal Curve ở tầng trên (ax_dist)
                        x_min, x_max = r_data.min() - 3*r_std, r_data.max() + 3*r_std
                        x_axis = np.linspace(x_min, x_max, 200)
                        y_curve = stats.norm.pdf(x_axis, r_mean, r_std)
                        
                        ax_dist.plot(x_axis, y_curve, color=palette[idx], lw=2.5, label=f'{resin} (N={len(r_data)})')
                        ax_dist.fill_between(x_axis, y_curve, alpha=0.2, color=palette[idx])
                        
                        # 2. Vẽ dải Statistical Range ở tầng dưới (ax_box)
                        ctrl_lower = r_mean - 3*r_std
                        ctrl_upper = r_mean + 3*r_std
                        
                        # Đường dày: Control Limits (+- 3 sigma)
                        ax_box.plot([ctrl_lower, ctrl_upper], [idx, idx], color=palette[idx], lw=8, alpha=0.4, solid_capstyle='round', label='Control Limits (±3σ)' if idx == 0 else "")
                        # Đường nét đứt mỏng: Min đến Max thực tế
                        ax_box.plot([r_min, r_max], [idx, idx], color='gray', lw=1.5, linestyle='--', zorder=1, label='Min-Max Spread' if idx == 0 else "")
                        # Chấm tròn: Mean
                        ax_box.scatter(r_mean, idx, color=palette[idx], edgecolor='white', s=120, zorder=3, label='Mean' if idx == 0 else "")
                        # Vạch thẳng đứng: Min và Max
                        ax_box.scatter([r_min, r_max], [idx, idx], color='black', marker='|', s=100, zorder=4)
                    
                    elif len(r_data) == 1:
                        ax_dist.axvline(r_data.iloc[0], color=palette[idx], linestyle=':', lw=2.5, label=f'{resin} (N=1)')
                        ax_box.scatter(r_data.iloc[0], idx, color=palette[idx], edgecolor='white', s=120, zorder=3)
                
                # Định dạng hiển thị
                ax_dist.set_title(f"Gloss Capability Comparison: {sel_group_spec}", fontweight='bold')
                ax_dist.set_ylabel("Probability Density")
                ax_dist.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                ax_dist.grid(axis='x', alpha=0.3)
                
                ax_box.set_yticks(range(len(available_resins)))
                ax_box.set_yticklabels(available_resins, fontweight='bold', fontsize=10)
                ax_box.set_xlabel("Online Line Gloss (GU)", fontweight='bold')
                ax_box.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                ax_box.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_resin)
                plt.close(fig_resin)
                
                # Bảng tính điểm Cpk tự động
                with st.expander("🔍 View Capability Statistics (Numerical Details)", expanded=True):
                    stats_df = df_resin_subset.groupby('Coating_Type').agg(
                        Batches=('Batch_Lot', 'nunique'),
                        Coils=('Online_Gloss_Top', 'count'),
                        Mean=('Online_Gloss_Top', 'mean'),
                        Min=('Online_Gloss_Top', 'min'),
                        Max=('Online_Gloss_Top', 'max'),
                        Std=('Online_Gloss_Top', 'std')
                    ).reset_index()
                    
                    def calc_cpk_resin(row):
                        if pd.isna(row['Std']) or row['Std'] == 0: return np.nan
                        cpk = min((usl_val - row['Mean']) / (3 * row['Std']), (row['Mean'] - lsl_val) / (3 * row['Std']))
                        return max(0, cpk)
                    
                    stats_df['Cpk (Line)'] = stats_df.apply(calc_cpk_resin, axis=1)
                    
                    st.dataframe(stats_df.style.format({
                        'Mean': '{:.1f}', 'Min': '{:.1f}', 'Max': '{:.1f}', 'Std': '{:.2f}', 'Cpk (Line)': '{:.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Cpk (Line)']), use_container_width=True, hide_index=True)
            else:
                st.info("🚫 Hiện tại không có Nhóm màu nào sử dụng chung từ 2 loại nhựa trở lên cùng một dải Spec.")
            # --- ADD-ON: PHÂN TÍCH ĐỘ NHẠY MÀU SẮC (DELTA E) GIỮA CÁC LOẠI NHỰA ---
                    st.markdown("---")
                    st.markdown("#### 🎨 Color Stability Comparison (ΔE Dispersion)")
                    st.caption("Đánh giá xem loại nhựa nào giữ màu ổn định nhất qua lò sấy (ΔE càng thấp và ít dao động càng tốt).")
                    
                    fig_color, ax_color = plt.subplots(figsize=(12, 4))
                    
                    # Vẽ Boxplot đọ sức ΔE
                    sns.boxplot(
                        data=df_resin_subset, 
                        x='ΔE', 
                        y='Coating_Type', 
                        palette="tab10", # Dùng chung hệ màu với biểu đồ Gloss ở trên
                        linewidth=1.5,
                        flierprops={"marker": "x", "color": "red", "s": 40},
                        ax=ax_color
                    )
                    
                    # Vẽ ranh giới NG của màu (Thường ΔE > 1.0 là Fail)
                    ax_color.axvline(1.0, color='red', linestyle='--', lw=2, label='Critical Spec (ΔE = 1.0)')
                    ax_color.axvline(0.8, color='orange', linestyle=':', lw=1.5, label='Warning Limit (ΔE = 0.8)')
                    
                    ax_color.set_xlabel("Total Color Difference (ΔE)")
                    ax_color.set_ylabel("Resin Type")
                    ax_color.legend(loc='upper right')
                    ax_color.grid(axis='x', alpha=0.3)
                    
                    st.pyplot(fig_color)
                    plt.close(fig_color)
# ==========================================
# ==========================================
# ==========================================
# VIEW 2: STATISTICAL LIMITS (SCOPE COMPARISON)
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
            
    dff_spc = dff[dff['Ma_Son'] == sel_code].copy().dropna(subset=['Online_Gloss_Top']).sort_values('Ngay_SX')
    
    if len(dff_spc) >= 5:
        line_lsl = dff_spc['Line_LSL'].iloc[0]
        line_usl = dff_spc['Line_USL'].iloc[0]

        st.markdown("---")
        st.subheader("⚙️ Parameter Settings (K-Factor & Sigma)")
        
        col_k, col_mill, col_rel = st.columns(3)
        with col_k:
            k_factor = st.number_input("📏 IQR K-Factor (Outlier Filter)", min_value=0.5, max_value=4.0, value=1.5, step=0.1, format="%.1f", help="Standard is 1.5.")
        with col_mill:
            sigma_mill = st.number_input("🏭 Mill Range (Sigma)", min_value=0.5, max_value=3.0, value=1.0, step=0.1, format="%.1f", help="Internal control limit.")
        with col_rel:
            sigma_release = st.number_input("📦 Release Range (Sigma)", min_value=1.0, max_value=4.0, value=2.0, step=0.1, format="%.1f", help="External delivery control limit.")

        q1 = dff_spc['Online_Gloss_Top'].quantile(0.25)
        q3 = dff_spc['Online_Gloss_Top'].quantile(0.75)
        iqr = q3 - q1
        lower_limit_iqr = q1 - k_factor * iqr
        upper_limit_iqr = q3 + k_factor * iqr
        
        clean_data = dff_spc[(dff_spc['Online_Gloss_Top'] >= lower_limit_iqr) & (dff_spc['Online_Gloss_Top'] <= upper_limit_iqr)]
        outliers = dff_spc[(dff_spc['Online_Gloss_Top'] < lower_limit_iqr) | (dff_spc['Online_Gloss_Top'] > upper_limit_iqr)]
        
        mean_clean = clean_data['Online_Gloss_Top'].mean()
        std_clean = clean_data['Online_Gloss_Top'].std()

        lcl_mill = mean_clean - sigma_mill * std_clean
        ucl_mill = mean_clean + sigma_mill * std_clean
        
        lcl_release = mean_clean - sigma_release * std_clean
        ucl_release = mean_clean + sigma_release * std_clean

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

        st.markdown("---")
        st.subheader("📈 SPC Trending: Outliers, Mill & Release Ranges")
        fig_trend, ax_trend = plt.subplots(figsize=(14, 5))

        seq_index = range(len(dff_spc))
        
        clean_idx = clean_data.index.map(lambda x: dff_spc.index.get_loc(x))
        outlier_idx = outliers.index.map(lambda x: dff_spc.index.get_loc(x))
        
        ax_trend.plot(seq_index, dff_spc['Online_Gloss_Top'], color='gray', alpha=0.3, ls='-') 
        ax_trend.scatter(clean_idx, clean_data['Online_Gloss_Top'], color='#3498db', label='Clean Coils', zorder=3)
        if not outliers.empty:
            ax_trend.scatter(outlier_idx, outliers['Online_Gloss_Top'], color='red', marker='x', s=60, lw=2, label='Filtered Outliers (IQR)', zorder=4)

        ax_trend.axhline(mean_clean, color='black', lw=2, label=f'Mean ({mean_clean:.1f})')
        ax_trend.axhline(ucl_mill, color='#27ae60', ls='--', lw=2, label=f'Mill UCL ({ucl_mill:.1f})')
        ax_trend.axhline(lcl_mill, color='#27ae60', ls='--', lw=2, label=f'Mill LCL ({lcl_mill:.1f})')
        ax_trend.axhline(ucl_release, color='#e67e22', ls='-.', lw=2, label=f'Release UCL ({ucl_release:.1f})')
        ax_trend.axhline(lcl_release, color='#e67e22', ls='-.', lw=2, label=f'Release LCL ({lcl_release:.1f})')
        ax_trend.axhline(line_usl, color='red', ls='-', lw=1.5, alpha=0.5, label='Spec USL')
        ax_trend.axhline(line_lsl, color='red', ls='-', lw=1.5, alpha=0.5, label='Spec LSL')

        ax_trend.set_xlabel("Production Sequence")
        ax_trend.set_ylabel("Online Gloss Top (GU)")
        
        plt.xticks(seq_index, dff_spc['Batch_Lot'].astype(str), rotation=45, ha='right')
        if len(seq_index) > 20:
            step = max(1, len(seq_index) // 15)
            for i, label in enumerate(ax_trend.xaxis.get_ticklabels()):
                if i % step != 0: label.set_visible(False)
                
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

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
# VIEW 3: PREDICTIVE COMPENSATION MODEL
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
                st.metric("Historical Process Bias", f"{mean_loss:+.2f} GU", help="Average drift caused by the production line for this specific paint.")
                st.metric("Standard Deviation (Sigma, σ)", f"{std_loss:.2f} GU", help="Calculated variation (σ) of the historical bias. Used to define the Internal Control Limit.")

            with col_guidance:
                st.success(f"#### Recommended Lab Input: **{optimal_lab_input:.1f} GU**")
                st.write(f"To ensure the final product hits the exact target of **{target_line:.1f} GU** on the line, the laboratory should aim for a pre-production mix of **{optimal_lab_input:.1f} GU** to compensate for the process drift.")
                st.warning(f"**Internal Control Limit (ICL): {icl_lcl:.1f} - {icl_ucl:.1f}** *(±1σ, with σ = {std_loss:.2f})*")
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
# VIEW 4: SUPPLIER CAPABILITY BENCHMARKING
# ==========================================
elif view_mode == "🤝 Supplier Capability":
    st.header("🤝 Executive Supplier Benchmarking")
    st.info("💡 Apples-to-Apples Analysis: Evaluates suppliers directly against each other using the same Color Group and Paint Code specification.")

    dff_v5 = dff.dropna(subset=['Online_Gloss_Top', 'Supplier', 'Gloss_LSL', 'Gloss_USL', 'Color_Group', 'Ma_Son', 'dL_N', 'da_N', 'db_N']).copy()
    dff_v5 = dff_v5[(dff_v5['Gloss_LSL'] > 0) & (dff_v5['Gloss_USL'] > 0) & (dff_v5['Online_Gloss_Top'] > 0)]
    
    dff_v5['Gloss_Target'] = (dff_v5['Gloss_LSL'] + dff_v5['Gloss_USL']) / 2.0

    st.write("### 🔍 Comparison Filters")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        list_color_group = sorted(dff_v5['Color_Group'].unique().tolist())
        sel_color_group = st.selectbox("🎨 Step 1: Select Color Group:", list_color_group) if list_color_group else None

    with col_f2:
        if sel_color_group:
            dff_nhom = dff_v5[dff_v5['Color_Group'] == sel_color_group].copy()
            list_ma_son = ['All (View Normalized Deviations)'] + sorted(dff_nhom['Ma_Son'].unique().tolist())
            sel_ma_son = st.selectbox("🎯 Step 2: Select Full Paint Code (Ma_Son):", list_ma_son)
        else:
            sel_ma_son = None

    if sel_ma_son:
        is_mixed = "All" in sel_ma_son
        if is_mixed:
            dff_comp = dff_nhom.copy()
            title_suffix = f"Group: {sel_color_group} (Multiple Specs)"
        else:
            dff_comp = dff_nhom[dff_nhom['Ma_Son'] == sel_ma_son].copy()
            title_suffix = f"Code: {sel_ma_son}"
        
        dff_comp['Gloss_Dev'] = dff_comp['Online_Gloss_Top'] - dff_comp['Gloss_Target']

        batch_counts = dff_comp.groupby('Supplier')['Batch_Lot'].nunique()
        valid_suppliers = batch_counts[batch_counts >= 2].index
        dff_comp = dff_comp[dff_comp['Supplier'].isin(valid_suppliers)]
        
        if not dff_comp.empty:
            st.markdown("---")
            
            comp_table = dff_comp.groupby(['Supplier', 'Ma_Son']).agg(
                Batches=('Batch_Lot', 'nunique'), 
                Coils=('Online_Gloss_Top', 'count'), 
                LSL=('Gloss_LSL', 'first'), 
                USL=('Gloss_USL', 'first'), 
                Mean_Gloss=('Online_Gloss_Top', 'mean'), 
                Std_Gloss=('Online_Gloss_Top', 'std'), 
                Avg_dE=('ΔE', 'mean')
            ).reset_index()
            
            def calc_cpk_table(row):
                if pd.isna(row['Std_Gloss']) or row['Std_Gloss'] == 0: return np.nan
                return min((row['USL'] - row['Mean_Gloss']) / (3 * row['Std_Gloss']), (row['Mean_Gloss'] - row['LSL']) / (3 * row['Std_Gloss']))
            
            comp_table['Cpk (Line)'] = comp_table.apply(calc_cpk_table, axis=1)
            comp_table['Target'] = (comp_table['LSL'] + comp_table['USL']) / 2.0
            comp_table['Bias'] = comp_table['Mean_Gloss'] - comp_table['Target']
            comp_table = comp_table.sort_values(['Supplier', 'Cpk (Line)'], ascending=[True, False])

            c1, c2 = st.columns([2.5, 2.5]) 
            
            with c1:
                st.subheader("📊 Executive Performance Matrix")
                st.caption(title_suffix)
                
                fig_matrix, ax_matrix = plt.subplots(figsize=(9, 6))
                
                max_cpk = max(2.5, comp_table['Cpk (Line)'].max() + 0.2) if not comp_table['Cpk (Line)'].isna().all() else 2.5
                max_bias_abs = max(abs(comp_table['Bias'].max()), abs(comp_table['Bias'].min())) + 1 if not pd.isna(comp_table['Bias'].max()) else 5

                ax_matrix.axhspan(1.33, max_cpk, facecolor='#27ae60', alpha=0.6, label='Excellent (Cpk > 1.33)')
                ax_matrix.axhspan(1.0, 1.33, facecolor='#f1c40f', alpha=0.6, label='Warning (1.0 < Cpk < 1.33)')
                ax_matrix.axhspan(0, 1.0, facecolor='#c0392b', alpha=0.6, label='High Risk (Cpk < 1.0)')
                
                ax_matrix.axvline(0, color='black', linestyle='--', linewidth=2.5, label='Target Center (Bias = 0)')
                ax_matrix.axhline(1.33, color='black', linewidth=1, alpha=0.4)
                ax_matrix.axhline(1.0, color='black', linewidth=1, alpha=0.4)
                
                sns.scatterplot(
                    data=comp_table, x='Bias', y='Cpk (Line)', hue='Supplier', 
                    s=600, edgecolor='white', linewidth=2, palette='tab10', ax=ax_matrix, zorder=5 
                )
                
                import matplotlib.patheffects as path_effects
                for i in range(comp_table.shape[0]):
                    label_text = comp_table['Supplier'].iloc[i]
                    if is_mixed: 
                        label_text += f"\n({comp_table['Ma_Son'].iloc[i][-4:]})"
                    
                    x_pos = comp_table['Bias'].iloc[i]
                    y_pos = comp_table['Cpk (Line)'].iloc[i]
                    
                    if x_pos > (max_bias_abs * 0.7):
                        ha_align, x_offset = 'right', -0.2
                    else:
                        ha_align, x_offset = 'left', 0.2

                    ax_matrix.text(
                        x_pos + x_offset, y_pos + 0.05, label_text, 
                        fontsize=10, fontweight='bold', color='black',
                        horizontalalignment=ha_align, zorder=6,
                        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")]
                    )

                ax_matrix.set_xlabel("Systematic Bias (Average Gloss - Target) [GU]")
                ax_matrix.set_ylabel("Stability Index (Cpk)")
                ax_matrix.set_ylim(0, max_cpk)
                ax_matrix.set_xlim(-max_bias_abs, max_bias_abs)
                ax_matrix.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
                plt.tight_layout()
                st.pyplot(fig_matrix)
                plt.close(fig_matrix)

            with c2:
                st.subheader("Capability Table per Supplier")
                disp_table = comp_table[['Supplier', 'Ma_Son', 'Batches', 'Coils', 'Mean_Gloss', 'Std_Gloss', 'Bias', 'Cpk (Line)', 'Avg_dE']].copy()
                disp_table.columns = ['Supplier', 'Paint Code', 'Batches', 'Coils', 'Mean (Line)', 'Std (Line)', 'Bias', 'Cpk (Line)', 'Avg ΔE']
                
                st.dataframe(
                    disp_table.style.format({
                        'Mean (Line)': '{:.1f}', 'Std (Line)': '{:.2f}', 
                        'Bias': '{:+.2f}', 'Cpk (Line)': '{:.2f}', 'Avg ΔE': '{:.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Cpk (Line)'])
                      .background_gradient(cmap='bwr', subset=['Bias'], vmin=-2, vmax=2), 
                    use_container_width=True, hide_index=True
                )

            st.markdown("---")
            st.subheader("📈 Quality Trend Analysis (Time-Series)")
            st.caption("Visualizing Gloss and Color shifts over sequential production dates. Ideal for spotting erratic supplier behavior or progressive degradation.")
            
            trend_data = dff_comp.groupby(['Supplier', 'Ma_Son', 'Batch_Lot']).agg(
                Prod_Date=('Ngay_SX', 'min'),
                Mean_Gloss=('Online_Gloss_Top', 'mean'),
                Mean_Dev=('Gloss_Dev', 'mean'),
                Max_dE=('ΔE', 'max'),
                LSL=('Gloss_LSL', 'first'),
                USL=('Gloss_USL', 'first')
            ).reset_index().sort_values(by=['Supplier', 'Prod_Date'])
            
            trend_data['Prod_Date'] = pd.to_datetime(trend_data['Prod_Date'])
            trend_data['Timeline_Label'] = trend_data['Prod_Date'].dt.strftime('%Y-%m-%d') + "\n(" + trend_data['Batch_Lot'].astype(str) + ")"

            fig_trend, (ax_gloss, ax_color) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

            if is_mixed:
                trend_data['Trace'] = trend_data['Supplier'] + " (" + trend_data['Ma_Son'].str[-4:] + ")"
                y_col = 'Mean_Dev'
                y_label = "Deviation from Target (ΔGloss)"
                chart_title = "Gloss Deviation Trend (Normalized to 0 Target)"
            else:
                trend_data['Trace'] = trend_data['Supplier']
                y_col = 'Mean_Gloss'
                y_label = "Average Line Gloss (GU)"
                target_val = (trend_data['LSL'].iloc[0] + trend_data['USL'].iloc[0]) / 2.0
                chart_title = f"Absolute Gloss Trend (Target: {target_val:.1f} GU)"

            sns.lineplot(data=trend_data, x='Timeline_Label', y=y_col, hue='Trace', marker='o', markersize=8, lw=2.5, ax=ax_gloss)
            
            if is_mixed:
                ax_gloss.axhline(0, color='green', ls='-', lw=2, label='Target Standard (0)')
            else:
                ax_gloss.axhline(target_val, color='green', ls='-', lw=2, label=f'Target ({target_val})')
                ax_gloss.axhline(trend_data['LSL'].iloc[0], color='red', ls='--', lw=1.5, alpha=0.7, label=f"LSL ({trend_data['LSL'].iloc[0]})")
                ax_gloss.axhline(trend_data['USL'].iloc[0], color='red', ls='--', lw=1.5, alpha=0.7, label=f"USL ({trend_data['USL'].iloc[0]})")
            
            ax_gloss.set_title(chart_title, fontweight='bold')
            ax_gloss.set_ylabel(y_label)
            ax_gloss.set_xlabel("")
            
            ax_gloss.tick_params(axis='x', rotation=45, labelsize=9)
            step_gloss = max(1, len(trend_data) // 12) 
            for ind, label in enumerate(ax_gloss.get_xticklabels()):
                if ind % step_gloss != 0:
                    label.set_visible(False)
                    
            ax_gloss.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            ax_gloss.grid(True, linestyle='--', alpha=0.5)

            sns.lineplot(data=trend_data, x='Timeline_Label', y='Max_dE', hue='Trace', marker='s', markersize=8, lw=2.5, ax=ax_color)
            ax_color.axhline(1.0, color='red', ls='--', lw=2, label='Critical Limit (ΔE = 1.0)')
            ax_color.axhline(0.5, color='#f39c12', ls='--', lw=1.5, label='Warning Limit (ΔE = 0.5)')
            
            ax_color.set_title("Color Consistency (Max ΔE) Over Time", fontweight='bold')
            ax_color.set_ylabel("Max Color Difference (ΔE)")
            ax_color.set_xlabel("Production Date & Batch")
            
            ax_color.tick_params(axis='x', rotation=45, labelsize=9)
            step_color = max(1, len(trend_data) // 12) 
            for ind, label in enumerate(ax_color.get_xticklabels()):
                if ind % step_color != 0:
                    label.set_visible(False)
            
            max_y_color = max(1.2, trend_data['Max_dE'].max() + 0.2) if not pd.isna(trend_data['Max_dE'].max()) else 1.2
            ax_color.set_ylim(0, max_y_color)
            ax_color.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            ax_color.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig_trend)
            plt.close(fig_trend)
            
            with st.expander("🔍 View Raw Batch Data (Details)"):
                st.dataframe(trend_data[['Supplier', 'Ma_Son', 'Batch_Lot', 'Timeline_Label', 'Mean_Gloss', 'Mean_Dev', 'Max_dE']], use_container_width=True, hide_index=True)

        else:
            st.warning("⚠️ Insufficient data (needs at least 2 batches per supplier) to perform comparison.")

# ==========================================
# ==========================================
# VIEW 5: COLOR SHIFT ANALYSIS
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
            st.caption("Fitted with Normal distribution curves to assess the true statistical variation capability.")
            
            col_L, col_a, col_b = st.columns(3)
            
            # Helper function to plot Histogram + Normal Curve
            def plot_norm_hist(data_series, ax, color, title, xlabel):
                data = data_series.dropna()
                if len(data) > 1:
                    mean_val = data.mean()
                    std_val = data.std()
                    # stat='density' aligns the histogram scale with the probability density function (PDF)
                    sns.histplot(data, stat="density", color=color, alpha=0.5, ax=ax, kde=False)
                    
                    if std_val > 0:
                        x_min, x_max = data.min() - 3*std_val, data.max() + 3*std_val
                        x_axis = np.linspace(x_min, x_max, 200)
                        y_curve = stats.norm.pdf(x_axis, mean_val, std_val)
                        ax.plot(x_axis, y_curve, color='black', lw=2.5, label=f'Norm Curve\n(μ={mean_val:.2f}, σ={std_val:.2f})')
                        ax.legend(fontsize=8, loc='upper right')
                else:
                    sns.histplot(data, color=color, ax=ax)
                    
                ax.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density")

            with col_L:
                fig_L, ax_L = plt.subplots(figsize=(5, 4))
                plot_norm_hist(dff_c['dL_N'], ax_L, '#95a5a6', "Lightness (ΔL)", "ΔL (+ Lighter / - Darker)")
                st.pyplot(fig_L)
                
            with col_a:
                fig_a, ax_a = plt.subplots(figsize=(5, 4))
                plot_norm_hist(dff_c['da_N'], ax_a, '#e74c3c', "Red/Green (Δa)", "Δa (+ Redder / - Greener)")
                st.pyplot(fig_a)
                
            with col_b:
                fig_b, ax_b = plt.subplots(figsize=(5, 4))
                plot_norm_hist(dff_c['db_N'], ax_b, '#f1c40f', "Yellow/Blue (Δb)", "Δb (+ Yellower / - Bluer)")
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

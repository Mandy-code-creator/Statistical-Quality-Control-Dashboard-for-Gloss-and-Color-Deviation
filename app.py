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
@st.cache_data(ttl=300) # Increased TTL to 5 minutes to improve performance
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
    # REFRESH DATA BUTTON
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear() 
        st.rerun()            
        
    st.markdown("---")
    
    st.markdown("### 📊 View Mode")
    view_mode = st.radio(
        "Select Analysis View:",
        [
            "✨ Gloss Analysis (SPC)",
            "🎨 Color & ΔE Analysis",
            "📊 Statistical Control Limits",
            "⚖️ Paired Difference Analysis", 
            "🤝 Supplier Comparison",
            "📋 Summary Data Report"
        ],
        label_visibility="collapsed"
    )
    
    # DYNAMIC LIMIT SETTINGS
    st.markdown("---")
    st.markdown("### ⚙️ Limit Settings")
    line_offset = st.number_input("Line Offset (Line = Lab ± X)", value=2.0, step=0.5, help="Allowed deviation between Line and Lab limits")

    with st.expander("🛠️ Custom Gloss Limits", expanded=False):
        st.caption("Enter Paint Code to set custom control limits.")
        
        if "custom_gloss_rules" not in st.session_state:
            init_data = pd.DataFrame({
                "Ma_Son": ["", "", ""],
                "Lab_LSL": [0.0, 0.0, 0.0],
                "Lab_USL": [0.0, 0.0, 0.0],
                "Line_LSL": [0.0, 0.0, 0.0],
                "Line_USL": [0.0, 0.0, 0.0]
            })
            st.session_state["custom_gloss_rules"] = init_data

        edited_rules = st.data_editor(st.session_state["custom_gloss_rules"], num_rows="dynamic", hide_index=True)
        st.session_state["custom_gloss_rules"] = edited_rules

    st.markdown("---")
    st.markdown("### 🔍 Data Filters")
    
    min_date, max_date = df['Ngay_SX'].min(), df['Ngay_SX'].max()
    date_range = st.date_input("📅 Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    list_sup = ['All'] + sorted(df['Supplier'].unique().tolist())
    sel_sup = st.selectbox("🏭 Supplier:", list_sup)
    
    list_res = ['All'] + sorted(df['Coating_Type'].unique().tolist())
    sel_res = st.selectbox("🧪 Coating Type:", list_res)
    
    list_col = ['All'] + sorted(df['Color_Group'].unique().tolist())
    sel_col = st.selectbox("🎨 Color Group:", list_col)

# ==============================================================================
# APPLY LIMITS & CALC PASS/FAIL
# ==============================================================================
df['Line_LSL'] = df['Gloss_LSL'] - line_offset
df['Line_USL'] = df['Gloss_USL'] + line_offset

custom_df = st.session_state["custom_gloss_rules"].dropna(subset=["Ma_Son"])
custom_df = custom_df[custom_df["Ma_Son"].str.strip() != ""]

for _, rule in custom_df.iterrows():
    ma_son = str(rule["Ma_Son"]).strip()
    mask = df["Ma_Son"] == ma_son

    if mask.any():
        if pd.notna(rule["Lab_LSL"]) and rule["Lab_LSL"] > 0:
            df.loc[mask, 'Gloss_LSL'] = float(rule["Lab_LSL"])
        if pd.notna(rule["Lab_USL"]) and rule["Lab_USL"] > 0:
            df.loc[mask, 'Gloss_USL'] = float(rule["Lab_USL"])
        if pd.notna(rule["Line_LSL"]) and rule["Line_LSL"] > 0:
            df.loc[mask, 'Line_LSL'] = float(rule["Line_LSL"])
        if pd.notna(rule["Line_USL"]) and rule["Line_USL"] > 0:
            df.loc[mask, 'Line_USL'] = float(rule["Line_USL"])

df['Lab_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
df['Line_Pass'] = (df['Online_Gloss_Top'] >= df['Line_LSL']) & (df['Online_Gloss_Top'] <= df['Line_USL'])
df['Gloss_Pass'] = df['Lab_Pass'] & df['Line_Pass']
df['Color_Pass'] = df['ΔE'] <= 1.0
df['Final_Status'] = np.where(df['Gloss_Pass'] & df['Color_Pass'], '✅ PASS', '❌ FAIL/NG')

# ==============================================================================
# APPLY FILTERS TO DFF
# ==============================================================================
dff = df.copy()
if len(date_range) == 2:
    dff = dff[(dff['Ngay_SX'] >= date_range[0]) & (dff['Ngay_SX'] <= date_range[1])]
if sel_sup != 'All': dff = dff[dff['Supplier'] == sel_sup]
if sel_res != 'All': dff = dff[dff['Coating_Type'] == sel_res]
if sel_col != 'All': dff = dff[dff['Color_Group'] == sel_col]

with st.sidebar:
    st.markdown("---")
    st.caption(f"📦 Showing: {len(dff)} coils (Invalid codes filtered out)")

# --- 4. DISPLAY VIEWS ---
st.title(view_mode)
st.markdown("---")

# ==========================================
# VIEW 1: GLOSS ANALYSIS (SPC)
# ==========================================
if view_mode == "✨ Gloss Analysis (SPC)":
    st.info("💡 SPC Analysis: Monitor the actual Gloss trend (Lab vs Line) across different production batches to detect process shifts.")
    
    # EARLY WARNING RADAR
    risk_alert = pd.DataFrame()
    with st.expander("🚨 Early Warning Radar (Click to view at-risk codes)", expanded=True):
        st.caption("This table automatically scans data to identify paint codes (≥ 5 Batches produced) that are Out of Spec (NG) or approaching the control limits (Margin ≤ 1.0 GU).")
        
        df_valid_radar = dff.dropna(subset=['Online_Gloss_Top', 'Line_LSL', 'Line_USL', 'Gloss_Lab', 'Gloss_LSL', 'Gloss_USL', 'Batch_Lot'])
        
        if not df_valid_radar.empty:
            risk_summary = df_valid_radar.groupby(['Ma_Son', 'Supplier']).agg(
                Batches=('Batch_Lot', 'nunique'), 
                Coils=('Online_Gloss_Top', 'count'),
                Line_Min=('Online_Gloss_Top', 'min'),
                Line_Max=('Online_Gloss_Top', 'max'),
                Line_LSL=('Line_LSL', 'first'),
                Line_USL=('Line_USL', 'first'),
                Lab_Min=('Gloss_Lab', 'min'),
                Lab_Max=('Gloss_Lab', 'max'),
                Lab_LSL=('Gloss_LSL', 'first'),
                Lab_USL=('Gloss_USL', 'first')
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
                    
                    def highlight_source(val):
                        if 'Lab' in str(val) or 'Line' in str(val): return 'font-weight: bold; color: #d35400;'
                        return ''

                    st.dataframe(
                        risk_alert.style.format({
                            'Lab Min': '{:.1f}', 'Lab Max': '{:.1f}', 'Line Min': '{:.1f}', 'Line Max': '{:.1f}'
                        }).map(highlight_status, subset=['Status'])
                          .map(highlight_source, subset=['Issue Source']),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.success("🎉 No paint codes (with ≥ 5 batches) are out of limits or critically near limits at this time!")
            else:
                st.info("Not enough data. No paint codes have reached the minimum requirement of 5 batches.")

    st.markdown("---")
    
    # REUSABLE SPC RENDER
    def render_spc_analysis(paint_code, data_source, key_suffix):
        dff_g = data_source[data_source['Ma_Son'] == paint_code].copy()
        dff_g = dff_g.dropna(subset=['Gloss_LSL', 'Gloss_USL', 'Gloss_Lab', 'Online_Gloss_Top'])
        dff_g = dff_g[
            (dff_g['Gloss_LSL'] > 0) & 
            (dff_g['Gloss_USL'] > 0) & 
            (dff_g['Gloss_Lab'] > 0) & 
            (dff_g['Online_Gloss_Top'] > 0)
        ]
        
        if len(dff_g) <= 1:
            st.warning(f"⚠️ Insufficient data for {paint_code} (needs at least 2 valid coils) for SPC analysis.")
            return

        lsl_val = dff_g['Gloss_LSL'].iloc[0]
        usl_val = dff_g['Gloss_USL'].iloc[0]
        line_lsl_val = dff_g['Line_LSL'].iloc[0]
        line_usl_val = dff_g['Line_USL'].iloc[0]
        
        mean_lab, std_lab = dff_g['Gloss_Lab'].mean(), dff_g['Gloss_Lab'].std()
        mean_line, std_line = dff_g['Online_Gloss_Top'].mean(), dff_g['Online_Gloss_Top'].std()

        min_date = dff_g['Ngay_SX'].min()
        max_date = dff_g['Ngay_SX'].max()
        batch_count = dff_g['Batch_Lot'].nunique()
        coil_count = len(dff_g)
        
        st.success(f"📅 **Timeframe:** `{min_date}` to `{max_date}` | **Volume:** `{batch_count}` Batches (`{coil_count}` Coils).")

        dff_batch = dff_g.groupby('Batch_Lot', as_index=False).agg({
            'Ngay_SX': 'min',
            'Order_No': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'Line': 'first',
            'Gloss_LSL': 'first',
            'Gloss_USL': 'first',
            'Line_LSL': 'first',
            'Line_USL': 'first',
            'Gloss_Lab': 'first', 
            'Online_Gloss_Top': 'mean' 
        }).sort_values('Ngay_SX')
        
        # USE BATCH LOT ONLY FOR X-AXIS
        dff_batch['Label_X'] = dff_batch['Batch_Lot'].astype(str)
        
        fig_trend, ax_trend = plt.subplots(figsize=(14, 4.5))
        ax_trend.plot(dff_batch['Label_X'], dff_batch['Gloss_Lab'], marker='o', color='#3498db', lw=2, label='Lab Gloss')
        ax_trend.plot(dff_batch['Label_X'], dff_batch['Online_Gloss_Top'], marker='s', color='#e67e22', lw=2, label='Avg Line Gloss')
        
        ax_trend.axhline(lsl_val, color='red', ls='-', lw=2, label=f'Lab LSL ({lsl_val:.1f})')
        ax_trend.axhline(usl_val, color='red', ls='-', lw=2, label=f'Lab USL ({usl_val:.1f})')
        ax_trend.axhline(line_lsl_val, color='orange', ls='--', lw=2, alpha=0.7, label=f'Line LSL ({line_lsl_val:.1f})')
        ax_trend.axhline(line_usl_val, color='orange', ls='--', lw=2, alpha=0.7, label=f'Line USL ({line_usl_val:.1f})')
        
        ax_trend.set_xlabel("Batch Lot")
        ax_trend.set_ylabel("Gloss (GU)")
        
        # ROTATE AND REDUCE DENSITY OF LABELS
        plt.xticks(rotation=45, ha='right')
        locs, labels = plt.xticks()
        
        if len(locs) > 30:
            step = max(1, len(locs) // 20) 
            for i, label in enumerate(labels):
                if i % step != 0: 
                    label.set_visible(False)
        
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

        ng_batches = dff_batch[
            (dff_batch['Gloss_Lab'] < dff_batch['Gloss_LSL']) | 
            (dff_batch['Gloss_Lab'] > dff_batch['Gloss_USL']) | 
            (dff_batch['Online_Gloss_Top'] < dff_batch['Line_LSL']) | 
            (dff_batch['Online_Gloss_Top'] > dff_batch['Line_USL'])
        ].copy()
        
        if not ng_batches.empty:
            st.error(f"🚨 Out of Spec Batches ({len(ng_batches)} Batches)")
            def get_batch_error(row):
                errs = []
                if row['Gloss_Lab'] < row['Gloss_LSL']: errs.append(f"Lab Low (<{row['Gloss_LSL']:.1f})")
                elif row['Gloss_Lab'] > row['Gloss_USL']: errs.append(f"Lab High (>{row['Gloss_USL']:.1f})")
                if row['Online_Gloss_Top'] < row['Line_LSL']: errs.append(f"Line Low (<{row['Line_LSL']:.1f})")
                elif row['Online_Gloss_Top'] > row['Line_USL']: errs.append(f"Line High (>{row['Line_USL']:.1f})")
                return " + ".join(errs)
                
            ng_batches['Error Details'] = ng_batches.apply(get_batch_error, axis=1)
            ng_display = ng_batches[['Batch_Lot', 'Ngay_SX', 'Gloss_Lab', 'Online_Gloss_Top', 'Error Details']].copy()
            ng_display.columns = ['Batch Lot', 'Production Date', 'Lab Gloss', 'Avg Line Gloss', 'Error Details']
            
            st.dataframe(
                ng_display.style.format({
                    'Lab Gloss': '{:.1f}', 'Avg Line Gloss': '{:.1f}'
                }).set_properties(**{
                    'background-color': '#ffebee', 'color': '#c0392b', 'font-weight': 'bold'
                }, subset=['Error Details']),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("✅ All batches meet gloss standards (Lab & Line).")

        st.write("**Gloss Distribution (Histogram & Normal Curve)**")
        fig_g1, ax_g1 = plt.subplots(figsize=(10, 5)) 
        
        min_val = min(dff_g['Gloss_Lab'].min(), dff_g['Online_Gloss_Top'].min())
        max_val = max(dff_g['Gloss_Lab'].max(), dff_g['Online_Gloss_Top'].max())
        if min_val == max_val: 
            min_val -= 1
            max_val += 1
            
        bins_arr = np.linspace(min_val, max_val, 12) 
        bin_width = bins_arr[1] - bins_arr[0]
        
        sns.histplot(dff_g['Gloss_Lab'], stat="count", bins=bins_arr, color='#3498db', alpha=0.4, label='Lab Bins', ax=ax_g1)
        sns.histplot(dff_g['Online_Gloss_Top'], stat="count", bins=bins_arr, color='#e67e22', alpha=0.4, label='Line Bins', ax=ax_g1)
        
        plot_min = min(line_lsl_val, min_val) - 2
        plot_max = max(line_usl_val, max_val) + 2
        x_axis = np.linspace(plot_min, plot_max, 200)
        
        if pd.notna(std_lab) and std_lab > 0:
            y_lab_scaled = stats.norm.pdf(x_axis, mean_lab, std_lab) * len(dff_g['Gloss_Lab']) * bin_width
            ax_g1.plot(x_axis, y_lab_scaled, color='#2980b9', lw=2.5, label=f'Lab Curve')
            
        if pd.notna(std_line) and std_line > 0:
            y_line_scaled = stats.norm.pdf(x_axis, mean_line, std_line) * len(dff_g['Online_Gloss_Top']) * bin_width
            ax_g1.plot(x_axis, y_line_scaled, color='#d35400', lw=2.5, label=f'Line Curve')

        ax_g1.axvline(lsl_val, color='red', ls='-', lw=1.5)
        ax_g1.axvline(usl_val, color='red', ls='-', lw=1.5)
        ax_g1.axvline(line_lsl_val, color='orange', ls='--', lw=1.5, alpha=0.7)
        ax_g1.axvline(line_usl_val, color='orange', ls='--', lw=1.5, alpha=0.7)
        
        ax_g1.set_xlim(plot_min, plot_max)
        ax_g1.set_xlabel("Gloss (GU)")
        ax_g1.set_ylabel("Number of Coils (Count)")
        
        handles, labels = ax_g1.get_legend_handles_labels()
        ax_g1.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=9, frameon=True)
        
        plt.tight_layout() 
        st.pyplot(fig_g1)
        plt.close(fig_g1)

        with st.expander("🔍 View Full Batch Details", expanded=False):
            batch_table = dff_batch[['Batch_Lot', 'Order_No', 'Line', 'Ngay_SX', 'Gloss_LSL', 'Gloss_USL', 'Gloss_Lab', 'Line_LSL', 'Line_USL', 'Online_Gloss_Top']].copy()
            batch_table['Gap (Line - Lab)'] = batch_table['Online_Gloss_Top'] - batch_table['Gloss_Lab']
            batch_table.columns = ['Batch Lot', 'Order Number', 'Line', 'Production Date', 'Lab LSL', 'Lab USL', 'Lab Gloss', 'Line LSL', 'Line USL', 'Avg Line Gloss', 'Gap (Line - Lab)']
            st.dataframe(
                batch_table.style.format({
                    'Lab LSL': '{:.0f}', 'Lab USL': '{:.0f}', 'Line LSL': '{:.0f}', 'Line USL': '{:.0f}',
                    'Lab Gloss': '{:.1f}', 'Avg Line Gloss': '{:.1f}', 'Gap (Line - Lab)': '{:+.1f}'
                }).background_gradient(cmap='RdYlGn_r', subset=['Gap (Line - Lab)']), 
                use_container_width=True, hide_index=True
            )

    list_ma_son_tab2 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab2:
        tab_top_risk, tab_custom = st.tabs(["🚨 Auto-Analysis: Top 15 At-Risk Codes", "🔍 Manual Search & Analysis"])

        with tab_top_risk:
            st.markdown("### Top 15 Paint Codes Approaching or Exceeding Limits")
            st.caption("Only displaying codes with ≥ 5 production batches.")
            if not risk_alert.empty:
                top_15_codes = risk_alert['Paint Code'].head(15).tolist()
                for i, code in enumerate(top_15_codes):
                    st.markdown(f"#### #{i+1}: Paint Code `{code}`")
                    render_spc_analysis(code, dff, key_suffix=f"top15_{i}")
                    st.markdown("---")
            else:
                st.success("✅ No at-risk codes found! All processes are highly stable.")

        with tab_custom:
            st.markdown("### Manual Paint Code Analysis")
            col_search1, col_search2 = st.columns([1, 2])
            with col_search1:
                search_keyword = st.text_input("🔍 Quick Search Paint Code:", "", placeholder="Type part of code (e.g., SC8, 14SE...)").upper()
            
            filtered_list = [code for code in list_ma_son_tab2 if search_keyword in str(code).upper()]
            
            with col_search2:
                if filtered_list:
                    sel_ma_son_tab2 = st.selectbox(f"🎯 Select Paint Code ({len(filtered_list)} found):", filtered_list, key="custom_select")
                else:
                    st.warning(f"❌ No paint code found containing '{search_keyword}'")
                    st.stop()
            
            st.markdown("---")
            render_spc_analysis(sel_ma_son_tab2, dff, key_suffix="custom")

# ==========================================
# VIEW 2: COLOR DEVIATION
# ==========================================
elif view_mode == "🎨 Color & ΔE Analysis":
    st.info("💡 Trend analysis of Total Color Difference (ΔE) and distribution of individual color components (ΔL, Δa, Δb) to detect color drift.")
    
    list_ma_son_tab3 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    
    if list_ma_son_tab3:
        sel_ma_son_tab3 = st.selectbox("🎯 Select Full Paint Code for Color Analysis:", list_ma_son_tab3, key="tab3_mason")
        dff_c = dff[dff['Ma_Son'] == sel_ma_son_tab3].copy()
        
        if not dff_c.empty:
            dff_c_batch = dff_c.groupby('Batch_Lot', as_index=False).agg({
                'Ngay_SX': 'min',
                'ΔE': 'mean'
            }).sort_values('Ngay_SX')
            
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
                ax_L.set_ylabel("Count")
                st.pyplot(fig_L)
                
            with col_a:
                fig_a, ax_a = plt.subplots(figsize=(5, 4))
                sns.histplot(dff_c['da_N'], kde=True, color='#e74c3c', ax=ax_a) 
                ax_a.axvline(0, color='black', ls='--', lw=1.5)
                ax_a.set_title("Red/Green (Δa)")
                ax_a.set_xlabel("Δa (+ Redder / - Greener)")
                ax_a.set_ylabel("")
                st.pyplot(fig_a)
                
            with col_b:
                fig_b, ax_b = plt.subplots(figsize=(5, 4))
                sns.histplot(dff_c['db_N'], kde=True, color='#f1c40f', ax=ax_b) 
                ax_b.axvline(0, color='black', ls='--', lw=1.5)
                ax_b.set_title("Yellow/Blue (Δb)")
                ax_b.set_xlabel("Δb (+ Yellower / - Bluer)")
                ax_b.set_ylabel("")
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
# VIEW 3: STATISTICAL CONTROL LIMITS
# ==========================================
elif view_mode == "📊 Statistical Control Limits":
    st.info("💡 Strict Input Control: Calculates dynamic process limits for Lab Gloss based on production history. To reduce downstream variation, new Lab limits are constrained so they NEVER exceed current official specs.")
    
    list_ma_son_tab4 = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab4:
        col_search1, col_search2 = st.columns([1, 2])
        with col_search1:
            search_keyword = st.text_input("🔍 Quick Search Paint Code:", "", placeholder="Type part of code (e.g., SC8...)").upper()
        
        filtered_list = [code for code in list_ma_son_tab4 if search_keyword in str(code).upper()]
        
        with col_search2:
            if filtered_list:
                sel_ma_son_tab4 = st.selectbox(f"🎯 Select Paint Code ({len(filtered_list)} found):", filtered_list, key="spc_limit_select")
            else:
                st.warning(f"❌ No paint code found containing '{search_keyword}'")
                st.stop()
        
        # Sort data sequentially by date and coil number
        dff_spc = dff[dff['Ma_Son'] == sel_ma_son_tab4].copy()
        dff_spc = dff_spc.dropna(subset=['Online_Gloss_Top', 'Gloss_Lab', 'Ngay_SX']).sort_values(['Ngay_SX', 'Coil_No'])
        
        if len(dff_spc) >= 5:
            lab_data = dff_spc['Gloss_Lab'].values
            line_data = dff_spc['Online_Gloss_Top'].values
            
            # Official Limits
            lab_lsl = dff_spc['Gloss_LSL'].iloc[0]
            lab_usl = dff_spc['Gloss_USL'].iloc[0]
            line_lsl = dff_spc['Line_LSL'].iloc[0]
            line_usl = dff_spc['Line_USL'].iloc[0]
            
            st.success(f"📅 **Historical Volume:** `{dff_spc['Batch_Lot'].nunique()}` Batches (`{len(lab_data)}` Coils).")

            # =========================================================
            # ADJUSTABLE MULTIPLIERS
            # =========================================================
            st.markdown("---")
            st.subheader("⚙️ Adjust Statistical Multipliers")
            c_mod1, c_mod2 = st.columns(2)
            with c_mod1:
                sigma_mult = st.number_input("Sigma Multiplier (Standard Deviation)", value=3.0, step=0.5, help="Standard is 3.0. Lower values (e.g. 2.0) will make limits stricter.")
            with c_mod2:
                k_mult = st.number_input("K Multiplier (IQR Method)", value=1.5, step=0.1, help="Standard is 1.5. Higher values (e.g. 3.0) allow more variation before flagging as outlier.")
            
            # --- 1. Standard Deviation Method ---
            mean_lab = np.mean(lab_data)
            std_lab = np.std(lab_data, ddof=1)
            lcl_std_raw = mean_lab - sigma_mult * std_lab
            ucl_std_raw = mean_lab + sigma_mult * std_lab
            
            # --- 2. IQR Method ---
            q1 = np.percentile(lab_data, 25)
            q3 = np.percentile(lab_data, 75)
            iqr_val = q3 - q1
            lcl_iqr_raw = q1 - k_mult * iqr_val
            ucl_iqr_raw = q3 + k_mult * iqr_val
            
            # --- 3. I-MR Method (Individuals-Moving Range) ---
            mr = np.abs(np.diff(lab_data))
            mean_mr = np.mean(mr) if len(mr) > 0 else 0
            lcl_imr_raw = mean_lab - 2.66 * mean_mr
            ucl_imr_raw = mean_lab + 2.66 * mean_mr
            
            # --- APPLY BUSINESS RULE: MUST BE STRICTER THAN OFFICIAL ---
            lcl_std, ucl_std = max(lcl_std_raw, lab_lsl), min(ucl_std_raw, lab_usl)
            lcl_iqr, ucl_iqr = max(lcl_iqr_raw, lab_lsl), min(ucl_iqr_raw, lab_usl)
            lcl_imr, ucl_imr = max(lcl_imr_raw, lab_lsl), min(ucl_imr_raw, lab_usl)
            
            # =========================================================
            # METHODS COMPARISON SUMMARY TABLE
            # =========================================================
            st.markdown("---")
            st.subheader("📋 Methods Comparison Summary")
            st.caption(f"Comparing the new calculated limits against current official specs. (New limits are constrained to never exceed Official Lab Limits).")
            
            summary_data = [
                {
                    "Method": f"Standard Deviation ({sigma_mult}σ)", 
                    "Current Lab LSL": lab_lsl, 
                    "Current Lab USL": lab_usl, 
                    "New Lab LCL": lcl_std, 
                    "New Lab UCL": ucl_std,
                    "Mean": mean_lab,
                    "Std Dev (s)": std_lab
                },
                {
                    "Method": f"Interquartile Range (K={k_mult})", 
                    "Current Lab LSL": lab_lsl, 
                    "Current Lab USL": lab_usl, 
                    "New Lab LCL": lcl_iqr, 
                    "New Lab UCL": ucl_iqr,
                    "Mean": mean_lab,
                    "Std Dev (s)": std_lab
                },
                {
                    "Method": "Individuals-Moving Range (I-MR)", 
                    "Current Lab LSL": lab_lsl, 
                    "Current Lab USL": lab_usl, 
                    "New Lab LCL": lcl_imr, 
                    "New Lab UCL": ucl_imr,
                    "Mean": mean_lab,
                    "Std Dev (s)": std_lab
                }
            ]
            
            summary_df = pd.DataFrame(summary_data)
            
            st.dataframe(
                summary_df.style.format({
                    "Current Lab LSL": "{:.1f}", "Current Lab USL": "{:.1f}", 
                    "New Lab LCL": "{:.2f}", "New Lab UCL": "{:.2f}",
                    "Mean": "{:.2f}", "Std Dev (s)": "{:.2f}"
                }).set_properties(**{'font-weight': 'bold'}, subset=['Method'])
                  .set_properties(**{'background-color': '#e8f5e9', 'color': '#2e7d32', 'font-weight': 'bold'}, subset=['New Lab LCL', 'New Lab UCL']),
                use_container_width=True, hide_index=True
            )

            # =========================================================
            # VISUALIZATION
            # =========================================================
            st.markdown("---")
            sel_method = st.radio("🧮 Select Statistical Method to Visualize:", 
                                  [f"Standard Deviation ({sigma_mult}σ)", f"Interquartile Range (K={k_mult})", "Individuals-Moving Range (I-MR)"], 
                                  horizontal=True)
            
            if "Standard Deviation" in sel_method:
                plot_lcl, plot_ucl, plot_center = lcl_std, ucl_std, mean_lab
            elif "Interquartile Range" in sel_method:
                plot_lcl, plot_ucl, plot_center = lcl_iqr, ucl_iqr, np.median(lab_data)
            else:
                plot_lcl, plot_ucl, plot_center = lcl_imr, ucl_imr, mean_lab

            # --- METRICS DISPLAY ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Official Lab LSL", f"{lab_lsl:.1f}")
            col2.metric(f"New Lab LCL", f"{plot_lcl:.2f}", delta=f"{plot_lcl - lab_lsl:+.2f} (Stricter)" if plot_lcl > lab_lsl else "Unchanged", delta_color="inverse" if plot_lcl > lab_lsl else "off")
            col3.metric(f"New Lab UCL", f"{plot_ucl:.2f}", delta=f"{plot_ucl - lab_usl:+.2f} (Stricter)" if plot_ucl < lab_usl else "Unchanged", delta_color="inverse" if plot_ucl < lab_usl else "off")
            col4.metric("Official Lab USL", f"{lab_usl:.1f}")
            
            # --- CHART 1: TREND LINE ---
            st.markdown("---")
            st.write("**📈 Trend Line: Sequential Production (Lab & Line)**")
            fig_trend, ax_trend = plt.subplots(figsize=(14, 4.5))
            
            x_axis = range(len(lab_data))
            ax_trend.plot(x_axis, lab_data, marker='o', color='#3498db', lw=2, label='Lab Gloss')
            ax_trend.plot(x_axis, line_data, marker='s', color='#e67e22', lw=2, label='Line Gloss')
            
            # Draw Limits
            ax_trend.axhline(lab_usl, color='red', ls='-', lw=1.5, alpha=0.3, label=f'Official Lab USL ({lab_usl:.1f})')
            ax_trend.axhline(lab_lsl, color='red', ls='-', lw=1.5, alpha=0.3, label=f'Official Lab LSL ({lab_lsl:.1f})')
            
            ax_trend.axhline(line_usl, color='orange', ls='--', lw=1.5, alpha=0.4, label=f'Official Line USL ({line_usl:.1f})')
            ax_trend.axhline(line_lsl, color='orange', ls='--', lw=1.5, alpha=0.4, label=f'Official Line LSL ({line_lsl:.1f})')
            
            ax_trend.axhline(plot_ucl, color='green', ls='-', lw=2.5, label=f'NEW Lab UCL ({plot_ucl:.2f})')
            ax_trend.axhline(plot_lcl, color='green', ls='-', lw=2.5, label=f'NEW Lab LCL ({plot_lcl:.2f})')
            
            # Shade the "safe" new Lab zone
            ax_trend.fill_between(x_axis, plot_lcl, plot_ucl, color='green', alpha=0.05)

            ax_trend.set_xlabel("Sequential Production Order (Coil Index)")
            ax_trend.set_ylabel("Gloss (GU)")
            ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_trend)
            plt.close(fig_trend)
            
            # --- CHART 2: NORMAL DISTRIBUTION ---
            st.markdown("---")
            st.write("**📊 Data Distribution & Normal Curve (Lab & Line)**")
            fig_dist, ax_dist = plt.subplots(figsize=(14, 4.5))
            
            min_val = min(np.min(lab_data), np.min(line_data), lab_lsl, line_lsl) - 2
            max_val = max(np.max(lab_data), np.max(line_data), lab_usl, line_usl) + 2
            bins_arr = np.linspace(min_val, max_val, 15)
            bin_width = bins_arr[1] - bins_arr[0]
            
            sns.histplot(lab_data, stat="count", bins=bins_arr, color='#3498db', alpha=0.4, label='Lab Bins', ax=ax_dist)
            sns.histplot(line_data, stat="count", bins=bins_arr, color='#e67e22', alpha=0.4, label='Line Bins', ax=ax_dist)
            
            x_axis_dist = np.linspace(min_val, max_val, 200)
            
            # Lab Curve
            if std_lab > 0:
                y_lab_scaled = stats.norm.pdf(x_axis_dist, mean_lab, std_lab) * len(lab_data) * bin_width
                ax_dist.plot(x_axis_dist, y_lab_scaled, color='#2980b9', lw=2.5, label='Lab Normal Curve')
            
            # Line Curve
            mean_line = np.mean(line_data)
            std_line = np.std(line_data, ddof=1)
            if std_line > 0:
                y_line_scaled = stats.norm.pdf(x_axis_dist, mean_line, std_line) * len(line_data) * bin_width
                ax_dist.plot(x_axis_dist, y_line_scaled, color='#d35400', lw=2.5, label='Line Normal Curve')
                
            # Draw Limits on Distribution Chart
            ax_dist.axvline(lab_usl, color='red', ls='-', lw=1.5, alpha=0.3)
            ax_dist.axvline(lab_lsl, color='red', ls='-', lw=1.5, alpha=0.3)
            ax_dist.axvline(line_usl, color='orange', ls='--', lw=1.5, alpha=0.4)
            ax_dist.axvline(line_lsl, color='orange', ls='--', lw=1.5, alpha=0.4)
            
            ax_dist.axvline(plot_ucl, color='green', ls='-', lw=2.5, label=f'NEW Lab UCL ({plot_ucl:.2f})')
            ax_dist.axvline(plot_lcl, color='green', ls='-', lw=2.5, label=f'NEW Lab LCL ({plot_lcl:.2f})')
            
            # Shade the safe zone
            ax_dist.axvspan(plot_lcl, plot_ucl, color='green', alpha=0.05)
            
            ax_dist.set_xlabel("Gloss (GU)")
            ax_dist.set_ylabel("Number of Coils (Count)")
            ax_dist.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            
        else:
            st.warning("⚠️ Insufficient data. We need at least 5 consecutive production coils to calculate meaningful statistical control limits.")
    else:
        st.warning("No data available.")

# ==========================================
# VIEW 4: PAIRED DIFFERENCE ANALYSIS (NEW)
# ==========================================
elif view_mode == "⚖️ Paired Difference Analysis":
    st.info("💡 Paired Difference Analysis + Batch Aggregation: Quantifies the systematic offset between Lab and Line to redefine Lab control limits.")

    list_ma_son_tab_offset = sorted(dff['Ma_Son'].dropna().unique().tolist())
    if list_ma_son_tab_offset:
        col_search1, col_search2 = st.columns([1, 2])
        with col_search1:
            search_keyword = st.text_input("🔍 Quick Search Paint Code:", "", placeholder="Type part of code (e.g., SC8...)").upper()

        filtered_list = [code for code in list_ma_son_tab_offset if search_keyword in str(code).upper()]

        with col_search2:
            if filtered_list:
                sel_ma_son_tab_offset = st.selectbox(f"🎯 Select Paint Code ({len(filtered_list)} found):", filtered_list, key="offset_select")
            else:
                st.warning(f"❌ No paint code found containing '{search_keyword}'")
                st.stop()

        dff_offset = dff[dff['Ma_Son'] == sel_ma_son_tab_offset].copy()
        dff_offset = dff_offset.dropna(subset=['Online_Gloss_Top', 'Gloss_Lab', 'Ngay_SX']).sort_values(['Ngay_SX', 'Coil_No'])

        # 📌 STEP 1: Batch Aggregation
        batch_analysis = dff_offset.groupby('Batch_Lot').agg({
            'Gloss_Lab': 'first',         # Lab value per batch
            'Online_Gloss_Top': 'mean',   # Line mean per batch
            'Gloss_LSL': 'first',
            'Gloss_USL': 'first'
        }).reset_index()

        if len(batch_analysis) >= 5:
            # 📌 STEP 2: Calculate Delta (Δ)
            batch_analysis['Delta'] = batch_analysis['Online_Gloss_Top'] - batch_analysis['Gloss_Lab']

            # 📌 STEP 3: Analyze Delta
            mean_delta = batch_analysis['Delta'].mean()  # Systematic Offset
            std_delta = batch_analysis['Delta'].std()    # Variation

            # 📌 STEP 4: Calculate New Lab Target and Control Limits
            target_line = (batch_analysis['Gloss_LSL'].iloc[0] + batch_analysis['Gloss_USL'].iloc[0]) / 2

            # Core Formula: Lab_target = Target_Line - Mean(Δ)
            suggested_lab_target = target_line - mean_delta
            suggested_lab_ucl = suggested_lab_target + (3 * std_delta)
            suggested_lab_lcl = suggested_lab_target - (3 * std_delta)

            # --- DISPLAY METRICS ---
            st.markdown("### 🔑 Recommended Parameters (Based on Systematic Offset)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Constant Bias (Mean Δ)", f"{mean_delta:+.2f} GU", help=">0: Line higher than Lab. <0: Line lower than Lab")
            c2.metric("Δ Stability (Std)", f"{std_delta:.2f} GU")
            c3.metric("🎯 Suggested Lab Target", f"{suggested_lab_target:.1f} GU")

            # Display New Limits
            c4.metric(
                "Suggested Lab Limits (±3σ)",
                f"{suggested_lab_lcl:.1f} - {suggested_lab_ucl:.1f}",
                delta=f"Span: {suggested_lab_ucl - suggested_lab_lcl:.1f} GU",
                delta_color="off"
            )

            # --- STEP 5: COMPREHENSIVE VISUAL DIAGNOSTICS (Δ ANALYSIS) ---
            st.markdown("---")
            st.markdown("### 📊 Step 3: Comprehensive Δ Analysis")
            st.caption("Visualizing the systematic offset using Control Trends, Probability Density, and Empirical Distributions.")

            # --- Chart 1: Constant vs. Drift Bias (Time-Series Control Chart) ---
            st.markdown("#### 1. Bias Trend (Constant vs. Drift)")
            fig_trend, ax_trend = plt.subplots(figsize=(14, 4))
            
            # X-axis formatting
            batch_labels = batch_analysis['Batch_Lot'].astype(str)
            ax_trend.plot(batch_labels, batch_analysis['Delta'], marker='o', color='#2c3e50', lw=2, label='Batch Δ (Line - Lab)')
            
            # Draw Bias and Control Limits
            ax_trend.axhline(mean_delta, color='#e74c3c', ls='-', lw=2, label=f'Mean Bias ({mean_delta:+.2f})')
            ax_trend.axhline(mean_delta + (3 * std_delta), color='#27ae60', ls='--', lw=1.5, label='+3σ Limit')
            ax_trend.axhline(mean_delta - (3 * std_delta), color='#27ae60', ls='--', lw=1.5, label='-3σ Limit')
            ax_trend.axhline(0, color='#7f8c8d', ls=':', lw=1.5, label='Zero Offset (Ideal)')
            
            # Format X-axis to prevent clutter
            ax_trend.set_xlabel("Batch Lot Sequence")
            ax_trend.set_ylabel("Deviation Δ (GU)")
            plt.xticks(rotation=45, ha='right')
            locs, labels = plt.xticks()
            if len(locs) > 30:
                step = max(1, len(locs) // 20) 
                for i, label in enumerate(labels):
                    if i % step != 0: label.set_visible(False)
                    
            ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig_trend)
            plt.close(fig_trend)

            # Create columns for the next two charts
            col_chart1, col_chart2 = st.columns(2)

            # Setup data for distributions
            mean_lab = batch_analysis['Gloss_Lab'].mean()
            std_lab = batch_analysis['Gloss_Lab'].std() if batch_analysis['Gloss_Lab'].std() > 0 else 0.5
            mean_line = batch_analysis['Online_Gloss_Top'].mean()
            std_line = batch_analysis['Online_Gloss_Top'].std() if batch_analysis['Online_Gloss_Top'].std() > 0 else 0.5

            x_min = min(mean_lab - 3.5*std_lab, mean_line - 3.5*std_line)
            x_max = max(mean_lab + 3.5*std_lab, mean_line + 3.5*std_line)
            x_axis = np.linspace(x_min, x_max, 200)

            with col_chart1:
                # --- Chart 2: Assigned vs Achieved (Shifted Normal Curves) ---
                st.markdown("#### 2. Systematic Shift (Normal Curves)")
                fig_curve, ax_curve = plt.subplots(figsize=(6, 5))
                
                # Lab Curve (Assigned/Target)
                y_lab = stats.norm.pdf(x_axis, mean_lab, std_lab)
                ax_curve.plot(x_axis, y_lab, color='#27ae60', lw=2.5, label='Lab (Assigned)')
                ax_curve.axvline(mean_lab, color='#27ae60', ls='--', lw=1)
                
                # Line Curve (Achieved)
                y_line = stats.norm.pdf(x_axis, mean_line, std_line)
                ax_curve.plot(x_axis, y_line, color='#c0392b', lw=2.5, label='Line (Achieved)')
                ax_curve.axvline(mean_line, color='#c0392b', ls='--', lw=1)
                
                # Draw Bias Annotations
                y_annotate = max(max(y_lab), max(y_line)) * 0.5
                ax_curve.annotate('', xy=(mean_lab, y_annotate), xytext=(mean_line, y_annotate),
                                  arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=2))
                ax_curve.text((mean_lab + mean_line)/2, y_annotate + 0.01, f'Bias: {mean_delta:+.2f}', 
                              ha='center', va='bottom', color='#f39c12', fontweight='bold')

                ax_curve.set_xlabel("Gloss Value (GU)")
                ax_curve.set_ylabel("Probability Density")
                ax_curve.legend()
                st.pyplot(fig_curve)
                plt.close(fig_curve)

            with col_chart2:
                # --- Chart 3: Overlap of Distributions (Histograms) ---
                st.markdown("#### 3. Population Overlap (Histograms)")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
                
                # Determine shared bins
                min_bin = min(batch_analysis['Gloss_Lab'].min(), batch_analysis['Online_Gloss_Top'].min())
                max_bin = max(batch_analysis['Gloss_Lab'].max(), batch_analysis['Online_Gloss_Top'].max())
                bins_shared = np.linspace(min_bin - 1, max_bin + 1, 15)

                sns.histplot(batch_analysis['Gloss_Lab'], bins=bins_shared, color='#27ae60', alpha=0.4, label='Lab Population', ax=ax_hist)
                sns.histplot(batch_analysis['Online_Gloss_Top'], bins=bins_shared, color='#3498db', alpha=0.6, label='Line Population', ax=ax_hist)
                
                ax_hist.set_xlabel("Gloss Value Bin Range (GU)")
                ax_hist.set_ylabel("Batch Count")
                ax_hist.legend()
                st.pyplot(fig_hist)
                plt.close(fig_hist)

# ==========================================
# VIEW 5: SUPPLIER COMPARISON
# ==========================================
elif view_mode == "🤝 Supplier Comparison":
    st.info("💡 Flexible Mode: Select a specific code for capability comparison (Cpk). Select 'All' to evaluate overall stability of a color group based on Target Deviation (ΔGloss).")
    
    st.markdown("---")
    st.subheader("🚨 Negotiation Radar (Supplier Blacklist)")
    st.caption("Paint codes with **Cpk < 1.0** (unstable gloss) or **ΔE Max > 1.0** (color shift) will be flagged here.")
    
    dff_radar = dff.dropna(subset=['Online_Gloss_Top', 'Supplier', 'Gloss_LSL', 'Gloss_USL', 'Color_Group', 'Color_Code', 'dL_N', 'da_N', 'db_N'])
    dff_radar = dff_radar[(dff_radar['Gloss_LSL'] > 0) & (dff_radar['Gloss_USL'] > 0) & (dff_radar['Online_Gloss_Top'] > 0)]
    
    if not dff_radar.empty:
        radar_summary = dff_radar.groupby(['Color_Group', 'Color_Code', 'Supplier']).agg(
            Coil_Count=('Online_Gloss_Top', 'count'),
            LSL=('Gloss_LSL', 'first'),
            USL=('Gloss_USL', 'first'),
            Mean_Line=('Online_Gloss_Top', 'mean'),
            Std_Line=('Online_Gloss_Top', 'std'),
            dE_Max=('ΔE', 'max')
        ).reset_index()
        
        radar_summary = radar_summary[radar_summary['Coil_Count'] >= 3]
        
        def calc_cpk_radar(row):
            if pd.isna(row['Std_Line']) or row['Std_Line'] == 0: return np.nan
            return min((row['USL'] - row['Mean_Line']) / (3 * row['Std_Line']), (row['Mean_Line'] - row['LSL']) / (3 * row['Std_Line']))
            
        radar_summary['Cpk (Line)'] = radar_summary.apply(calc_cpk_radar, axis=1)
        radar_alert = radar_summary[(radar_summary['Cpk (Line)'] < 1.0) | (radar_summary['dE_Max'] > 1.0)].copy()
        
        if not radar_alert.empty:
            radar_alert = radar_alert.sort_values(by=['Cpk (Line)'], ascending=True)
            radar_alert.columns = ['Color Group', 'Color Code (4 Digits)', 'Supplier', 'Coils', 'LSL', 'USL', 'Mean (Line)', 'Std (Line)', 'Max ΔE', 'Cpk (Line)']
            st.dataframe(
                radar_alert.style.format({
                    'LSL': '{:.0f}', 'USL': '{:.0f}', 'Mean (Line)': '{:.1f}',
                    'Std (Line)': '{:.2f}', 'Max ΔE': '{:.2f}', 'Cpk (Line)': '{:.2f}'
                }).background_gradient(cmap='Reds_r', subset=['Cpk (Line)'])
                  .background_gradient(cmap='Reds', subset=['Max ΔE']),
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
        
        counts = dff_comp['Supplier'].value_counts()
        valid_suppliers = counts[counts >= 2].index
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
                        Coil_Count=('Batch_Lot', 'count'), Mean_Dev=('Gloss_Dev', 'mean'),
                        Std_Dev=('Gloss_Dev', 'std'), Avg_dE=('ΔE', 'mean')
                    ).reset_index()
                    comp_table = comp_table.sort_values('Std_Dev', ascending=True)
                    comp_table.columns = ['Supplier', 'Coils', 'Avg Target Dev', 'Dispersion (Std)', 'Avg ΔE']
                    st.dataframe(
                        comp_table.style.format({
                            'Avg Target Dev': '{:+.2f}', 'Dispersion (Std)': '{:.2f}', 'Avg ΔE': '{:.2f}'
                        }).background_gradient(cmap='RdYlGn_r', subset=['Dispersion (Std)']), 
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.subheader("Line Capability Table (Online Cpk)")
                    comp_table = dff_comp.groupby('Supplier').agg(
                        Coil_Count=('Batch_Lot', 'count'), LSL=('Gloss_LSL', 'mean'), USL=('Gloss_USL', 'mean'), 
                        Mean_Gloss=('Online_Gloss_Top', 'mean'), Std_Gloss=('Online_Gloss_Top', 'std'), Avg_dE=('ΔE', 'mean')
                    ).reset_index()
                    def calc_cpk(row):
                        if pd.isna(row['Std_Gloss']) or row['Std_Gloss'] == 0: return np.nan
                        return min((row['USL'] - row['Mean_Gloss']) / (3 * row['Std_Gloss']), (row['Mean_Gloss'] - row['LSL']) / (3 * row['Std_Gloss']))
                    comp_table['Cpk (Line)'] = comp_table.apply(calc_cpk, axis=1)
                    comp_table = comp_table.sort_values('Cpk (Line)', ascending=False)
                    comp_table.columns = ['Supplier', 'Coils', 'LSL', 'USL', 'Mean (Line)', 'Std (Line)', 'Avg ΔE', 'Cpk (Line)']
                    st.dataframe(
                        comp_table.style.format({
                            'LSL': '{:.0f}', 'USL': '{:.0f}', 'Mean (Line)': '{:.1f}',
                            'Std (Line)': '{:.2f}', 'Avg ΔE': '{:.2f}', 'Cpk (Line)': '{:.2f}'
                        }).background_gradient(cmap='RdYlGn', subset=['Cpk (Line)']), 
                        use_container_width=True, hide_index=True
                    )

            # --- BATCH TO BATCH ---
            st.markdown("---")
            st.subheader("📉 Batch-to-Batch Gloss Variation")
            st.caption("Detailed view of the specific batches with the highest and lowest average gloss. Exposes exact lot numbers for vendor accountability.")
            
            batch_means = dff_comp.groupby(['Supplier', 'Batch_Lot']).agg(
                Mean_Line=('Online_Gloss_Top', 'mean'),
                LSL=('Gloss_LSL', 'first'),
                USL=('Gloss_USL', 'first')
            ).reset_index()
            
            b2b_records = []
            for sup, group in batch_means.groupby('Supplier'):
                if len(group) >= 2:
                    idx_min = group['Mean_Line'].idxmin()
                    idx_max = group['Mean_Line'].idxmax()
                    
                    min_row = group.loc[idx_min]
                    max_row = group.loc[idx_max]
                    
                    b2b_records.append({
                        'Supplier': sup,
                        'Batches': len(group),
                        'LSL': min_row['LSL'],
                        'USL': min_row['USL'],
                        'Min Batch': min_row['Batch_Lot'],
                        'Min Avg': min_row['Mean_Line'],
                        'Max Batch': max_row['Batch_Lot'],
                        'Max Avg': max_row['Mean_Line'],
                        'Gap': max_row['Mean_Line'] - min_row['Mean_Line']
                    })
            
            b2b_table = pd.DataFrame(b2b_records)
            
            if not b2b_table.empty:
                b2b_table = b2b_table.sort_values('Gap', ascending=False)
                
                st.dataframe(
                    b2b_table.style.format({
                        'LSL': '{:.0f}', 'USL': '{:.0f}', 
                        'Min Avg': '{:.1f}', 'Max Avg': '{:.1f}', 'Gap': '{:.1f}'
                    }).background_gradient(cmap='Oranges', subset=['Gap']), 
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough multi-batch data to compare Batch-to-Batch gloss variation.")

            # --- COLOR DRIFT ---
            st.markdown("---")
            st.subheader("🎨 Batch Color Drift Detailed Analysis")
            st.caption("Table displays AVERAGE values of color components (ΔL, Δa, Δb) per batch. The **Full Paint Code** column helps Vendors trace exact formulas.")
            st.caption("🔴 **Dark Red:** Significant shift towards Lighter (ΔL+), Redder (Δa+), or Yellower (Δb+).")
            st.caption("🔵 **Dark Blue:** Significant shift towards Darker (ΔL-), Greener (Δa-), or Bluer (Δb-).")
            
            color_drift = dff_comp.groupby(['Supplier', 'Batch_Lot']).agg(
                Full_Paint_Code=('Ma_Son', 'first'),
                Ngay_SX=('Ngay_SX', 'min'),
                Mean_dL=('dL_N', 'mean'),
                Mean_da=('da_N', 'mean'),
                Mean_db=('db_N', 'mean'),
                Max_dE=('ΔE', 'max')
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
                color_drift.style.format({
                    'ΔL (Avg)': '{:+.2f}', 'Δa (Avg)': '{:+.2f}', 'Δb (Avg)': '{:+.2f}', 'Max ΔE': '{:.2f}'
                }).background_gradient(cmap='bwr', subset=['ΔL (Avg)'], vmin=-0.5, vmax=0.5) 
                  .background_gradient(cmap='bwr', subset=['Δa (Avg)'], vmin=-0.3, vmax=0.3) 
                  .background_gradient(cmap='bwr', subset=['Δb (Avg)'], vmin=-0.3, vmax=0.3) 
                  .background_gradient(cmap='Reds', subset=['Max ΔE'], vmin=0, vmax=1.0),
                use_container_width=True, hide_index=True
            )

        else:
            st.warning("⚠️ Insufficient data (needs at least 2 coils/supplier with Line data) to perform comparison.")

# ==========================================
# VIEW 6: SUMMARY DATA REPORT
# ==========================================
elif view_mode == "📋 Summary Data Report":
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

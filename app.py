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

# --- 2. DATA LOAD & PREP ---
@st.cache_data(ttl=300)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url, engine='pyarrow')
        
        # COLUMN MAPPING
        col_map = {
            '產出鋼捲號碼': 'Coil_No', '鋼捲號碼': 'Coil_No', '鋼捲號': 'Coil_No', '卷号': 'Coil_No', 'Coil ID': 'Coil_No',
            '訂單號碼': 'Order_No', '訂單號': 'Order_No', '工單號': 'Order_No', '工單': 'Order_No', 
            '線別': 'Line', '產線': 'Line', '生產線': 'Line', '機台': 'Line', 
            '生產日期': 'Prod_Date', '製造批號': 'Batch_Lot', '塗料編號': 'Paint_Code',
            '光澤': 'Gloss_Lab',
            'NORTH_TOP_BLANCH': 'G_Top_N', 'SOUTH_TOP_BLANCH': 'G_Top_S',
            'NORTH_BACK_BLANCH': 'G_Back_N', 'SOUTH_BACK_BLANCH': 'G_Back_S',
            'NORTH_TOP_DELTA_E': 'dE_N', 'SOUTH_TOP_DELTA_E': 'dE_S',
            'NORTH_TOP_DELTA_L': 'dL_N', 'NORTH_TOP_DELTA_A': 'da_N', 'NORTH_TOP_DELTA_B': 'db_N',
            'NORTH_TOP_FILM_THICK': 'DFT_N', 'SOUTH_TOP_FILM_THICK': 'DFT_S',
            '正面漆膜厚': 'Target_Top', 'TTMFILM_THICK': 'Target_Primer'
        }
        
        for col in df.columns:
            if '下限' in col and '光澤' in col: col_map[col] = 'Gloss_LSL'
            elif '上限' in col and '光澤' in col: col_map[col] = 'Gloss_USL'
            
        df = df.rename(columns=col_map)
        
        if 'Line' not in df.columns: df['Line'] = 'Unknown Line'
        if 'Order_No' not in df.columns: df['Order_No'] = 'Unknown Order'
        if 'Coil_No' not in df.columns: df['Coil_No'] = 'Unknown Coil'
            
        df['Paint_Code_Str'] = df['Paint_Code'].astype(str).str.upper().str.strip()

        v_map = {
            'S':'Yungchi', 'T':'AKZO NOBEL(Taiwan)', 'A':'AKZO NOBEL', 'B':'Beckers', 
            'C':'Nan Pao', 'U':'Quali Poly', 'N':'Nippon', 'K':'Kansai', 
            'V':'Valspar', 'J':'Valspar (SW)', 'L':'KCC', 'R':'Noroo', 'Q':'Paoqun'
        }
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Paint_Code_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Paint_Code_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Paint_Code_Str'].str[6].map(c_map).fillna('Other')
        df['Color_Code'] = df['Paint_Code_Str'].str[-4:] 
        
        # Consolidate specific color codes into a unified reporting group
        df['Color_Code'] = df['Color_Code'].replace({'GE00': 'GE_Group', 'GE01': 'GE_Group'})

        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL', 'DFT_N', 'DFT_S', 'Target_Top', 'Target_Primer']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        # DATA CLEANING
        df = df.dropna(subset=['Gloss_Lab', 'Paint_Code'])
        df = df[df['Gloss_Lab'] > 0] 
        invalid_codes = ['0', '00', '000', '0000', 'NA', 'N/A', 'NAN', 'NULL', 'NONE']
        df = df[~df['Paint_Code_Str'].isin(invalid_codes)]
        df = df[~df['Color_Code'].isin(invalid_codes)]

        df = df.dropna(subset=['Gloss_LSL', 'Gloss_USL'])
        df = df[(df['Gloss_LSL'] > 0) & (df['Gloss_USL'] > 0)]

        df['Prod_Date'] = pd.to_datetime(df['Prod_Date'], errors='coerce').dt.date
        
        if 'G_Top_N' in df.columns and 'G_Top_S' in df.columns:
            df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        else:
            df['Online_Gloss_Top'] = np.nan
            
        df = df.dropna(subset=['Online_Gloss_Top'])
        df = df[df['Online_Gloss_Top'] > 0]

        if 'dE_N' in df.columns and 'dE_S' in df.columns:
            df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        else:
            df['ΔE'] = np.nan
            
        df['Gap_Gloss'] = df['Online_Gloss_Top'] - df['Gloss_Lab']
        
        if 'DFT_N' in df.columns and 'DFT_S' in df.columns:
            df['Avg_DFT'] = df[['DFT_N', 'DFT_S']].mean(axis=1)
        else:
            df['Avg_DFT'] = np.nan
            
        if 'Target_Top' in df.columns and 'Target_Primer' in df.columns:
            df['Target_DFT'] = (df['Target_Top'].fillna(0) + df['Target_Primer'].fillna(0)) * 0.9
        else:
            df['Target_DFT'] = np.nan

        # DOWNCASTING TO CATEGORY TO SAVE RAM
        for col in ['Supplier', 'Coating_Type', 'Color_Group']:
            df[col] = df[col].astype('category')
            
        return df.dropna(subset=['Supplier', 'Prod_Date']).sort_values('Prod_Date')
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return pd.DataFrame()

df_raw = load_and_prep_data()
if df_raw.empty: st.stop()

# --- 3. SIDEBAR: 4-TIER ARCHITECTURE ---
with st.sidebar:
    st.title("🏭 QA Dashboard Pro")
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🛠️ Analysis Structure")
    
    analysis_level = st.radio(
        "Select Analysis Tier:",
        ["📋 Tier 1: Executive View", 
         "🤝 Tier 2: Supplier Intelligence",
         "📈 Tier 3: Operational View", 
         "🔬 Tier 4: Diagnostic View"]
    )

    if "Tier 1" in analysis_level:
        view_mode = "Master Summary & Pareto"
    elif "Tier 2" in analysis_level:
        view_mode = "Supplier Intelligence (Apples-to-Apples)"
    elif "Tier 3" in analysis_level:
        view_mode = st.selectbox("Select Report:", ["Gloss Trend (SPC)", "Color Shift Analysis", "Statistical Limits (Scope Comparison)"])
    else:
        view_mode = st.selectbox("Select Report:", ["Process vs Material (DFT & Root Cause)", "Predictive Compensation & Targeting"])

    st.markdown("---")
    st.subheader("🔍 Global Filters")
    min_date, max_date = df_raw['Prod_Date'].min(), df_raw['Prod_Date'].max()
    date_range = st.date_input("Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    list_sup = ['All'] + sorted(df_raw['Supplier'].unique().tolist())
    sel_sup = st.selectbox("🏭 Supplier:", list_sup)
    
    list_res = ['All'] + sorted(df_raw['Coating_Type'].unique().tolist())
    sel_res = st.selectbox("🧪 Coating Type:", list_res)
    
    list_col = ['All'] + sorted(df_raw['Color_Group'].unique().tolist())
    sel_col = st.selectbox("🎨 Color Group:", list_col)

    with st.expander("🛠️ Custom Paint Specs", expanded=False):
        st.caption("Override control limits for specific paint codes.")
        
        # Check if it doesn't exist OR if it's holding the old Vietnamese column name
        if "custom_gloss_rules" not in st.session_state or "Ma_Son" in st.session_state["custom_gloss_rules"].columns:
            init_data = pd.DataFrame({
                "Paint_Code": ["", "", ""], "Lab_LSL": [0.0, 0.0, 0.0], "Lab_USL": [0.0, 0.0, 0.0],
                "Line_LSL": [0.0, 0.0, 0.0], "Line_USL": [0.0, 0.0, 0.0]
            })
            st.session_state["custom_gloss_rules"] = init_data

        edited_rules = st.data_editor(st.session_state["custom_gloss_rules"], num_rows="dynamic", hide_index=True)
        st.session_state["custom_gloss_rules"] = edited_rules

# ==============================================================================
# DATA FILTERING & PASS/FAIL CALCULATION (VECTORIZED)
# ==============================================================================
STANDARD_LINE_OFFSET = 2.0 
df = df_raw.copy()
df['Line_LSL'] = df['Gloss_LSL'] - STANDARD_LINE_OFFSET
df['Line_USL'] = df['Gloss_USL'] + STANDARD_LINE_OFFSET

custom_df = st.session_state["custom_gloss_rules"].dropna(subset=["Paint_Code"])
custom_df = custom_df[custom_df["Paint_Code"].str.strip() != ""]

if not custom_df.empty:
    custom_df = custom_df.rename(columns={
        'Lab_LSL': 'c_Lab_LSL', 'Lab_USL': 'c_Lab_USL', 
        'Line_LSL': 'c_Line_LSL', 'Line_USL': 'c_Line_USL'
    })
    df = df.merge(custom_df, on='Paint_Code', how='left')
    
    df['Gloss_LSL'] = np.where(df['c_Lab_LSL'].notna() & (df['c_Lab_LSL'] > 0), df['c_Lab_LSL'], df['Gloss_LSL'])
    df['Gloss_USL'] = np.where(df['c_Lab_USL'].notna() & (df['c_Lab_USL'] > 0), df['c_Lab_USL'], df['Gloss_USL'])
    df['Line_LSL'] = np.where(df['c_Line_LSL'].notna() & (df['c_Line_LSL'] > 0), df['c_Line_LSL'], df['Line_LSL'])
    df['Line_USL'] = np.where(df['c_Line_USL'].notna() & (df['c_Line_USL'] > 0), df['c_Line_USL'], df['Line_USL'])
    
    df = df.drop(columns=['c_Lab_LSL', 'c_Lab_USL', 'c_Line_LSL', 'c_Line_USL'])

df['Lab_Pass'] = (df['Gloss_Lab'] >= df['Gloss_LSL']) & (df['Gloss_Lab'] <= df['Gloss_USL'])
df['Line_Pass'] = (df['Online_Gloss_Top'] >= df['Line_LSL']) & (df['Online_Gloss_Top'] <= df['Line_USL'])
df['Gloss_Pass'] = df['Lab_Pass'] & df['Line_Pass']
df['Color_Pass'] = df['ΔE'] <= 1.0
df['Final_Status'] = np.where(df['Gloss_Pass'] & df['Color_Pass'], '✅ PASS', '❌ FAIL/NG')

dff = df.copy()
if len(date_range) == 2:
    dff = dff[(dff['Prod_Date'] >= date_range[0]) & (dff['Prod_Date'] <= date_range[1])]
if sel_sup != 'All': dff = dff[dff['Supplier'] == sel_sup]
if sel_res != 'All': dff = dff[dff['Coating_Type'] == sel_res]
if sel_col != 'All': dff = dff[dff['Color_Group'] == sel_col]

with st.sidebar:
    st.markdown("---")
    st.caption(f"📦 Showing: {len(dff)} coils")

# --- DISPLAY VIEWS ---
st.title(view_mode)
st.markdown("---")

# ==========================================
# TIER 1: EXECUTIVE VIEW
# ==========================================
if view_mode == "Master Summary & Pareto":
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
    
    st.markdown("---")
    st.markdown("### 📉 Pareto Chart: Top NG Contributors")
    df_ng = dff[dff['Final_Status'] == '❌ FAIL/NG'].copy()

    if not df_ng.empty:
        pareto_data = df_ng.groupby(['Paint_Code', 'Supplier']).size().reset_index(name='NG_Count')
        pareto_data = pareto_data.sort_values(by='NG_Count', ascending=False)
        pareto_data['Cum_Percentage'] = pareto_data['NG_Count'].cumsum() / pareto_data['NG_Count'].sum() * 100
        pareto_data_top = pareto_data.head(15)

        fig_pareto, ax1 = plt.subplots(figsize=(14, 6))
        
        sns.barplot(data=pareto_data_top, x='Paint_Code', y='NG_Count', hue='Supplier', dodge=False, ax=ax1, palette='pastel')
        
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, title='Supplier', loc='lower center', 
                   bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True)

        ax1.set_ylabel("NG Coils", fontweight='bold')
        ax1.set_xlabel("Paint Code", fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        ax2.plot(range(len(pareto_data_top)), pareto_data_top['Cum_Percentage'], color='red', marker='D', ms=7, lw=2.5)
        ax2.axhline(80, color='gray', linestyle='--', lw=1.5, label='80% Threshold')
        ax2.set_ylabel("Cumulative Percentage (%)", color='red', fontweight='bold')
        ax2.set_ylim(0, 110) 
        
        for i, txt in enumerate(pareto_data_top['Cum_Percentage']):
            ax2.annotate(f"{txt:.1f}%", (i, txt), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=8, fontweight='bold')

        fig_pareto.tight_layout()
        st.pyplot(fig_pareto)
        plt.close(fig_pareto)
    else:
        st.success("🎉 Great! No NG data recorded in this filtered period.")

# ==========================================
# TIER 2: SUPPLIER INTELLIGENCE 
# ==========================================
elif view_mode == "Supplier Intelligence (Apples-to-Apples)":
    st.info("💡 Logic: Overall Capability (Cpk) ranking via Heatmap. Select a high-risk segment from the list to drill down into Root Cause analysis.")

    df_valid_specs = dff.dropna(subset=['Coating_Type', 'Color_Group', 'Gloss_LSL', 'Gloss_USL', 'Target_Top', 'Target_Primer', 'Avg_DFT']).copy()
    
    df_valid_specs['Gloss_Spec'] = df_valid_specs.apply(lambda r: f"{r['Gloss_LSL']:g}~{r['Gloss_USL']:g}", axis=1)
    df_valid_specs['DFT_Spec'] = df_valid_specs.apply(lambda r: f"{r['Target_Top']:g} + {r['Target_Primer']:g}", axis=1)
    df_valid_specs['Numeric_Target'] = (df_valid_specs['Gloss_LSL'] + df_valid_specs['Gloss_USL']) / 2.0

    if df_valid_specs.empty:
        st.warning("⚠️ Insufficient technical spec data for comparison.")
    else:
        df_scan = df_valid_specs.groupby(['Coating_Type', 'Color_Group', 'Gloss_Spec', 'DFT_Spec']).agg(
            Coils=('Online_Gloss_Top', 'count'),
            Mean_Gloss=('Online_Gloss_Top', 'mean'),
            Std_Gloss=('Online_Gloss_Top', 'std'),
            Line_LSL=('Line_LSL', 'first'),
            Line_USL=('Line_USL', 'first')
        ).reset_index()
        
        df_scan = df_scan[df_scan['Coils'] >= 5].copy()
        
        if df_scan.empty:
            st.warning("⚠️ Min. 5 coils required in a single segment for statistical modeling.")
        else:
            df_scan['Tolerance'] = df_scan['Line_USL'] - df_scan['Line_LSL']
            df_scan['Numeric_Target'] = (df_scan['Line_LSL'] + df_scan['Line_USL']) / 2.0
            df_scan['Std_Gloss'] = df_scan['Std_Gloss'].replace(0, 0.1) 
            
            df_scan['Cp'] = df_scan['Tolerance'] / (6 * df_scan['Std_Gloss'])
            df_scan['Ca (%)'] = (df_scan['Mean_Gloss'] - df_scan['Numeric_Target']) / (df_scan['Tolerance'] / 2) * 100
            df_scan['Cpk'] = df_scan['Cp'] * (1 - df_scan['Ca (%)'].abs() / 100)
            
            df_scan = df_scan.sort_values('Cpk', ascending=True)

            st.write("### 🗺️ Quality Capability Matrix (Cpk Heatmap)")
            st.caption("Correlation between Resin and Color Group. Red/Orange indicates process risk (Cpk < 1.33).")
            
            pivot_cpk = df_scan.pivot_table(index='Coating_Type', columns='Color_Group', values='Cpk', aggfunc='mean')
            st.dataframe(
                pivot_cpk.style.format("{:.2f}", na_rep="-")
                         .background_gradient(cmap='RdYlGn', vmin=0.5, vmax=1.5, axis=None),
                use_container_width=True
            )

            st.markdown("---")
            st.write("### 🔍 Supplier Detailed Analysis (Drill-Down)")
            
            def make_label(r):
                icon = "🔴" if r['Cpk'] < 1.0 else ("🟠" if r['Cpk'] < 1.33 else "🟢")
                return f"{icon} Cpk: {r['Cpk']:.2f} | Resin: {r['Coating_Type']} | Color: {r['Color_Group']} | Gloss: {r['Gloss_Spec']} | DFT: {r['DFT_Spec']} ({r['Coils']} coils)"
                
            df_scan['Smart_Label'] = df_scan.apply(make_label, axis=1)
            sel_label = st.selectbox("🎯 Select priority segment (Auto-sorted by highest risk):", df_scan['Smart_Label'].tolist())
            
            sel_row = df_scan[df_scan['Smart_Label'] == sel_label].iloc[0]
            f_resin = sel_row['Coating_Type']
            f_color = sel_row['Color_Group']
            f_gloss_spec = sel_row['Gloss_Spec']
            f_dft_spec = sel_row['DFT_Spec']

            df_seg = df_valid_specs[(df_valid_specs['Coating_Type']==f_resin) & 
                                    (df_valid_specs['Color_Group']==f_color) & 
                                    (df_valid_specs['Gloss_Spec']==f_gloss_spec) & 
                                    (df_valid_specs['DFT_Spec']==f_dft_spec)].copy()
                                    
            numeric_gloss_target = sel_row['Numeric_Target']
            lsl_val, usl_val = sel_row['Line_LSL'], sel_row['Line_USL']
            tolerance = usl_val - lsl_val

            comp_table = df_seg.groupby('Supplier').agg(
                Coils=('Online_Gloss_Top', 'count'), 
                Mean_Gloss=('Online_Gloss_Top', 'mean'), 
                Std_Gloss=('Online_Gloss_Top', 'std'), 
                Std_DFT=('Avg_DFT', 'std'), 
                Avg_dE=('ΔE', 'mean')
            ).reset_index()

            res_stds = []
            for sup in comp_table['Supplier']:
                sup_df = df_seg[df_seg['Supplier'] == sup]
                if len(sup_df) > 2 and sup_df['Avg_DFT'].std() > 0:
                    model = LinearRegression().fit(sup_df[['Avg_DFT']].values, sup_df['Online_Gloss_Top'].values)
                    res_stds.append((sup_df['Online_Gloss_Top'].values - model.predict(sup_df[['Avg_DFT']].values)).std())
                else: 
                    res_stds.append(sup_df['Online_Gloss_Top'].std() if len(sup_df)>1 else 0)
            comp_table['Paint Instability (Res.Std)'] = res_stds

            def calc_spc(row):
                if row['Std_Gloss'] == 0 or pd.isna(row['Std_Gloss']): return pd.Series([np.nan, np.nan, np.nan])
                cp = tolerance / (6 * row['Std_Gloss'])
                ca = (row['Mean_Gloss'] - numeric_gloss_target) / (tolerance / 2) * 100
                return pd.Series([cp, ca, cp * (1 - abs(ca)/100)])

            comp_table[['Cp', 'Ca (%)', 'Cpk']] = comp_table.apply(calc_spc, axis=1)
            comp_table['Bias'] = comp_table['Mean_Gloss'] - numeric_gloss_target
            comp_table = comp_table.sort_values('Cpk', ascending=False)

            st.markdown("---")
            col_m, col_t = st.columns([3, 2.5])
            
            with col_m:
                st.subheader("📊 Capability Matrix (Accuracy vs Stability)")
                fig_m, ax_m = plt.subplots(figsize=(9, 6))
                ax_m.axhspan(1.33, 2.0, facecolor='#27ae60', alpha=0.3, label='Excellent')
                ax_m.axhspan(1.0, 1.33, facecolor='#f1c40f', alpha=0.3, label='Warning')
                ax_m.axhspan(0, 1.0, facecolor='#c0392b', alpha=0.3, label='High Risk')
                ax_m.axvline(0, color='black', ls='--', lw=2)
                sns.scatterplot(data=comp_table, x='Bias', y='Cpk', hue='Supplier', s=600, edgecolor='black', ax=ax_m, zorder=5)
                
                leg = ax_m.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, markerscale=0.4)
                for h in (leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles):
                    if hasattr(h, 'set_sizes'): h.set_sizes([100])
                    if hasattr(h, 'set_markersize'): h.set_markersize(8)
                st.pyplot(fig_m)
                plt.close(fig_m)
                
            with col_t:
                st.subheader("🏆 Leaderboard")
                st.dataframe(
                    comp_table[['Supplier', 'Coils', 'Cp', 'Cpk', 'Ca (%)', 'Std_DFT', 'Paint Instability (Res.Std)']].style.format({
                        'Cp':'{:.2f}', 
                        'Cpk':'{:.2f}', 
                        'Ca (%)':'{:+.1f}%', 
                        'Std_DFT':'{:.2f}', 
                        'Paint Instability (Res.Std)':'{:.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Cpk', 'Cp']) 
                      .background_gradient(cmap='coolwarm', subset=['Ca (%)'], vmin=-100, vmax=100), 
                    use_container_width=True, hide_index=True
                )

            st.markdown("---")
            st.subheader("🔬 Root Cause Validation: DFT vs Gloss Correlation")
            batch_data = df_seg.groupby(['Supplier', 'Batch_Lot']).agg(Date_Min=('Prod_Date', 'min'), Mean_Gloss=('Online_Gloss_Top', 'mean'), Mean_DFT=('Avg_DFT', 'mean')).reset_index().sort_values('Date_Min')
            batch_data['Bias'] = batch_data['Mean_Gloss'] - numeric_gloss_target
            
            fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
            for idx, sup in enumerate(comp_table['Supplier']):
                sup_d = batch_data[batch_data['Supplier']==sup]
                if len(sup_d) > 1: 
                    sns.regplot(data=sup_d, x='Mean_DFT', y='Bias', label=sup, ax=ax_sc, ci=None, scatter_kws={'s':80, 'alpha':0.7}, line_kws={'lw':2.5})
                else: 
                    sns.scatterplot(data=sup_d, x='Mean_DFT', y='Bias', label=sup, ax=ax_sc, s=80)
                    
            ax_sc.axhline(0, color='black', lw=2)
            ax_sc.axhline(1.5, color='red', ls=':', lw=1.5)
            ax_sc.axhline(-1.5, color='red', ls=':', lw=1.5)
            ax_sc.set_xlabel("Actual Average DFT (µm)", fontweight='bold')
            ax_sc.set_ylabel("Gloss Bias [GU]", fontweight='bold')
            ax_sc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_sc)
            plt.close(fig_sc)

            st.markdown("---")
            st.subheader("📊 X-bar Control Chart (Supplier Stability)")
            fig_x, axes = plt.subplots(len(comp_table['Supplier']), 1, figsize=(14, 4 * len(comp_table['Supplier'])))
            if len(comp_table['Supplier']) == 1: axes = [axes]
            
            for i, sup in enumerate(comp_table['Supplier']):
                sup_d = batch_data[batch_data['Supplier']==sup].copy().reset_index()
                ax = axes[i]
                mu = sup_d['Mean_Gloss'].mean()
                sig = sup_d['Mean_Gloss'].std() if len(sup_d)>1 else 0.1
                
                ax.plot(sup_d.index, sup_d['Mean_Gloss'], marker='o', lw=2, label='Batch Mean')
                ax.axhline(mu, color='green', label=f'Process Mean (μ): {mu:.1f}')
                ax.axhline(mu+3*sig, color='red', ls='--', label=f'UCL: {mu+3*sig:.1f}')
                ax.axhline(mu-3*sig, color='red', ls='--', label=f'LCL: {mu-3*sig:.1f}')
                ax.axhline(usl_val, color='#d35400', ls='-.', lw=2, label=f'Line Max: {usl_val:.1f}')
                ax.axhline(lsl_val, color='#d35400', ls='-.', lw=2, label=f'Line Min: {lsl_val:.1f}')
                ax.set_title(f"Supplier: {sup}", fontweight='bold')
                ax.set_xticks(sup_d.index)
                ax.set_xticklabels(sup_d['Batch_Lot'], rotation=45, ha='right')
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig_x)
            plt.close(fig_x)

# ==========================================
# TIER 3: OPERATIONAL VIEW
# ==========================================
elif view_mode == "Gloss Trend (SPC)":
    st.info("💡 SPC Analysis: Monitor the actual Gloss trend (Lab vs Line) across raw production sequence.")
    
    risk_alert = pd.DataFrame()
    with st.expander("🚨 Early Warning Radar (Click to view at-risk codes)", expanded=True):
        st.caption("This table scans paint codes (≥ 5 Batches) that are Out of Spec (NG) or approaching limits.")
        df_valid_radar = dff.dropna(subset=['Online_Gloss_Top', 'Line_LSL', 'Line_USL', 'Gloss_Lab', 'Gloss_LSL', 'Gloss_USL', 'Batch_Lot'])
        
        if not df_valid_radar.empty:
            risk_summary = df_valid_radar.groupby(['Paint_Code', 'Supplier']).agg(
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
                    risk_alert = risk_alert.sort_values('Status')
                    st.dataframe(risk_alert[['Paint_Code', 'Supplier', 'Batches', 'Coils', 'Issue Source', 'Lab_Min', 'Lab_Max', 'Line_Min', 'Line_Max', 'Status']].style.format({
                        'Lab_Min': '{:.1f}', 'Lab_Max': '{:.1f}', 'Line_Min': '{:.1f}', 'Line_Max': '{:.1f}'
                    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    def render_spc_analysis(paint_code, data_source, key_suffix):
        dff_g = data_source[data_source['Paint_Code'] == paint_code].copy()
        dff_g = dff_g.dropna(subset=['Gloss_LSL', 'Gloss_USL', 'Gloss_Lab', 'Online_Gloss_Top'])
        
        if len(dff_g) <= 1:
            st.warning(f"⚠️ Insufficient data for {paint_code}")
            return

        lsl_val, usl_val = dff_g['Gloss_LSL'].iloc[0], dff_g['Gloss_USL'].iloc[0]
        line_lsl_val, line_usl_val = dff_g['Line_LSL'].iloc[0], dff_g['Line_USL'].iloc[0]
        
        st.success(f"📅 **Timeframe:** `{dff_g['Prod_Date'].min()}` to `{dff_g['Prod_Date'].max()}` | **Volume:** {dff_g['Batch_Lot'].nunique()} Batches ({len(dff_g)} Coils).")

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

        num_labels = len(batch_info)
        step = max(1, num_labels // 12) 
        
        kept_ticks = batch_info['mean'].iloc[::step].tolist()
        kept_labels = batch_info['Batch_Lot'].iloc[::step].astype(str).tolist()
        
        if batch_info['mean'].iloc[-1] not in kept_ticks:
            kept_ticks.append(batch_info['mean'].iloc[-1])
            kept_labels.append(str(batch_info['Batch_Lot'].iloc[-1]))

        ax_trend.set_xticks(kept_ticks)
        ax_trend.set_xticklabels(kept_labels, rotation=45, ha='right', fontsize=8)
        
        ax_trend.set_ylabel("Gloss (GU)")
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
        st.pyplot(fig_trend)
        plt.close(fig_trend)

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
        
        for data, color, label, loop_mean, loop_std in [
            (dff_g['Gloss_Lab'], '#1f77b4', 'Lab', mean_lab, std_lab), 
            (dff_g['Online_Gloss_Top'], '#ff7f0e', 'Line', mean_line, std_line)
        ]:
            if loop_std > 0:
                y_curve = stats.norm.pdf(x_axis, loop_mean, loop_std) * len(data) * bin_width
                ax_dist.plot(x_axis, y_curve, color=color, lw=2, label=f'{label} Curve (σ={loop_std:.2f})')

        ax_dist.set_xlabel("Gloss Value (GU)", fontsize=9)
        ax_dist.set_ylabel("Number of Coils", fontsize=9)
        ax_dist.legend(loc='upper right', fontsize=7, ncol=2)
        ax_dist.grid(axis='y', alpha=0.2)
        
        st.pyplot(fig_dist)
        plt.close('all')

    list_paint_codes_tab2 = sorted(dff['Paint_Code'].dropna().unique().tolist())
    if list_paint_codes_tab2:
        tab_top_risk, tab_custom = st.tabs(["🚨 Top At-Risk Codes", "🔍 Manual Analysis"])
        
        with tab_top_risk:
            if not risk_alert.empty:
                top_15 = risk_alert['Paint_Code'].head(15).tolist()
                for i, code in enumerate(top_15):
                    st.markdown(f"#### #{i+1}: `{code}`")
                    render_spc_analysis(code, dff, f"risk_{i}")
                    st.markdown("---")
            else:
                st.success("✅ All processes are stable.")

        with tab_custom:
            sel_paint_code = st.selectbox("🎯 Select Paint Code:", list_paint_codes_tab2, key="manual_sel")
            render_spc_analysis(sel_paint_code, dff, "manual")

elif view_mode == "Color Shift Analysis":
    st.info("💡 Trend analysis of Total Color Difference (ΔE) and distribution of individual color components (ΔL, Δa, Δb) to detect color drift.")
    
    list_paint_codes_tab3 = sorted(dff['Paint_Code'].dropna().unique().tolist())
    if list_paint_codes_tab3:
        sel_paint_code_tab3 = st.selectbox("🎯 Select Full Paint Code for Color Analysis:", list_paint_codes_tab3, key="tab3_paintcode")
        dff_c = dff[dff['Paint_Code'] == sel_paint_code_tab3].copy()
        
        if not dff_c.empty:
            dff_c_batch = dff_c.groupby('Batch_Lot', as_index=False).agg({'Prod_Date': 'min', 'ΔE': 'mean'}).sort_values('Prod_Date')
            dff_c_batch['Batch_Lot'] = dff_c_batch['Batch_Lot'].astype(str)

            st.markdown("---")
            st.subheader(f"📈 Avg Total Color Difference Trend (ΔE) - {sel_paint_code_tab3}")
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
            
            def plot_norm_hist(data_series, ax, color, title, xlabel):
                data = data_series.dropna()
                if len(data) > 1:
                    mean_val = data.mean()
                    std_val = data.std()
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

elif view_mode == "Statistical Limits (Scope Comparison)":
    st.header("📊 Control Limits: IQR & Sigma Scopes")
    st.info("💡 Determine dynamic control limits based on **Standard Deviation**. Outliers are automatically filtered using the **IQR Method** to ensure accurate baseline calculations.")

    paint_code_list = sorted(dff['Paint_Code'].dropna().unique().tolist())
    if not paint_code_list:
        st.stop()
        
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        search_keyword = st.text_input("🔍 Search Paint Code:", "").upper()
    filtered_list = [code for code in paint_code_list if search_keyword in code]
    with col_s2:
        if filtered_list:
            sel_code = st.selectbox("🎯 Select Paint Code:", filtered_list)
        else:
            st.warning("❌ No paint code found.")
            st.stop()
            
    dff_spc = dff[dff['Paint_Code'] == sel_code].copy().dropna(subset=['Online_Gloss_Top']).sort_values('Prod_Date')
    
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
# TIER 4: DIAGNOSTIC VIEW
# ==========================================
elif view_mode == "Process vs Material (DFT & Root Cause)":
    st.header("📏 Process vs. Material (Thickness Correlation & Residuals)")
    st.caption("Identify Root Cause: Does gloss fluctuate because the paint formulation is unstable (Material Issue), or because the operator applies uneven paint thickness (Process Issue)?")
    
    required_cols = ['DFT_N', 'DFT_S', 'Target_Top', 'Target_Primer']
    missing_cols = [c for c in required_cols if c not in dff.columns]
    
    if missing_cols:
        st.error(f"❌ Cannot perform analysis. Missing columns in dataset: {', '.join(missing_cols)}")
    else:
        st.info("💡 **Dry Film Thickness (DFT)** is the average of North/South edges. It is compared against the **Target DFT** `(Top Paint + Primer) * 90%`.")
        
        dff_dft = dff.dropna(subset=['Color_Group', 'Supplier', 'Online_Gloss_Top', 'Coating_Type', 'Gloss_LSL', 'Gloss_USL', 'Avg_DFT', 'Target_DFT']).copy()
        
        dff_dft['Segment_Name'] = (
            "🎨 " + dff_dft['Color_Group'].astype(str) + " | 🏭 " + dff_dft['Supplier'].astype(str) + 
            " | 🧪 " + dff_dft['Coating_Type'].astype(str) + 
            " | 🌟 Gloss: " + dff_dft['Line_LSL'].apply(lambda x: f"{x:g}") + "~" + dff_dft['Line_USL'].apply(lambda x: f"{x:g}") + " GU" +
            " | 📏 Target DFT: " + dff_dft['Target_DFT'].apply(lambda x: f"{x:.1f}") + " µm"
        )
        
        analysis_level_dft = st.radio(
            "🔍 Select Analysis Level:", 
            ["Detailed View (By Paint Code - Coil Level)", "Macro View (By Product Segment - Batch Average)"], 
            horizontal=True
        )
        
        is_macro = "Macro View" in analysis_level_dft
        
        if is_macro:
            seg_counts = dff_dft.groupby('Segment_Name')['Batch_Lot'].nunique()
            valid_targets = seg_counts[seg_counts >= 3].index.tolist()
            label_text = "🎯 Select Product Segment (Macro View):"
            col_target = 'Segment_Name'
        else:
            valid_targets = dff_dft.groupby("Paint_Code")['Batch_Lot'].nunique()
            valid_targets = valid_targets[valid_targets >= 3].index.tolist()
            label_text = "🎯 Select Paint Code (Detailed View):"
            col_target = 'Paint_Code'
        
        if not valid_targets:
            st.warning("⚠️ Need at least 3 batches of data to run correlation analysis.")
        else:
            sel_target = st.selectbox(label_text, valid_targets, key="dft_target_sel")
            
            df_raw_dft = dff_dft[dff_dft[col_target] == sel_target].copy()
            
            if is_macro:
                df_plot = df_raw_dft.groupby('Batch_Lot').agg(
                    Date_Min=('Prod_Date', 'min'),
                    Online_Gloss_Top=('Online_Gloss_Top', 'mean'),
                    Avg_DFT=('Avg_DFT', 'mean'),
                    Target_DFT=('Target_DFT', 'mean')
                ).sort_values('Date_Min').reset_index()
                x_label_seq = "Production Sequence (Batch Number)"
            else:
                df_plot = df_raw_dft.sort_values('Prod_Date').reset_index(drop=True)
                df_plot['Date_Min'] = df_plot['Prod_Date']
                x_label_seq = "Production Sequence (Coil Number)"
                
            df_plot['x_seq'] = list(range(len(df_plot)))
            df_plot['DFT_Diff'] = df_plot['Avg_DFT'] - df_plot['Target_DFT']

            if len(df_plot) > 5:
                corr_val = df_plot['Avg_DFT'].corr(df_plot['Online_Gloss_Top'])
                r2_val = corr_val**2 if pd.notna(corr_val) else 0.0
                
                avg_actual = df_plot['Avg_DFT'].mean()
                avg_target = df_plot['Target_DFT'].mean()
                avg_diff = df_plot['DFT_Diff'].mean()
                
                st.markdown("---")
                
                c_t1, c_t2, c_t3 = st.columns(3)
                c_t1.metric("📏 Avg Actual DFT", f"{avg_actual:.2f} µm")
                c_t2.metric("🎯 Avg Target DFT", f"{avg_target:.2f} µm")
                
                if avg_diff > 1.0: diff_status = "🔴 Over-thickness (Waste)"
                elif avg_diff < -1.0: diff_status = "🔴 Under-thickness"
                else: diff_status = "🟢 On Target"
                    
                c_t3.metric("⚖️ Avg Deviation", f"{avg_diff:+.2f} µm", diff_status, delta_color="inverse" if abs(avg_diff) > 1.0 else "normal")

                X_val = df_plot[['Avg_DFT']].values
                y_val = df_plot['Online_Gloss_Top'].values
                model = LinearRegression().fit(X_val, y_val)
                
                df_plot['Predicted_Gloss'] = model.predict(X_val)
                df_plot['Residual'] = df_plot['Online_Gloss_Top'] - df_plot['Predicted_Gloss']
                res_std = df_plot['Residual'].std() if len(df_plot) > 1 else 0

                st.markdown("#### 🤖 AI Root-Cause Verdict")
                if r2_val > 0.6:
                    st.error(f"**🔴 PROCESS ISSUE (R² = {r2_val:.2f}):** Gloss strictly follows Film Thickness. The Supplier's paint is fine, but the Line Operator is applying paint unevenly. Fix coater settings!")
                elif res_std > 1.5:
                    st.warning(f"**🟠 MATERIAL ISSUE (Residual Std = {res_std:.2f} GU):** Gloss fluctuates wildly (>{1.5} GU) EVEN AFTER removing the DFT impact. Paint formulation is unstable. Contact Supplier!")
                else:
                    st.info(f"**🟢 MIXED/STABLE FACTORS:** Both process and material are within acceptable statistical control limits.")

                st.markdown("#### 1. Thickness vs Gloss Scatter Correlation")
                fig_dft_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
                sns.regplot(data=df_plot, x='Avg_DFT', y='Online_Gloss_Top', ax=ax_scatter,
                            ci=None, 
                            scatter_kws={'alpha':0.6, 's':60, 'color':'#8e44ad'}, 
                            line_kws={'color':'red', 'lw':2, 'label': f'Trend Line (R²={r2_val:.2f})'})
                ax_scatter.axvline(avg_target, color='green', linestyle='--', lw=2, label=f'Target DFT ({avg_target:.1f})')
                ax_scatter.set_xlabel("Average Dry Film Thickness (µm)", fontweight='bold')
                ax_scatter.set_ylabel("Online Gloss (GU)", fontweight='bold')
                ax_scatter.grid(True, alpha=0.3)
                ax_scatter.legend()
                st.pyplot(fig_dft_scatter)
                plt.close(fig_dft_scatter)

                st.markdown("#### 2. Dual-Axis Production Trend")
                fig_dual, ax1 = plt.subplots(figsize=(14, 5))
                
                color1 = '#e67e22'
                ax1.set_xlabel(x_label_seq, fontweight='bold')
                ax1.set_ylabel('Gloss (GU)', color=color1, fontweight='bold')
                line1 = ax1.plot(df_plot['x_seq'], df_plot['Online_Gloss_Top'], marker='s', color=color1, lw=2, label='Line Gloss')
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.grid(axis='x', alpha=0.3)

                ax2 = ax1.twinx()  
                color2 = '#2980b9'
                ax2.set_ylabel('Thickness (µm)', color=color2, fontweight='bold')
                line2 = ax2.plot(df_plot['x_seq'], df_plot['Avg_DFT'], marker='o', color=color2, lw=2, linestyle='-', label='Actual DFT')
                line3 = ax2.plot(df_plot['x_seq'], df_plot['Target_DFT'], color='green', lw=2.5, linestyle='--', label='Target DFT (90%)')
                ax2.tick_params(axis='y', labelcolor=color2)

                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')

                st.pyplot(fig_dual)
                plt.close(fig_dual)
                
                st.markdown("#### 3. Residual Analysis (Root Cause)")
                st.caption("This chart isolates the Gloss deviation REMAINING after subtracting the DFT impact. If points scatter widely beyond the red lines, it indicates a Material (Paint) instability issue.")
                
                fig_res, ax_res = plt.subplots(figsize=(10, 4))
                ax_res.scatter(df_plot['Avg_DFT'], df_plot['Residual'], color='purple', alpha=0.6, s=60)
                ax_res.axhline(0, color='black', ls='--', lw=2)
                ax_res.axhline(1.5, color='red', ls=':', lw=1.5, label='Risk Threshold (±1.5 GU)')
                ax_res.axhline(-1.5, color='red', ls=':', lw=1.5)
                ax_res.set_xlabel("Average DFT (µm)", fontweight='bold')
                ax_res.set_ylabel("Residual Gloss (GU)", fontweight='bold')
                ax_res.set_title("Residual Plot (Gloss Variation Independent of DFT)", fontweight='bold')
                ax_res.legend()
                st.pyplot(fig_res)
                plt.close(fig_res)

            else:
                st.info("⚠️ Not enough data points to run correlation analysis.")

elif view_mode == "Predictive Compensation & Targeting":
    st.header("⚖️ Predictive Compensation & Lab Optimization")
    st.info("Logic: App learns the historical bias (Loss) per paint code to calculate the 'Theoretical Value' required to hit the exact target specification on the line.")

    paint_code_list = sorted(dff['Paint_Code'].dropna().unique().tolist())
    if paint_code_list:
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            search_keyword = st.text_input("🔍 Search Paint Code to Optimize:", "").upper()
        filtered_list = [code for code in paint_code_list if search_keyword in code]
        
        with col_s2:
            if filtered_list:
                sel_code = st.selectbox("🎯 Select Paint Code:", filtered_list)
            else:
                st.warning("No paint code found.")
                st.stop()

        dff_model = dff[dff['Paint_Code'] == sel_code].dropna(subset=['Online_Gloss_Top', 'Gloss_Lab']).sort_values(['Prod_Date', 'Coil_No']).copy()

        if len(dff_model) >= 5:
            # Shift calculations to Coil-level directly
            dff_model['Loss'] = dff_model['Online_Gloss_Top'] - dff_model['Gloss_Lab']
            
            mean_lab_hist = dff_model['Gloss_Lab'].mean()
            mean_line_hist = dff_model['Online_Gloss_Top'].mean()
            std_lab_hist = dff_model['Gloss_Lab'].std() if dff_model['Gloss_Lab'].std() > 0 else 0.5
            std_line_hist = dff_model['Online_Gloss_Top'].std() if dff_model['Online_Gloss_Top'].std() > 0 else 0.5
            
            actual_visual_bias = mean_line_hist - mean_lab_hist
            std_loss = dff_model['Loss'].std() if dff_model['Loss'].std() > 0 else 0.5
            
            line_lsl = dff_model['Line_LSL'].iloc[0]
            line_usl = dff_model['Line_USL'].iloc[0]
            
            # Batch analysis kept only for sequence plotting at the bottom
            batch_analysis = dff_model.groupby('Batch_Lot').agg({
                'Prod_Date': 'min',
                'Gloss_Lab': 'mean', 
                'Online_Gloss_Top': 'mean',
            }).sort_values('Prod_Date').reset_index()
            
            st.markdown("---")
            st.subheader("🎯 Center Target Definition")
            st.info("Since tolerances can be asymmetric, the arithmetic mean (Max+Min)/2 is not always the true target. Please specify the exact target.")
            
            default_target = (line_lsl + line_usl) / 2.0
            target_line = st.number_input("Line Gloss Target [GU]:", value=float(default_target), step=0.1, help="Input the exact target requested by the customer/spec.")
            
            optimal_lab_input = target_line - actual_visual_bias
            
            icl_lcl = optimal_lab_input - (1 * std_loss)
            icl_ucl = optimal_lab_input + (1 * std_loss)

            st.markdown(f"### 🚀 Optimization Guidance for `{sel_code}`")
            
            col_target, col_guidance = st.columns([1, 2])
            
            with col_target:
                st.metric("Line Gloss Target", f"{target_line:.1f} GU", help="Defined explicitly by user.")
                st.metric("Average System Bias", f"{actual_visual_bias:+.2f} GU", help="Average drift caused by the production line for this specific paint.")
                st.metric("Standard Deviation (Sigma, σ)", f"{std_loss:.2f} GU", help="Calculated variation (σ) of the historical bias. Used to define the Internal Control Limit.")

            with col_guidance:
                st.success(f"#### Required Lab Input (Theoretical Value): **{optimal_lab_input:.1f} GU**")
                st.write(f"To ensure the final product hits the exact target of **{target_line:.1f} GU** on the line, the laboratory should aim for a pre-production mix of **{optimal_lab_input:.1f} GU** to compensate for the process drift.")
                st.warning(f"**Internal Control Limit (ICL): {icl_lcl:.1f} - {icl_ucl:.1f}** *(±1σ, with σ = {std_loss:.2f})*")
                st.caption("Production is only authorized if Lab testing falls within this tightened range (±1σ).")

            st.markdown("---")
            st.subheader("🔔 Bias Distribution Shift (Lab vs. Line)")
            st.caption("Illustrates the systematic offset ('Absolute Bias') between the Assigned Value (Lab) and Achieved Value (Line).")

            fig_bell, ax_bell = plt.subplots(figsize=(12, 5))

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
            ax_bell.text((mean_lab_hist + mean_line_hist)/2, y_annotate + 0.02, f'Absolute Bias: {actual_visual_bias:+.2f} GU', 
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
            ax_model.plot(batch_labels, batch_analysis['Gloss_Lab'], marker='o', ls='--', color='gray', alpha=0.6, label='Actual Lab Input (Avg)')
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
                st.dataframe(dff_model[['Batch_Lot', 'Gloss_Lab', 'Online_Gloss_Top', 'Loss']].tail(10))

        else:
            st.warning("⚠️ Insufficient historical data for this paint code to build a reliable compensation model (Min. 5 coils required).")

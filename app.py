import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 

# --- 1. UI SETUP ---
st.set_page_config(page_title="Steel QA Master Dashboard", layout="wide", page_icon="🏭")

# Custom Theme for Sharpness
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# --- 2. DATA LOAD & PREP ---
@st.cache_data(ttl=300)
def load_and_prep_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1ugm7G1kgGmSlk5PhoKk62h_bs5pX4bDuwUgdaELLYHE/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
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
        df['Ma_Son_Str'] = df['Ma_Son'].astype(str).str.upper().str.strip()

        v_map = {'S':'Yungchi', 'T':'AKZO NOBEL', 'A':'AKZO NOBEL', 'B':'Beckers', 'C':'Nan Pao', 'U':'Quali Poly', 'N':'Nippon', 'K':'Kansai', 'V':'Valspar', 'J':'Valspar (SW)', 'L':'KCC', 'R':'Noroo', 'Q':'Paoqun'}
        r_map = {'1':'PU','2':'PE','3':'EPOXY','4':'PVC','5':'PVDF','6':'SMP','7':'AC','8':'WB','9':'IP','A':'PVB','B':'PVF'}
        c_map = {'0':'Clear','1':'Red','R':'Red','O':'Orange','2':'Orange','Y':'Yellow','3':'Yellow','4':'Green','G':'Green','5':'Blue','L':'Blue','V':'Violet','6':'Violet','N':'Brown','7':'Brown','T':'White','H':'White','W':'White','8':'White','A':'Gray','C':'Gray','9':'Gray','B':'Black','S':'Silver','M':'Metallic'}
        
        df['Supplier'] = df['Ma_Son_Str'].str[1].map(v_map).fillna('Unknown')
        df['Coating_Type'] = df['Ma_Son_Str'].str[2].map(r_map).fillna('Unknown')
        df['Color_Group'] = df['Ma_Son_Str'].str[6].map(c_map).fillna('Other')
        df['Color_Code'] = df['Ma_Son_Str'].str[-4:] 

        num_cols = ['Gloss_Lab', 'G_Top_N', 'G_Top_S', 'G_Back_N', 'G_Back_S', 'dE_N', 'dE_S', 'dL_N', 'da_N', 'db_N', 'Gloss_LSL', 'Gloss_USL']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['Gloss_Lab', 'Ma_Son', 'Gloss_LSL', 'Gloss_USL'])
        df['Ngay_SX'] = pd.to_datetime(df['Ngay_SX'], errors='coerce').dt.date
        df['Online_Gloss_Top'] = df[['G_Top_N', 'G_Top_S']].mean(axis=1)
        df['ΔE'] = df[['dE_N', 'dE_S']].mean(axis=1)
        
        return df.dropna(subset=['Online_Gloss_Top']).sort_values('Ngay_SX')
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
        return pd.DataFrame()

df = load_and_prep_data()
if df.empty: st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    view_mode = st.radio("Select Analysis View:", [
        "✨ Gloss Trend (SPC)",
        "🎨 Color Shift Analysis",
        "📊 Statistical Limits (Scope Comparison)",
        "⚖️ Predictive Compensation & Targeting",
        "🤝 Supplier Capability",
        "📋 Master Summary Report"
    ])
    st.markdown("---")
    min_date, max_date = df['Ngay_SX'].min(), df['Ngay_SX'].max()
    date_range = st.date_input("📅 Date Range:", [min_date, max_date])
    sel_sup = st.selectbox("🏭 Supplier:", ['All'] + sorted(df['Supplier'].unique().tolist()))

# Filtering
dff = df.copy()
if len(date_range) == 2:
    dff = dff[(dff['Ngay_SX'] >= date_range[0]) & (dff['Ngay_SX'] <= date_range[1])]
if sel_sup != 'All': dff = dff[dff['Supplier'] == sel_sup]

# Define Global Colors for Consistency
C_LAB = "#0D47A1"    # Deep Blue
C_LINE = "#E65100"   # Bright Orange
C_SPEC = "#B71C1C"   # Crimson Red
C_TARGET = "#1B5E20" # Forest Green
C_CTRL = "#546E7A"   # Blue Grey (for limits)

st.title(view_mode)
st.markdown("---")

# ==========================================
# VIEW: STATISTICAL LIMITS (SCOPE COMPARISON)
# ==========================================
if view_mode == "📊 Statistical Limits (Scope Comparison)":
    ma_son_list = sorted(dff['Ma_Son_Str'].unique().tolist())
    sel_code = st.selectbox("🎯 Select Paint Code:", ma_son_list)
    dff_spc = dff[dff['Ma_Son_Str'] == sel_code].copy()
    
    if len(dff_spc) >= 5:
        # Simple Stats
        mean_ind = dff_spc['Online_Gloss_Top'].mean()
        std_ind = dff_spc['Online_Gloss_Top'].std()
        lsl, usl = dff_spc['Gloss_LSL'].iloc[0], dff_spc['Gloss_USL'].iloc[0]
        
        st.subheader("📊 Distribution Overlap: Individual vs. Batch")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bell Curve
        x = np.linspace(mean_ind - 4*std_ind, mean_ind + 4*std_ind, 200)
        y = stats.norm.pdf(x, mean_ind, std_ind)
        ax.plot(x, y, color=C_LINE, lw=3, label='Line Distribution')
        ax.fill_between(x, y, alpha=0.2, color=C_LINE)
        
        # Vertical Lines with High Contrast
        ax.axvline(lsl, color=C_SPEC, lw=2.5, ls='-', label=f'Spec LSL ({lsl})')
        ax.axvline(usl, color=C_SPEC, lw=2.5, ls='-', label=f'Spec USL ({usl})')
        ax.axvline(mean_ind, color=C_TARGET, lw=2, ls='--', label=f'Mean ({mean_ind:.1f})')
        
        # Annotations
        y_max = ax.get_ylim()[1]
        ax.text(lsl, y_max*0.9, ' LSL', color=C_SPEC, fontweight='bold')
        ax.text(usl, y_max*0.9, ' USL', color=C_SPEC, fontweight='bold')
        
        ax.set_xlabel("Gloss (GU)")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        st.pyplot(fig)
    else:
        st.warning("Insufficient data.")

# ==========================================
# VIEW: PREDICTIVE COMPENSATION
# ==========================================
elif view_mode == "⚖️ Predictive Compensation & Targeting":
    ma_son_list = sorted(dff['Ma_Son_Str'].unique().tolist())
    sel_code = st.selectbox("🎯 Select Paint Code:", ma_son_list)
    dff_model = dff[dff['Ma_Son_Str'] == sel_code].copy()
    
    if len(dff_model) >= 5:
        # Logic: Loss = Line - Lab
        dff_model['Loss'] = dff_model['Online_Gloss_Top'] - dff_model['Gloss_Lab']
        avg_loss = dff_model['Loss'].mean()
        
        lsl, usl = dff_model['Gloss_LSL'].iloc[0], dff_model['Gloss_USL'].iloc[0]
        center_target = (lsl + usl) / 2
        theoretical_lab = center_target - avg_loss
        
        st.markdown(f"### 🚀 Optimization for `{sel_code}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Bias", f"{avg_loss:+.2f} GU")
        col2.metric("Ideal Line Target", f"{center_target:.1f} GU")
        col3.metric("Theoretical Lab Input", f"{theoretical_lab:.1f} GU", delta_color="normal")
        
        st.subheader("📉 Systematic Drift Pattern (Lab vs. Line)")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        
        # Plotting with new high-contrast colors
        batches = dff_model['Batch_Lot'].astype(str)
        ax2.plot(batches, dff_model['Gloss_Lab'], marker='o', color=C_LAB, lw=2, label='Actual Lab Input', alpha=0.8)
        ax2.plot(batches, dff_model['Online_Gloss_Top'], marker='s', color=C_LINE, lw=2.5, label='Actual Line Output')
        
        # Horizontal Guide Lines
        ax2.axhline(center_target, color=C_TARGET, lw=2, ls='-', label='Ideal Center Target')
        ax2.axhline(theoretical_lab, color=C_TARGET, lw=2, ls=':', label='Theoretical Lab Target')
        ax2.axhline(lsl, color=C_SPEC, lw=1.5, ls='-', alpha=0.6)
        ax2.axhline(usl, color=C_SPEC, lw=1.5, ls='-', alpha=0.6)
        
        # Add Arrows for Bias representation
        for i in range(len(dff_model)):
            ax2.annotate('', xy=(i, dff_model['Online_Gloss_Top'].iloc[i]), 
                         xytext=(i, dff_model['Gloss_Lab'].iloc[i]),
                         arrowprops=dict(arrowstyle='<->', color='gray', lw=1, alpha=0.5))

        plt.xticks(rotation=45, ha='right')
        ax2.set_ylabel("Gloss (GU)")
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        st.pyplot(fig2)
    else:
        st.warning("Insufficient data.")

# ==========================================
# VIEW: GLOSS TREND (SPC) - FULL VERSION
# ==========================================
elif view_mode == "✨ Gloss Trend (SPC)":
    ma_son_list = sorted(dff['Ma_Son_Str'].unique().tolist())
    sel_code = st.selectbox("🎯 Select Paint Code:", ma_son_list)
    dff_trend = dff[dff['Ma_Son_Str'] == sel_code].copy()
    
    if not dff_trend.empty:
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        batches = dff_trend['Batch_Lot'].astype(str)
        
        ax3.plot(batches, dff_trend['Gloss_Lab'], color=C_LAB, marker='o', label='Lab Gloss', lw=2)
        ax3.plot(batches, dff_trend['Online_Gloss_Top'], color=C_LINE, marker='s', label='Line Gloss', lw=2)
        
        # Bold Specs
        lsl, usl = dff_trend['Gloss_LSL'].iloc[0], dff_trend['Gloss_USL'].iloc[0]
        ax3.axhline(lsl, color=C_SPEC, lw=3, label=f'Spec LSL ({lsl})')
        ax3.axhline(usl, color=C_SPEC, lw=3, label=f'Spec USL ({usl})')
        
        plt.xticks(rotation=45, ha='right')
        ax3.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        st.pyplot(fig3)

# ... (Các view khác tương tự với bộ màu C_LAB, C_LINE, C_SPEC) ...

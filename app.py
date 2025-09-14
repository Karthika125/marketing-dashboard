# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide")

@st.cache_data
def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def preprocess_marketing(dfs: dict):
    """Take dict of dataframes for channels, normalize and concat with channel column."""
    rows = []
    for channel, df in dfs.items():
        if df.empty:
            continue
        d = df.copy()
        # normalize column names
        d.columns = [c.strip() for c in d.columns]
        if 'date' in d.columns:
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
        else:
            st.warning(f"{channel} data missing 'date' column")
        # numeric conversions and fillna
        for col in ['impression', 'impressions', 'clicks', 'spend', 'attributed_revenue']:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')
        # harmonize impressions name
        if 'impressions' in d.columns and 'impression' not in d.columns:
            d['impression'] = d['impressions']
        d['channel'] = channel
        rows.append(d)
    if rows:
        all_marketing = pd.concat(rows, ignore_index=True, sort=False)
    else:
        all_marketing = pd.DataFrame()
    # standardize columns that we will use
    for c in ['campaign', 'tactic', 'state']:
        if c not in all_marketing.columns:
            all_marketing[c] = np.nan
    all_marketing['impression'] = all_marketing['impression'].fillna(0)
    all_marketing['clicks'] = all_marketing['clicks'].fillna(0)
    all_marketing['spend'] = all_marketing['spend'].fillna(0)
    all_marketing['attributed_revenue'] = all_marketing.get('attributed_revenue', 0).fillna(0)
    return all_marketing

@st.cache_data
def preprocess_business(df):
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
    numeric_cols = ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'COGS']
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
        else:
            d[col] = 0
    return d

def aggregate_marketing(df_marketing):
    # daily per channel aggregates
    group_cols = ['date', 'channel']
    agg = df_marketing.groupby(group_cols).agg({
        'impression':'sum',
        'clicks':'sum',
        'spend':'sum',
        'attributed_revenue':'sum'
    }).reset_index()
    # metrics
    agg['CTR'] = np.where(agg['impression']>0, agg['clicks']/agg['impression'], 0)
    agg['CPC'] = np.where(agg['clicks']>0, agg['spend']/agg['clicks'], np.nan)
    agg['CPM'] = np.where(agg['impression']>0, agg['spend']/(agg['impression']/1000), np.nan)
    agg['ROAS'] = np.where(agg['spend']>0, agg['attributed_revenue']/agg['spend'], np.nan)
    return agg

def merge_business_marketing(business_df, marketing_agg):
    merged = pd.merge(business_df, marketing_agg, on='date', how='left')
    # fill NaNs for channels aggregated; later filters will split by channel
    merged[['impression','clicks','spend','attributed_revenue']] = merged[['impression','clicks','spend','attributed_revenue']].fillna(0)
    # additional derived metrics
    merged['Orders_per_Click'] = np.where(merged['clicks']>0, merged['orders']/merged['clicks'], np.nan)
    merged['CAC'] = np.where(merged['new_customers']>0, merged['spend']/merged['new_customers'], np.nan)
    return merged

# --- UI ---
st.title("Marketing Intelligence Dashboard — Assessment 1")
st.markdown("Interactive dashboard connecting marketing campaigns (Facebook/Google/TikTok) to business outcomes.")

# Load data
with st.sidebar.expander("1) Upload / Select Data"):
    st.write("Place CSVs in `data/` folder or upload manually.")
    fb = load_csv_safe("data/Facebook.csv")
    ggl = load_csv_safe("data/Google.csv")
    tkt = load_csv_safe("data/TikTok.csv")
    biz = load_csv_safe("data/Business.csv")
    st.write("Loaded datasets:")
    st.write(f"- Facebook rows: {len(fb)}")
    st.write(f"- Google rows: {len(ggl)}")
    st.write(f"- TikTok rows: {len(tkt)}")
    st.write(f"- Business rows: {len(biz)}")

# preprocess
marketing = preprocess_marketing({'Facebook': fb, 'Google': ggl, 'TikTok': tkt})
business = preprocess_business(biz)

if marketing.empty or business.empty:
    st.warning("One or more datasets are empty. Upload data in the data/ folder and refresh.")
    st.stop()

marketing_agg = aggregate_marketing(marketing)
# For channel-agnostic daily totals, sum across channels:
daily_marketing_total = marketing_agg.groupby('date').agg({
    'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'
}).reset_index()
daily_marketing_total['CTR'] = np.where(daily_marketing_total['impression']>0, daily_marketing_total['clicks']/daily_marketing_total['impression'], 0)
daily_marketing_total['ROAS'] = np.where(daily_marketing_total['spend']>0, daily_marketing_total['attributed_revenue']/daily_marketing_total['spend'], np.nan)

# Merge business with daily marketing totals for a high-level view
business_daily = pd.merge(business, daily_marketing_total, on='date', how='left').fillna(0)

# sidebar filters
with st.sidebar.expander("2) Filters"):
    min_date = min(marketing['date'].min(), business['date'].min())
    max_date = max(marketing['date'].max(), business['date'].max())
    date_range = st.date_input("Date range", value=(min_date, max_date))
    chosen_channels = st.multiselect("Channels", options=marketing['channel'].unique().tolist(), default=list(marketing['channel'].unique()))
    chosen_states = st.multiselect("States (optional)", options=marketing['state'].dropna().unique().tolist(), default=[])
    top_n = st.slider("Top N campaigns to show", 5, 50, 10)

# apply filters
start_date, end_date = date_range
mask_marketing = (marketing['date'] >= pd.to_datetime(start_date)) & (marketing['date'] <= pd.to_datetime(end_date)) & (marketing['channel'].isin(chosen_channels))
if chosen_states:
    mask_marketing &= marketing['state'].isin(chosen_states)
marketing_f = marketing[mask_marketing].copy()

# aggregate filtered
marketing_agg_f = aggregate_marketing(marketing_f)
daily_total_f = marketing_agg_f.groupby('date').agg({'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'}).reset_index()
daily_total_f['CTR'] = np.where(daily_total_f['impression']>0, daily_total_f['clicks']/daily_total_f['impression'], 0)
daily_total_f['ROAS'] = np.where(daily_total_f['spend']>0, daily_total_f['attributed_revenue']/daily_total_f['spend'], np.nan)

mask_business = (business['date'] >= pd.to_datetime(start_date)) & (business['date'] <= pd.to_datetime(end_date))
business_f = business[mask_business].copy()
merged_f = pd.merge(business_f, daily_total_f, on='date', how='left').fillna(0)

# KPI row
kpi1_col, kpi2_col, kpi3_col, kpi4_col = st.columns(4)
with kpi1_col:
    total_spend = marketing_f['spend'].sum()
    st.metric("Total Spend", f"₹{total_spend:,.0f}")
with kpi2_col:
    total_attrib_rev = marketing_f['attributed_revenue'].sum()
    st.metric("Attributed Revenue", f"₹{total_attrib_rev:,.0f}", delta=f"{(total_attrib_rev - total_spend):,.0f}")
with kpi3_col:
    total_orders = business_f['orders'].sum()
    st.metric("Total Orders (Business)", f"{int(total_orders)}")
with kpi4_col:
    total_new_customers = business_f['new_customers'].sum()
    cac = total_spend / total_new_customers if total_new_customers>0 else np.nan
    st.metric("CAC (Spend / New Customers)", f"₹{cac:,.2f}" if not np.isnan(cac) else "N/A")

# Time series charts
st.markdown("### Time series: Spend vs Revenue vs Orders")
ts1, ts2 = st.columns([2,1])
with ts1:
    fig = px.line(merged_f, x='date', y=['spend','attributed_revenue','total_revenue'], labels={'value':'Amount','date':'Date'}, title="Daily: Spend vs Attributed Revenue vs Total Revenue")
    st.plotly_chart(fig, use_container_width=True)
with ts2:
    fig2 = px.line(merged_f, x='date', y=['orders','new_customers'], labels={'value':'Count','date':'Date'}, title="Orders & New Customers (Daily)")
    st.plotly_chart(fig2, use_container_width=True)

# Channel comparison
st.markdown("### Channel comparison (aggregated over selected period)")
ch_comp = marketing_f.groupby('channel').agg({'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'}).reset_index()
ch_comp['CTR'] = np.where(ch_comp['impression']>0, ch_comp['clicks']/ch_comp['impression'], 0)
ch_comp['ROAS'] = np.where(ch_comp['spend']>0, ch_comp['attributed_revenue']/ch_comp['spend'], np.nan)
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(ch_comp, x='channel', y='spend', title="Spend by Channel", text='spend')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(ch_comp, x='channel', y='ROAS', title="ROAS by Channel", text='ROAS')
    st.plotly_chart(fig, use_container_width=True)

# Top campaigns
st.markdown("### Top campaigns by Attributed Revenue (selected channels & dates)")
camp_agg = marketing_f.groupby(['channel','campaign']).agg({'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'}).reset_index()
camp_agg['ROAS'] = np.where(camp_agg['spend']>0, camp_agg['attributed_revenue']/camp_agg['spend'], np.nan)
camp_agg = camp_agg.sort_values('attributed_revenue', ascending=False)
st.dataframe(camp_agg.head(top_n).reset_index(drop=True))

# State level performance
if 'state' in marketing.columns and marketing['state'].notnull().any():
    st.markdown("### Performance by State")
    state_agg = marketing_f.groupby('state').agg({'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'}).reset_index()
    state_agg['ROAS'] = np.where(state_agg['spend']>0, state_agg['attributed_revenue']/state_agg['spend'], np.nan)
    fig = px.choropleth(state_agg, locations='state', locationmode='USA-states' if False else None, color='attributed_revenue', hover_data=['spend','ROAS'], title="Attributed Revenue by State (if codes provided)")
    # Note: choropleth needs proper state codes; fallback to bar
    st.plotly_chart(fig, use_container_width=True)
    st.write("If choropleth doesn't render (no geo codes), view bar chart below:")
    fig2 = px.bar(state_agg.sort_values('attributed_revenue', ascending=False).head(20), x='state', y='attributed_revenue', title="Top 20 States by Attributed Revenue")
    st.plotly_chart(fig2, use_container_width=True)

# Campaign trend viewer
st.markdown("### Campaign trend (select campaign)")
campaigns = marketing_f['campaign'].dropna().unique().tolist()
campaign_sel = st.selectbox("Campaign", options=["(all)"] + campaigns)
if campaign_sel != "(all)":
    camp_df = marketing_f[marketing_f['campaign']==campaign_sel].groupby('date').agg({'impression':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'}).reset_index()
    camp_df['CTR'] = np.where(camp_df['impression']>0, camp_df['clicks']/camp_df['impression'], 0)
    fig = px.line(camp_df, x='date', y=['impression','clicks','spend','attributed_revenue'], title=f"{campaign_sel} — daily trends")
    st.plotly_chart(fig, use_container_width=True)

# Insights / suggested next steps
st.sidebar.header("Insights / Next steps")
st.sidebar.markdown("""
- Check campaigns with low ROAS & high spend — consider pausing or optimizing.
- Investigate states with unusually low ROAS or high CAC.
- Compare days when spend rose but orders didn't — possible inefficiency or poor targeting.
- Consider cohort / time-lag attribution (click → order lag) if you have user-level data.
""")

st.markdown("### Notes & Limitations")
st.markdown("""
- Attribution here uses the `attributed_revenue` provided in ad CSVs. If multi-touch attribution is needed, additional modeling is required.
- Aggregation is daily and by channel/campaign. If users require session-level or user-level attribution, a different dataset is needed.
""")

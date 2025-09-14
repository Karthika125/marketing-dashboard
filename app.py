import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    layout="wide",
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}
.insight-box {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    color: #2c3e50;
}
.insight-box h4 {
    color: #1a365d;
    margin-bottom: 0.5rem;
}
.insight-box p, .insight-box li {
    color: #4a5568;
    line-height: 1.5;
}
.insight-box strong {
    color: #2d3748;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare datasets."""
    try:
        # Load marketing data
        df_fb = pd.read_csv('data/Facebook.csv')
        df_google = pd.read_csv('data/Google.csv')
        df_tiktok = pd.read_csv('data/TikTok.csv')
        df_business = pd.read_csv('data/business.csv')
        
        # Standardize column names
        for df in [df_fb, df_google, df_tiktok]:
            df.columns = df.columns.str.lower().str.replace(' ', '_')
        df_business.columns = df_business.columns.str.lower().str.replace(' ', '_').str.replace('#_', 'num_')
        
        # Combine marketing data
        df_marketing = pd.concat([df_fb, df_google, df_tiktok], ignore_index=True)
        
        # Convert dates
        df_marketing['date'] = pd.to_datetime(df_marketing['date'])
        df_business['date'] = pd.to_datetime(df_business['date'])
        
        # Derived metrics
        df_marketing['ctr'] = (df_marketing['clicks'] / df_marketing['impression']).fillna(0)
        df_marketing['cpc'] = (df_marketing['spend'] / df_marketing['clicks']).fillna(0)
        df_marketing['roas'] = (df_marketing['attributed_revenue'] / df_marketing['spend']).fillna(0)
        
        return df_marketing, df_business
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def calculate_kpis(df_marketing, df_business):
    """Calculate key performance indicators."""
    marketing_agg = df_marketing.groupby('date').agg({
        'impression':'sum', 'clicks':'sum', 'spend':'sum', 'attributed_revenue':'sum'
    }).reset_index()
    
    df_combined = pd.merge(marketing_agg, df_business, on='date', how='outer')
    
    total_spend = df_marketing['spend'].sum()
    total_attr_rev = df_marketing['attributed_revenue'].sum()
    total_business_rev = df_business['total_revenue'].sum()
    total_orders = df_business['num_of_orders'].sum()
    total_impr = df_marketing['impression'].sum()
    total_clicks = df_marketing['clicks'].sum()
    
    return {
        'total_spend': total_spend,
        'total_attributed_revenue': total_attr_rev,
        'total_business_revenue': total_business_rev,
        'total_orders': total_orders,
        'overall_roas': total_attr_rev / total_spend if total_spend>0 else 0,
        'overall_cpa': total_spend / total_orders if total_orders>0 else 0,
        'overall_ctr': total_clicks / total_impr if total_impr>0 else 0,
        'df_combined': df_combined
    }

def create_filters(df_marketing):
    """Sidebar filters."""
    st.sidebar.header("🎛️ Filters & Analysis")
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df_marketing['date'].min(), df_marketing['date'].max()),
        min_value=df_marketing['date'].min(),
        max_value=df_marketing['date'].max()
    )
    
    channels = st.sidebar.multiselect(
        "Select Channels",
        options=df_marketing['tactic'].unique(),
        default=df_marketing['tactic'].unique()
    )
    
    top_states = df_marketing.groupby('state')['attributed_revenue'].sum().nlargest(10).index.tolist()
    states = st.sidebar.multiselect(
        "Select States (Top 10 shown)",
        options=top_states,
        default=top_states[:5]
    )
    
    return date_range, channels, states

def create_kpi_overview(kpis):
    """Top-level KPI overview."""
    st.markdown('<div class="main-header"><h1>📊 Marketing Intelligence Dashboard</h1><p>E-commerce Marketing & Business Performance Analytics</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("💰 Total Revenue", f"${kpis['total_business_revenue']:,.0f}", help="Total business revenue")
    col2.metric("📈 Marketing Spend", f"${kpis['total_spend']:,.0f}", help="Total marketing investment")
    col3.metric("🎯 Overall ROAS", f"{kpis['overall_roas']:.2f}", delta=f"{'📈 Strong' if kpis['overall_roas']>3 else '📉 Needs Attention'}", help="Return on Ad Spend")
    col4.metric("🛒 Total Orders", f"{kpis['total_orders']:,.0f}", help="Total number of orders")

def create_channel_metrics(df_marketing):
    """Channel-wise comparison metrics."""
    st.header("📊 Channel Performance Metrics")
    platform_metrics = df_marketing.groupby('tactic').agg({
        'spend':'sum','attributed_revenue':'sum','clicks':'sum','impression':'sum'
    }).reset_index()
    
    platform_metrics['ctr'] = platform_metrics['clicks']/platform_metrics['impression']
    platform_metrics['cpc'] = platform_metrics['spend']/platform_metrics['clicks']
    platform_metrics['roas'] = platform_metrics['attributed_revenue']/platform_metrics['spend']
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    for _, row in platform_metrics.iterrows():
        metrics = {"Total Spend": f"${row['spend']:,.2f}", "ROAS": f"{row['roas']:.2f}", "CTR": f"{row['ctr']:.2%}", "CPC": f"${row['cpc']:.2f}"}
        if row['tactic']=="Facebook":
            with col1: st.subheader("Facebook 📘"); [st.metric(k,v) for k,v in metrics.items()]
        elif row['tactic']=="Google":
            with col2: st.subheader("Google 🎯"); [st.metric(k,v) for k,v in metrics.items()]
        elif row['tactic']=="TikTok":
            with col3: st.subheader("TikTok 🎵"); [st.metric(k,v) for k,v in metrics.items()]
    
    # Metrics chart
    st.subheader("Channel Comparison")
    fig = go.Figure()
    for m,title in zip(['spend','attributed_revenue','roas','ctr','cpc'], ['Spend ($)','Revenue ($)','ROAS','CTR (%)','CPC ($)']):
        fig.add_trace(go.Bar(name=title,x=platform_metrics['tactic'],y=platform_metrics[m], text=platform_metrics[m].round(2), textposition='auto'))
    fig.update_layout(title="Key Metrics by Channel", barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics table
    metrics_display = platform_metrics.copy()
    metrics_display['ctr'] = metrics_display['ctr'].map('{:.2%}'.format)
    metrics_display['roas'] = metrics_display['roas'].map('{:.2f}'.format)
    metrics_display['cpc'] = metrics_display['cpc'].map('${:.2f}'.format)
    metrics_display['spend'] = metrics_display['spend'].map('${:,.0f}'.format)
    metrics_display['attributed_revenue'] = metrics_display['attributed_revenue'].map('${:,.0f}'.format)
    
    st.dataframe(metrics_display.rename(columns={
        'tactic':'Channel','spend':'Total Spend','attributed_revenue':'Attributed Revenue',
        'clicks':'Total Clicks','impression':'Impressions','ctr':'CTR','cpc':'CPC','roas':'ROAS'
    }), hide_index=True)

def create_trend_analysis(kpis):
    st.header("📈 Performance Trends")
    df_combined = kpis['df_combined']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['total_revenue'], name="Business Revenue", line=dict(color="#1f77b4", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['spend'], name="Marketing Spend", line=dict(color="#ff7f0e", width=3)), secondary_y=True)
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Spend ($)", secondary_y=True)
    fig.update_layout(title="Revenue vs Marketing Spend Correlation", hovermode='x unified', height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    col1, col2 = st.columns(2)
    with col1:
        corr = df_combined[['spend','total_revenue']].corr().iloc[0,1]
        st.markdown(f"""<div class="insight-box"><h4>Marketing-Revenue Correlation</h4><p><strong>Correlation Score:</strong> {corr:.3f}</p><p><strong>Interpretation:</strong> {'Strong positive correlation!' if corr>0.7 else 'Moderate correlation' if corr>0.4 else 'Weak correlation'}</p></div>""", unsafe_allow_html=True)
    with col2:
        avg_orders = df_combined['num_of_orders'].mean()
        avg_new_customers = df_combined['new_customers'].mean()
        st.markdown(f"""<div class="insight-box"><h4>Business Health Snapshot</h4><p><strong>Avg Orders:</strong> {avg_orders:.0f}</p><p><strong>Avg New Customers/Day:</strong> {avg_new_customers:.0f}</p><p><strong>Acquisition Rate:</strong> {(avg_new_customers/avg_orders)*100:.1f}%</p></div>""", unsafe_allow_html=True)

def create_channel_tactic_analysis(df_filtered):
    """Creates channel and tactic performance analysis."""
    st.header("Channel & Tactic Performance")
    col_plat, col_tac = st.columns(2)

    with col_plat:
        if not df_filtered.empty:
            # Changed 'platform' to 'tactic' here
            platform_perf = df_filtered.groupby('tactic').agg(
                spend=('spend', 'sum'),
                attributed_revenue=('attributed_revenue', 'sum')
            ).reset_index()
            platform_perf['roas'] = platform_perf['attributed_revenue'] / platform_perf['spend']
            platform_perf = platform_perf.sort_values('roas', ascending=True)

            fig_roas = px.bar(platform_perf, x='roas', y='tactic', orientation='h',
                             title="Return On Ad Spend (ROAS) by Channel",
                             text=platform_perf['roas'].round(2))
            st.plotly_chart(fig_roas, use_container_width=True)
        else:
            st.warning("No data for selected channel filters.")

    with col_tac:
        if not df_filtered.empty:
            # Calculate daily performance by channel
            tactic_perf = df_filtered.groupby(['date', 'tactic']).agg(
                spend=('spend', 'sum'),
                attributed_revenue=('attributed_revenue', 'sum')
            ).reset_index()
            tactic_perf['roas'] = tactic_perf['attributed_revenue'] / tactic_perf['spend']

            # Create time series plot
            fig_tactic = px.line(tactic_perf, x='date', y='roas', 
                                color='tactic', 
                                title="ROAS Trends by Channel Over Time")
            fig_tactic.update_layout(yaxis_title="ROAS")
            st.plotly_chart(fig_tactic, use_container_width=True)
        else:
            st.warning("No data for selected time period.")
def create_channel_comparison(df_filtered):
    """Creates comparison between Facebook, Google and TikTok."""
    st.header("📊 Channel Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_filtered.empty:
            # ROAS Analysis
            channel_perf = df_filtered.groupby('tactic').agg({
                'spend': 'sum',
                'attributed_revenue': 'sum',
                'clicks': 'sum',
                'impression': 'sum'
            }).reset_index()
            
            channel_perf['roas'] = channel_perf['attributed_revenue'] / channel_perf['spend']
            channel_perf['ctr'] = channel_perf['clicks'] / channel_perf['impression']
            
            # ROAS Chart
            fig_roas = px.bar(channel_perf, 
                            x='tactic', 
                            y='roas',
                            title="ROAS by Channel",
                            text=channel_perf['roas'].round(2))
            fig_roas.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_roas, use_container_width=True)
            
            # Metrics Table
            st.subheader("Channel Metrics")
            metrics_table = pd.DataFrame({
                'Channel': channel_perf['tactic'],
                'Spend': channel_perf['spend'].map('${:,.0f}'.format),
                'Revenue': channel_perf['attributed_revenue'].map('${:,.0f}'.format),
                'ROAS': channel_perf['roas'].map('{:.2f}'.format),
                'CTR': channel_perf['ctr'].map('{:.2%}'.format)
            })
            st.dataframe(metrics_table, hide_index=True)
        else:
            st.warning("No data available for the selected filters")
    
    with col2:
        if not df_filtered.empty:
            # Time Series Analysis
            daily_perf = df_filtered.groupby(['date', 'tactic']).agg({
                'spend': 'sum',
                'attributed_revenue': 'sum'
            }).reset_index()
            
            daily_perf['roas'] = daily_perf['attributed_revenue'] / daily_perf['spend']
            
            # ROAS Trends
            fig_trends = px.line(daily_perf, 
                               x='date', 
                               y='roas',
                               color='tactic',
                               title="ROAS Trends Over Time")
            fig_trends.update_layout(yaxis_title="ROAS")
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Channel Share
            total_spend = df_filtered['spend'].sum()
            share_data = df_filtered.groupby('tactic')['spend'].sum().reset_index()
            share_data['share'] = share_data['spend'] / total_spend * 100
            
            fig_share = px.pie(share_data,
                             values='share',
                             names='tactic',
                             title="Channel Share of Spend")
            st.plotly_chart(fig_share, use_container_width=True)
        else:
            st.warning("No time series data available")
def create_platform_comparison(df_marketing):
    """Creates separate analysis sections for Facebook, Google, and TikTok."""
    st.header("🔄 Platform Performance Analysis")

    # Facebook Analysis
    st.subheader("📘 Facebook Performance")
    facebook_data = df_marketing[df_marketing['tactic'] == 'Facebook']
    if not facebook_data.empty:
        # Basic metrics
        metrics = facebook_data.groupby('date').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum'
        }).reset_index()
        
        # Calculate daily ROAS
        metrics['roas'] = metrics['attributed_revenue'] / metrics['spend']
        
        # Show Facebook performance chart
        fig = px.line(metrics, x='date', y=['spend', 'attributed_revenue'], 
                     title='Facebook Performance Over Time',
                     labels={'value': 'Amount ($)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show ROAS trend
        fig_roas = px.line(metrics, x='date', y='roas',
                          title='Facebook ROAS Trend',
                          labels={'roas': 'ROAS', 'date': 'Date'})
        st.plotly_chart(fig_roas, use_container_width=True)

    # Google Analysis
    st.subheader("🎯 Google Performance")
    google_data = df_marketing[df_marketing['tactic'] == 'Google']
    if not google_data.empty:
        # Basic metrics
        metrics = google_data.groupby('date').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum'
        }).reset_index()
        
        # Calculate daily ROAS
        metrics['roas'] = metrics['attributed_revenue'] / metrics['spend']
        
        # Show Google performance chart
        fig = px.line(metrics, x='date', y=['spend', 'attributed revenue'],
                     title='Google Performance Over Time',
                     labels={'value': 'Amount ($)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show ROAS trend
        fig_roas = px.line(metrics, x='date', y='roas',
                          title='Google ROAS Trend',
                          labels={'roas': 'ROAS', 'date': 'Date'})
        st.plotly_chart(fig_roas, use_container_width=True)

    # TikTok Analysis
    st.subheader("🎵 TikTok Performance")
    tiktok_data = df_marketing[df_marketing['tactic'] == 'TikTok']
    if not tiktok_data.empty:
        # Basic metrics
        metrics = tiktok_data.groupby('date').agg({
            'spend': 'sum',
            'attributed revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum'
        }).reset_index()
        
        # Calculate daily ROAS
        metrics['roas'] = metrics['attributed revenue'] / metrics['spend']
        
        # Show TikTok performance chart
        fig = px.line(metrics, x='date', y=['spend', 'attributed revenue'],
                     title='TikTok Performance Over Time',
                     labels={'value': 'Amount ($)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show ROAS trend
        fig_roas = px.line(metrics, x='date', y='roas',
                          title='TikTok ROAS Trend',
                          labels={'roas': 'ROAS', 'date': 'Date'})
        st.plotly_chart(fig_roas, use_container_width=True)
def create_geographic_analysis(df_marketing):
    st.header("🗺️ Geographic Performance")
    state_perf = df_marketing.groupby('state').agg({'spend':'sum','attributed_revenue':'sum','clicks':'sum','impression':'sum'}).reset_index()
    state_perf['roas'] = state_perf['attributed_revenue']/state_perf['spend']
    top_states = state_perf.nlargest(10,'attributed_revenue')
    
    fig_geo = px.bar(top_states, x='state', y='attributed_revenue', color='roas', color_continuous_scale='RdYlGn', title="Top 10 States by Attributed Revenue")
    fig_geo.update_xaxes(tickangle=45)
    st.plotly_chart(fig_geo, use_container_width=True)

def main():
    df_marketing, df_business = load_data()
    date_range, channels, states = create_filters(df_marketing)
    
    df_filtered = df_marketing[
        (df_marketing['date'] >= pd.to_datetime(date_range[0])) &
        (df_marketing['date'] <= pd.to_datetime(date_range[1])) &
        (df_marketing['tactic'].isin(channels)) &
        (df_marketing['state'].isin(states))
    ]
    
    kpis = calculate_kpis(df_filtered, df_business)
    
    create_kpi_overview(kpis)
    create_channel_metrics(df_filtered)
    create_channel_tactic_analysis(df_filtered)  # Add this line
    create_platform_comparison(df_filtered)  # Add this line
    create_channel_comparison(df_filtered)  # Add this line
    create_trend_analysis(kpis)
    create_geographic_analysis(df_filtered)
    
    # Recommendations
    st.header("🎯 Action Items & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="insight-box"><h4>Growth Opportunities</h4><ul><li>Scale high-ROAS channels (ROAS > 3.0)</li><li>Optimize underperforming states</li><li>Increase budget during high-correlation periods</li><li>Focus on new customer acquisition</li></ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="insight-box"><h4>Areas for Improvement</h4><ul><li>Reduce spend on low-ROAS channels</li><li>Improve attribution tracking</li><li>Test new geographic markets</li><li>Optimize creative performance by channel</li></ul></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

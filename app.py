import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from pathlib import Path

st.set_page_config(page_title="General Conference Explorer", layout="wide")

st.title("🎤 General Conference Talk Explorer")

# Load data
@st.cache_data
def load_data():
    csv_path = Path("data/gc_talks.csv")
    if not csv_path.exists():
        st.error("Data file not found. Please run the data pipeline first: `data_pipeline()`")
        return None
    
    df = pd.read_csv(csv_path)
    return df

df = load_data()

if df is None:
    st.stop()

# Process embeddings for visualization
@st.cache_data
def prepare_embeddings(df):
    if 'embedding' not in df.columns:
        st.warning("No embeddings found in data")
        return df
    
    import json
    
    # Parse embeddings from JSON strings
    embeddings = []
    valid_indices = []
    
    for idx, emb_str in enumerate(df['embedding']):
        try:
            if pd.notna(emb_str) and isinstance(emb_str, str):
                emb = np.array(json.loads(emb_str))
                embeddings.append(emb)
                valid_indices.append(idx)
        except (json.JSONDecodeError, ValueError):
            pass
    
    if embeddings:
        # Apply PCA to get 2D representation
        embeddings_array = np.array(embeddings)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings_array)
        
        # Create a copy of df with only valid indices
        df_vis = df.iloc[valid_indices].copy()
        df_vis['PCA_1'] = pca_result[:, 0]
        df_vis['PCA_2'] = pca_result[:, 1]
        
        return df_vis
    else:
        st.warning("Could not parse embeddings from data")
        return df

df_vis = prepare_embeddings(df)

# Create tabs
tab1, tab2 = st.tabs(["📊 Explorer", "🔍 Keyword Analysis"])

# ======= TAB 1: EXPLORER =======
with tab1:
    # Sidebar controls
    st.sidebar.header("Controls")

    # Color by selector
    color_options = ['author', 'role', 'year', 'month']
    color_by = st.sidebar.selectbox(
        "Color points by:",
        color_options,
        index=1  # Default to 'role'
    )

    # Title filter
    title_filter = st.sidebar.text_input(
        "Filter by title (contains):",
        placeholder="e.g., 'faith', 'love'"
    )

    # Year filter
    st.sidebar.subheader("Filter by Year")
    available_years = sorted(df_vis['year'].dropna().unique())
    selected_years = st.sidebar.multiselect(
        "Select years:",
        options=available_years,
        default=available_years
    )

    # Month filter
    st.sidebar.subheader("Filter by Month")
    available_months = sorted(df_vis['month'].dropna().unique())
    selected_months = st.sidebar.multiselect(
        "Select months:",
        options=available_months,
        default=available_months
    )

    # Role filter
    st.sidebar.subheader("Filter by Role")
    available_roles = sorted(df_vis['role'].dropna().unique())
    selected_roles = st.sidebar.multiselect(
        "Select roles:",
        options=available_roles,
        default=available_roles
    )

    # Apply filters
    df_filtered = df_vis.copy()
    
    # Title filter
    if title_filter:
        df_filtered = df_filtered[df_filtered['title'].str.contains(title_filter, case=False, na=False)]
    
    # Year filter
    df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]
    
    # Month filter
    df_filtered = df_filtered[df_filtered['month'].isin(selected_months)]
    
    # Role filter
    df_filtered = df_filtered[df_filtered['role'].isin(selected_roles)]
    
    st.sidebar.info(f"Showing {len(df_filtered)} of {len(df_vis)} talks")

    # Create scatterplot
    if 'PCA_1' in df_filtered.columns and 'PCA_2' in df_filtered.columns:
        fig = px.scatter(
            df_filtered,
            x='PCA_1',
            y='PCA_2',
            color=color_by,
            hover_name='title',
            hover_data={
                'author': True,
                'role': True,
                'year': True,
                'month': True,
                'PCA_1': ':.2f',
                'PCA_2': ':.2f'
            },
            title=f"General Conference Talks (colored by {color_by})",
            labels={
                'PCA_1': 'First Principal Component',
                'PCA_2': 'Second Principal Component'
            },
            height=700,
            opacity=0.7
        )
        
        fig.update_layout(
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create visualization. Ensure embeddings are available.")

    # Data table
    st.sidebar.header("Data Summary")

    if st.sidebar.checkbox("Show data table", value=False):
        st.subheader("Filtered Data")
        
        # Select columns to display
        display_cols = ['title', 'author', 'role', 'year', 'month']
        display_cols = [col for col in display_cols if col in df_filtered.columns]
        
        st.dataframe(
            df_filtered[display_cols],
            use_container_width=True,
            height=400
        )

    # Statistics
    st.sidebar.header("Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Talks", len(df_filtered))

    with col2:
        st.metric("Unique Speakers", df_filtered['author'].nunique())

    with col3:
        st.metric("Unique Roles", df_filtered['role'].nunique())

    # Role breakdown
    st.sidebar.subheader("Speakers by Role")
    role_counts = df_filtered['role'].value_counts()
    st.sidebar.bar_chart(role_counts)

# ======= TAB 2: KEYWORD ANALYSIS =======
with tab2:
    st.subheader("Keyword Frequency Analysis by Year")
    
    # Keyword input
    keyword = st.text_input(
        "Enter a keyword to analyze:",
        placeholder="e.g., 'faith', 'love', 'Christ'"
    )
    
    if keyword:
        keyword_lower = keyword.lower()
        
        # Count keyword occurrences in each talk
        df['keyword_matches'] = df['text'].fillna('').str.lower().str.count(keyword_lower)
        
        # Group by year and sum keyword occurrences
        keyword_by_year = df.groupby('year')['keyword_matches'].sum().reset_index()
        keyword_by_year.columns = ['Year', 'Keyword Frequency']
        keyword_by_year['Year'] = keyword_by_year['Year'].astype(str)
        
        if len(keyword_by_year) > 0:
            # Create histogram
            fig = px.bar(
                keyword_by_year,
                x='Year',
                y='Keyword Frequency',
                title=f"Frequency of '{keyword}' by Year",
                labels={'Keyword Frequency': f"'{keyword}' Count"},
                height=500,
                color='Keyword Frequency',
                color_continuous_scale='blues'
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=f"Keyword Frequency",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional statistics
            col1, col2, col3 = st.columns(3)
            
            total_freq = keyword_by_year['Keyword Frequency'].sum()
            talks_with_keyword = (df['keyword_matches'] > 0).sum()
            avg_per_talk = total_freq / talks_with_keyword if talks_with_keyword > 0 else 0
            
            with col1:
                st.metric("Total Occurrences", int(total_freq))
            
            with col2:
                st.metric("Talks Containing Keyword", int(talks_with_keyword))
            
            with col3:
                st.metric("Avg per Talk", f"{avg_per_talk:.2f}")
            
            # Show talks with most mentions
            st.subheader("Top 10 Talks with Most Mentions")
            top_talks = df[df['keyword_matches'] > 0].nlargest(10, 'keyword_matches')[['title', 'author', 'year', 'keyword_matches']]
            top_talks.columns = ['Title', 'Author', 'Year', 'Mentions']
            st.dataframe(top_talks, use_container_width=True)
        else:
            st.warning("No data available for this keyword.")
    else:
        st.info("Enter a keyword to get started.")

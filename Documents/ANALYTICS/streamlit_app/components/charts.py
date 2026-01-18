"""
Composants de graphiques réutilisables avec Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# Couleurs du thème Brésil
BRAZIL_COLORS = {
    'green': '#009739',
    'yellow': '#FEDD00',
    'blue': '#002776',
    'palette': ['#009739', '#FEDD00', '#002776', '#F5F5F5']
}

def create_kpi_chart(title, value, icon="", delta=None, prefix="", suffix=""):
    """
    Crée une carte KPI stylisée en HTML.
    
    Args:
        title: Titre de la métrique
        value: Valeur à afficher (peut être str ou number)
        icon: Emoji ou icône à afficher (optionnel, pour compatibilité)
        delta: Variation (nombre, optionnel)
        prefix: Préfixe pour la valeur (optionnel)
        suffix: Suffixe pour la valeur (optionnel)
    
    Returns:
        HTML string de la carte KPI
    """
    # Formater la valeur
    if isinstance(value, (int, float)):
        display_value = f"{prefix}{value:,.0f}{suffix}"
    else:
        display_value = f"{prefix}{value}{suffix}"
    
    # Créer le HTML de la carte
    html = f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    '>
        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
        <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>{title}</div>
        <div style='font-size: 1.8rem; font-weight: bold;'>{display_value}</div>
    </div>
    """
    return html

def create_line_chart(df, x, y, title, color=None):
    """Crée un graphique linéaire"""
    
    fig = px.line(
        df, 
        x=x, 
        y=y,
        color=color,
        title=title,
        template='plotly_white'
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_bar_chart(df, x, y, title, orientation='v', color_column=None):
    """Crée un graphique en barres"""
    
    fig = px.bar(
        df,
        x=x if orientation == 'v' else y,
        y=y if orientation == 'v' else x,
        title=title,
        color=color_column,
        orientation=orientation,
        color_discrete_sequence=[BRAZIL_COLORS['green']],
        template='plotly_white'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green'],
        showlegend=False
    )
    
    fig.update_traces(
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.8
    )
    
    return fig

def create_pie_chart(df, values, names, title):
    """Crée un graphique en camembert"""
    
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3,
        template='plotly_white'
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )
    
    fig.update_layout(
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_heatmap(df, x, y, z, title):
    """Crée une heatmap"""
    
    fig = px.density_heatmap(
        df,
        x=x,
        y=y,
        z=z,
        title=title,
        color_continuous_scale='Greens',
        template='plotly_white'
    )
    
    fig.update_layout(
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_gauge_chart(value, title, max_value=100, color='green'):
    """Crée une jauge (gauge)"""
    
    colors = {
        'green': BRAZIL_COLORS['green'],
        'yellow': BRAZIL_COLORS['yellow'],
        'blue': BRAZIL_COLORS['blue']
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': BRAZIL_COLORS['green']}},
        delta={'reference': max_value * 0.8},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': colors.get(color, BRAZIL_COLORS['green'])},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font=dict(family='Arial', size=12),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_scatter_plot(df, x, y, title, size=None, color=None):
    """Crée un nuage de points"""
    
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        title=title,
        color_continuous_scale='Greens',
        template='plotly_white'
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        opacity=0.7
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_funnel_chart(df, x, y, title):
    """Crée un graphique en entonnoir"""
    
    fig = px.funnel(
        df,
        x=x,
        y=y,
        title=title,
        color_discrete_sequence=[BRAZIL_COLORS['green']],
        template='plotly_white'
    )
    
    fig.update_layout(
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_area_chart(df, x, y, title):
    """Crée un graphique en aire"""
    
    fig = px.area(
        df,
        x=x,
        y=y,
        title=title,
        color_discrete_sequence=[BRAZIL_COLORS['green']],
        template='plotly_white'
    )
    
    fig.update_traces(
        line=dict(width=3),
        fillcolor='rgba(0, 151, 57, 0.3)'
    )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_box_plot(df, x, y, title):
    """Crée un box plot"""
    
    fig = px.box(
        df,
        x=x,
        y=y,
        title=title,
        color_discrete_sequence=[BRAZIL_COLORS['green']],
        template='plotly_white'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green']
    )
    
    return fig

def create_geo_map(df, locations, values, title):
    """Crée une carte géographique du Brésil"""
    
    fig = px.choropleth(
        df,
        locations=locations,
        locationmode='country names',
        color=values,
        title=title,
        color_continuous_scale='Greens',
        template='plotly_white'
    )
    
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type='mercator'
    )
    
    fig.update_layout(
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_font_color=BRAZIL_COLORS['green'],
        height=600
    )
    
    return fig

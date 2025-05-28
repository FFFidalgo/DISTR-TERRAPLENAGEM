"""
Módulo de visualização para resultados de otimização
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Tuple


class OptimizationVisualizer:
    """Classe para criação de visualizações da otimização"""
    
    @staticmethod
    def create_distance_distribution_chart(allocations_df: pd.DataFrame):
        """
        Cria gráfico de distribuição de distâncias
        
        Args:
            allocations_df: DataFrame com alocações
        """
        if allocations_df.empty:
            st.warning("Nenhum dado para visualização.")
            return
        
        st.subheader("📏 Distribuição de Distâncias de Transporte")
        
        # Histograma de distâncias
        fig = px.histogram(
            allocations_df,
            x='distancia',
            color='tipo_material',
            title='Distribuição das Distâncias de Transporte',
            labels={'distancia': 'Distância (m)', 'count': 'Frequência'},
            nbins=20
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot por tipo de material
        fig_box = px.box(
            allocations_df,
            x='tipo_material',
            y='distancia',
            title='Distribuição de Distâncias por Tipo de Material',
            labels={'distancia': 'Distância (m)', 'tipo_material': 'Tipo de Material'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    @staticmethod
    def create_volume_distribution_chart(allocations_df: pd.DataFrame):
        """
        Cria gráfico de distribuição de volumes
        
        Args:
            allocations_df: DataFrame com alocações
        """
        if allocations_df.empty:
            st.warning("Nenhum dado para visualização.")
            return
        
        st.subheader("📦 Distribuição de Volumes Transportados")
        
        # Gráfico de pizza por tipo de material
        volume_by_material = allocations_df.groupby('tipo_material')['volume'].sum()
        
        fig_pie = px.pie(
            values=volume_by_material.values,
            names=volume_by_material.index,
            title='Distribuição de Volume por Tipo de Material'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Histograma de volumes
        fig_hist = px.histogram(
            allocations_df,
            x='volume',
            color='tipo_material',
            title='Distribuição dos Volumes de Transporte',
            labels={'volume': 'Volume (m³)', 'count': 'Frequência'},
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def create_transport_network_diagram(
        allocations_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        show_top_n: int = 20
    ):
        """
        Cria diagrama de rede de transporte
        
        Args:
            allocations_df: DataFrame com alocações
            origins_df: DataFrame das origens
            destinations_df: DataFrame dos destinos
            show_top_n: Número máximo de alocações a mostrar
        """
        if allocations_df.empty:
            st.warning("Nenhum dado para visualização da rede.")
            return
        
        st.subheader("🔗 Rede de Transporte")
        
        # Selecionar top N alocações por volume
        top_allocations = allocations_df.nlargest(show_top_n, 'volume')
        
        # Preparar dados para o gráfico de rede
        fig = go.Figure()
        
        # Adicionar origens
        for idx, origin in origins_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[origin['Centro de Massa (m)']],
                y=[1],  # Linha das origens
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='blue',
                    symbol='square'
                ),
                text=f"O{idx}",
                textposition='top center',
                name='Origens',
                showlegend=(idx == origins_df.index[0]),
                hovertemplate=f"Origem {idx}<br>Centro: {origin['Centro de Massa (m)']}m<br>Volume: {origin['Volume disponível (m³)']}m³<br>ISC: {origin['ISC']}<extra></extra>"
            ))
        
        # Adicionar destinos
        for idx, dest in destinations_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[dest['Centro de Massa (m)']],
                y=[0],  # Linha dos destinos
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='circle'
                ),
                text=f"D{idx}",
                textposition='bottom center',
                name='Destinos',
                showlegend=(idx == destinations_df.index[0]),
                hovertemplate=f"Destino {idx}<br>Centro: {dest['Centro de Massa (m)']}m<br>CFT: {dest.get('Volume CFT (m³)', 0)}m³<br>CA: {dest.get('Volume CA (m³)', 0)}m³<extra></extra>"
            ))
        
        # Adicionar conexões (alocações)
        for _, alloc in top_allocations.iterrows():
            origin_pos = origins_df.loc[alloc['origem'], 'Centro de Massa (m)']
            dest_pos = destinations_df.loc[alloc['destino'], 'Centro de Massa (m)']
            
            # Cor baseada no tipo de material
            color = 'green' if alloc['tipo_material'] == 'CFT' else 'orange'
            
            # Espessura baseada no volume
            width = max(1, min(10, alloc['volume'] / 100))
            
            fig.add_trace(go.Scatter(
                x=[origin_pos, dest_pos],
                y=[1, 0],
                mode='lines',
                line=dict(color=color, width=width),
                name=f"{alloc['tipo_material']}",
                showlegend=False,
                hovertemplate=f"O{alloc['origem']} → D{alloc['destino']}<br>Material: {alloc['tipo_material']}<br>Volume: {alloc['volume']:.2f}m³<br>Distância: {alloc['distancia']:.2f}m<extra></extra>"
            ))
        
        fig.update_layout(
            title=f'Rede de Transporte (Top {len(top_allocations)} Alocações)',
            xaxis_title='Posição (m)',
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Destinos', 'Origens']),
            height=400,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legenda
        st.write("**Legenda:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔵 Quadrados azuis: Origens")
            st.write("🟢 Linhas verdes: Transporte CFT")
        with col2:
            st.write("🔴 Círculos vermelhos: Destinos")
            st.write("🟠 Linhas laranjas: Transporte CA")
    
    @staticmethod
    def create_cost_efficiency_chart(allocations_df: pd.DataFrame):
        """
        Cria gráfico de eficiência de custos
        
        Args:
            allocations_df: DataFrame com alocações
        """
        if allocations_df.empty:
            st.warning("Nenhum dado para análise de eficiência.")
            return
        
        st.subheader("💡 Análise de Eficiência de Custos")
        
        # Calcular custo por unidade de volume
        allocations_df['custo_por_m3'] = allocations_df['custo_transporte'] / allocations_df['volume']
        
        # Scatter plot: Volume vs Distância vs Custo
        fig = px.scatter(
            allocations_df,
            x='distancia',
            y='volume',
            size='custo_transporte',
            color='tipo_material',
            title='Relação Volume vs Distância vs Custo de Transporte',
            labels={
                'distancia': 'Distância (m)',
                'volume': 'Volume (m³)',
                'custo_transporte': 'Custo de Transporte (m³·m)'
            },
            hover_data=['origem', 'destino', 'custo_por_m3']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de custo por m³
        fig_efficiency = px.box(
            allocations_df,
            x='tipo_material',
            y='custo_por_m3',
            title='Eficiência de Custo por Tipo de Material (Custo por m³)',
            labels={'custo_por_m3': 'Custo por m³ (m)', 'tipo_material': 'Tipo de Material'}
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    @staticmethod
    def create_geographic_overview(
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        allocations_df: pd.DataFrame = None
    ):
        """
        Cria visualização geográfica overview
        
        Args:
            origins_df: DataFrame das origens
            destinations_df: DataFrame dos destinos
            allocations_df: DataFrame das alocações (opcional)
        """
        st.subheader("🗺️ Visão Geográfica do Projeto")
        
        # Criar gráfico com origens e destinos
        fig = go.Figure()
        
        # Adicionar origens
        fig.add_trace(go.Scatter(
            x=origins_df['Centro de Massa (m)'],
            y=[1] * len(origins_df),
            mode='markers+text',
            marker=dict(
                size=origins_df['Volume disponível (m³)'] / 100,  # Tamanho proporcional ao volume
                color='blue',
                symbol='square',
                opacity=0.7
            ),
            text=[f"O{idx}" for idx in origins_df.index],
            textposition='top center',
            name='Origens',
            hovertemplate='Origem %{text}<br>Posição: %{x}m<br>Volume: %{marker.size}m³<extra></extra>'
        ))
        
        # Adicionar destinos
        cft_volumes = destinations_df['Volume CFT (m³)'].fillna(0)
        ca_volumes = destinations_df['Volume CA (m³)'].fillna(0)
        total_volumes = cft_volumes + ca_volumes
        
        fig.add_trace(go.Scatter(
            x=destinations_df['Centro de Massa (m)'],
            y=[0] * len(destinations_df),
            mode='markers+text',
            marker=dict(
                size=total_volumes / 50,  # Tamanho proporcional ao volume necessário
                color='red',
                symbol='circle',
                opacity=0.7
            ),
            text=[f"D{idx}" for idx in destinations_df.index],
            textposition='bottom center',
            name='Destinos',
            hovertemplate='Destino %{text}<br>Posição: %{x}m<br>Volume Total: %{marker.size}m³<extra></extra>'
        ))
        
        # Se há alocações, mostrar fluxos principais
        if allocations_df is not None and not allocations_df.empty:
            # Mostrar apenas as top 10 alocações por volume
            top_flows = allocations_df.nlargest(10, 'volume')
            
            for _, flow in top_flows.iterrows():
                origin_pos = origins_df.loc[flow['origem'], 'Centro de Massa (m)']
                dest_pos = destinations_df.loc[flow['destino'], 'Centro de Massa (m)']
                
                fig.add_trace(go.Scatter(
                    x=[origin_pos, dest_pos],
                    y=[1, 0],
                    mode='lines',
                    line=dict(
                        color='green' if flow['tipo_material'] == 'CFT' else 'orange',
                        width=max(1, flow['volume'] / 200)
                    ),
                    showlegend=False,
                    hovertemplate=f"Fluxo: {flow['volume']:.2f}m³<br>Material: {flow['tipo_material']}<extra></extra>"
                ))
        
        fig.update_layout(
            title='Distribuição Geográfica de Origens e Destinos',
            xaxis_title='Posição no Projeto (m)',
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Destinos', 'Origens'],
                range=[-0.2, 1.2]
            ),
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_summary_dashboard(
        allocations_df: pd.DataFrame,
        summary: Dict[str, Any],
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame
    ):
        """
        Cria dashboard resumido com métricas principais
        
        Args:
            allocations_df: DataFrame com alocações
            summary: Dicionário com resumo
            origins_df: DataFrame das origens
            destinations_df: DataFrame dos destinos
        """
        st.subheader("📊 Dashboard Executivo")
        
        # Criar subplot com múltiplos gráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volume por Material',
                'Distribuição de Distâncias',
                'Utilização das Origens',
                'Eficiência de Custo'
            ),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        if not allocations_df.empty:
            # 1. Volume por material (Pie chart)
            volume_by_material = allocations_df.groupby('tipo_material')['volume'].sum()
            fig.add_trace(
                go.Pie(
                    values=volume_by_material.values,
                    labels=volume_by_material.index,
                    name="Volume"
                ),
                row=1, col=1
            )
            
            # 2. Distribuição de distâncias (Histogram)
            fig.add_trace(
                go.Histogram(
                    x=allocations_df['distancia'],
                    name="Distâncias",
                    nbinsx=15
                ),
                row=1, col=2
            )
            
            # 3. Utilização das origens (Bar chart)
            utilization_data = []
            for idx, origin in origins_df.iterrows():
                used = allocations_df[allocations_df['origem'] == idx]['volume'].sum()
                available = origin['Volume disponível (m³)']
                utilization_data.append({
                    'origem': f"O{idx}",
                    'utilizacao_pct': (used / available * 100) if available > 0 else 0
                })
            
            util_df = pd.DataFrame(utilization_data)
            fig.add_trace(
                go.Bar(
                    x=util_df['origem'],
                    y=util_df['utilizacao_pct'],
                    name="Utilização %"
                ),
                row=2, col=1
            )
            
            # 4. Eficiência de custo (Scatter)
            allocations_df['custo_por_m3'] = allocations_df['custo_transporte'] / allocations_df['volume']
            fig.add_trace(
                go.Scatter(
                    x=allocations_df['distancia'],
                    y=allocations_df['custo_por_m3'],
                    mode='markers',
                    name="Eficiência"
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Dashboard de Resultados")
        st.plotly_chart(fig, use_container_width=True)


class InteractiveAnalysis:
    """Classe para análises interativas"""
    
    @staticmethod
    def create_what_if_analysis(
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        allocations_df: pd.DataFrame
    ):
        """
        Cria análise "What-If" interativa
        
        Args:
            origins_df: DataFrame das origens
            destinations_df: DataFrame dos destinos
            allocations_df: DataFrame das alocações
        """
        st.subheader("🔍 Análise What-If")
        
        st.write("Explore diferentes cenários alterando parâmetros:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por distância
            max_distance_filter = st.slider(
                "Distância máxima permitida (m)",
                min_value=int(allocations_df['distancia'].min()) if not allocations_df.empty else 0,
                max_value=int(allocations_df['distancia'].max()) if not allocations_df.empty else 10000,
                value=int(allocations_df['distancia'].max()) if not allocations_df.empty else 5000,
                step=100
            )
        
        with col2:
            # Filtro por volume mínimo
            min_volume_filter = st.slider(
                "Volume mínimo por alocação (m³)",
                min_value=0.0,
                max_value=float(allocations_df['volume'].max()) if not allocations_df.empty else 1000.0,
                value=0.0,
                step=10.0
            )
        
        # Aplicar filtros
        filtered_allocations = allocations_df[
            (allocations_df['distancia'] <= max_distance_filter) &
            (allocations_df['volume'] >= min_volume_filter)
        ]
        
        # Mostrar impacto
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_volume = allocations_df['volume'].sum()
            filtered_volume = filtered_allocations['volume'].sum()
            volume_impact = (filtered_volume / original_volume * 100) if original_volume > 0 else 0
            st.metric(
                "Volume Restante",
                f"{filtered_volume:.2f} m³",
                delta=f"{volume_impact:.1f}% do original"
            )
        
        with col2:
            original_cost = allocations_df['custo_transporte'].sum()
            filtered_cost = filtered_allocations['custo_transporte'].sum()
            cost_impact = (filtered_cost / original_cost * 100) if original_cost > 0 else 0
            st.metric(
                "Custo Restante", 
                f"{filtered_cost:.2f} m³·m",
                delta=f"{cost_impact:.1f}% do original"
            )
        
        with col3:
            original_count = len(allocations_df)
            filtered_count = len(filtered_allocations)
            count_impact = (filtered_count / original_count * 100) if original_count > 0 else 0
            st.metric(
                "Alocações Restantes",
                filtered_count,
                delta=f"{count_impact:.1f}% do original"
            )
        
        # Gráfico comparativo
        if not filtered_allocations.empty:
            comparison_data = pd.DataFrame({
                'Cenário': ['Original', 'Filtrado'],
                'Volume Total': [
                    allocations_df['volume'].sum(),
                    filtered_allocations['volume'].sum()
                ],
                'Custo Total': [
                    allocations_df['custo_transporte'].sum(),
                    filtered_allocations['custo_transporte'].sum()
                ],
                'Número de Alocações': [
                    len(allocations_df),
                    len(filtered_allocations)
                ]
            })
            
            fig = px.bar(
                comparison_data,
                x='Cenário',
                y=['Volume Total', 'Custo Total'],
                title='Comparação de Cenários',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

"""
Módulo para exibição de resultados da otimização
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import numpy as np

from config import DATA_CONFIG


class ResultsDisplay:
    """Classe para exibição de resultados de otimização"""
    
    @staticmethod
    def display_optimization_summary(summary: Dict[str, Any], status: str):
        """
        Exibe resumo da otimização
        
        Args:
            summary: Dicionário com estatísticas resumidas
            status: Status da otimização
        """
        st.subheader("📊 Resumo da Otimização")
        
        # Status da otimização
        if status == 'Optimal':
            st.success(f"✅ Otimização concluída com sucesso - Status: {status}")
        else:
            st.error(f"❌ Otimização falhou - Status: {status}")
            return
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Volume Total Transportado",
                f"{summary['total_volume_transported']:.2f} m³"
            )
        
        with col2:
            st.metric(
                "Distância Média",
                f"{summary['average_distance']:.2f} m"
            )
        
        with col3:
            st.metric(
                "Número de Alocações",
                summary['num_allocations']
            )
        
        with col4:
            st.metric(
                "Custo Total de Transporte",
                f"{summary['total_distance_weighted']:.2f} m³·m"
            )
        
        # Breakdown por tipo de material
        st.write("**Distribuição por Tipo de Material:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Volume CFT",
                f"{summary['volumes_by_material']['CFT']:.2f} m³"
            )
        
        with col2:
            st.metric(
                "Volume CA", 
                f"{summary['volumes_by_material']['CA']:.2f} m³"
            )
    
    @staticmethod
    def display_allocations_table(allocations_df: pd.DataFrame, max_rows: int = 20):
        """
        Exibe tabela de alocações com filtros
        
        Args:
            allocations_df: DataFrame com alocações
            max_rows: Número máximo de linhas a exibir inicialmente
        """
        st.subheader("📋 Detalhes das Alocações")
        
        if allocations_df.empty:
            st.warning("Nenhuma alocação encontrada.")
            return
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            material_filter = st.selectbox(
                "Filtrar por Material",
                ["Todos"] + list(allocations_df['tipo_material'].unique()),
                key="material_filter"
            )
        
        with col2:
            min_volume = st.number_input(
                "Volume Mínimo (m³)",
                min_value=0.0,
                value=0.0,
                key="min_volume_filter"
            )
        
        with col3:
            max_distance = st.number_input(
                "Distância Máxima (m)",
                min_value=0.0,
                value=float(allocations_df['distancia'].max()) if not allocations_df.empty else 10000.0,
                key="max_distance_filter"
            )
        
        # Aplicar filtros
        filtered_df = allocations_df.copy()
        
        if material_filter != "Todos":
            filtered_df = filtered_df[filtered_df['tipo_material'] == material_filter]
        
        filtered_df = filtered_df[filtered_df['volume'] >= min_volume]
        filtered_df = filtered_df[filtered_df['distancia'] <= max_distance]
        
        # Ordenação
        sort_by = st.selectbox(
            "Ordenar por",
            ['custo_transporte', 'volume', 'distancia', 'origem', 'destino'],
            key="sort_by"
        )
        
        sort_ascending = st.checkbox("Ordem crescente", value=False, key="sort_order")
        
        filtered_df = filtered_df.sort_values(sort_by, ascending=sort_ascending)
        
        # Exibir estatísticas dos dados filtrados
        st.write(f"**Exibindo {len(filtered_df)} de {len(allocations_df)} alocações**")
        
        if len(filtered_df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Volume Total Filtrado", f"{filtered_df['volume'].sum():.2f} m³")
            
            with col2:
                st.metric("Distância Média Filtrada", f"{filtered_df['distancia'].mean():.2f} m")
            
            with col3:
                st.metric("Custo Total Filtrado", f"{filtered_df['custo_transporte'].sum():.2f} m³·m")
        
        # Tabela principal
        st.dataframe(
            filtered_df.head(max_rows),
            use_container_width=True,
            column_config={
                'origem': st.column_config.NumberColumn('Origem', format='%d'),
                'destino': st.column_config.NumberColumn('Destino', format='%d'),
                'tipo_material': st.column_config.TextColumn('Material'),
                'volume': st.column_config.NumberColumn('Volume (m³)', format='%.2f'),
                'distancia': st.column_config.NumberColumn('Distância (m)', format='%.2f'),
                'custo_transporte': st.column_config.NumberColumn('Custo (m³·m)', format='%.2f')
            }
        )
        
        if len(filtered_df) > max_rows:
            st.info(f"Exibindo apenas as primeiras {max_rows} linhas. Use os filtros para refinar a visualização.")
    
    @staticmethod
    def display_origin_utilization(
        allocations_df: pd.DataFrame, 
        origins_df: pd.DataFrame
    ):
        """
        Exibe análise de utilização das origens
        
        Args:
            allocations_df: DataFrame com alocações
            origins_df: DataFrame original das origens
        """
        st.subheader("🏗️ Análise de Utilização das Origens")
        
        # Calcular utilização por origem
        utilization_data = []
        
        for idx, origin in origins_df.iterrows():
            origin_allocations = allocations_df[allocations_df['origem'] == idx]
            used_volume = origin_allocations['volume'].sum()
            available_volume = origin['Volume disponível (m³)']
            utilization_pct = (used_volume / available_volume * 100) if available_volume > 0 else 0
            
            utilization_data.append({
                'origem': idx,
                'tipo': origin.get('Tipo', 'Desconhecido'),
                'volume_disponivel': available_volume,
                'volume_utilizado': used_volume,
                'volume_restante': available_volume - used_volume,
                'utilizacao_pct': utilization_pct,
                'isc': origin['ISC']
            })
        
        util_df = pd.DataFrame(utilization_data)
        
        # Métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_utilization = util_df['utilizacao_pct'].mean()
            st.metric("Utilização Média", f"{avg_utilization:.1f}%")
        
        with col2:
            full_utilization = len(util_df[util_df['utilizacao_pct'] >= 99])
            st.metric("Origens Totalmente Utilizadas", full_utilization)
        
        with col3:
            unused_origins = len(util_df[util_df['utilizacao_pct'] == 0])
            st.metric("Origens Não Utilizadas", unused_origins)
        
        with col4:
            total_remaining = util_df['volume_restante'].sum()
            st.metric("Volume Total Restante", f"{total_remaining:.2f} m³")
        
        # Tabela de utilização
        st.write("**Detalhes por Origem:**")
        st.dataframe(
            util_df,
            use_container_width=True,
            column_config={
                'origem': st.column_config.NumberColumn('Origem', format='%d'),
                'tipo': st.column_config.TextColumn('Tipo'),
                'volume_disponivel': st.column_config.NumberColumn('Disponível (m³)', format='%.2f'),
                'volume_utilizado': st.column_config.NumberColumn('Utilizado (m³)', format='%.2f'),
                'volume_restante': st.column_config.NumberColumn('Restante (m³)', format='%.2f'),
                'utilizacao_pct': st.column_config.ProgressColumn('Utilização (%)', min_value=0, max_value=100),
                'isc': st.column_config.NumberColumn('ISC', format='%.1f')
            }
        )
        
        return util_df
    
    @staticmethod
    def display_destination_fulfillment(
        allocations_df: pd.DataFrame,
        destinations_df: pd.DataFrame
    ):
        """
        Exibe análise de atendimento dos destinos
        
        Args:
            allocations_df: DataFrame com alocações
            destinations_df: DataFrame original dos destinos
        """
        st.subheader("🎯 Análise de Atendimento dos Destinos")
        
        # Calcular atendimento por destino
        fulfillment_data = []
        
        for idx, dest in destinations_df.iterrows():
            dest_allocations = allocations_df[allocations_df['destino'] == idx]
            
            # CFT
            cft_needed = dest.get('Volume CFT (m³)', 0) or 0
            cft_received = dest_allocations[dest_allocations['tipo_material'] == 'CFT']['volume'].sum()
            cft_fulfillment = (cft_received / cft_needed * 100) if cft_needed > 0 else 100
            
            # CA
            ca_needed = dest.get('Volume CA (m³)', 0) or 0
            ca_received = dest_allocations[dest_allocations['tipo_material'] == 'CA']['volume'].sum()
            ca_fulfillment = (ca_received / ca_needed * 100) if ca_needed > 0 else 100
            
            fulfillment_data.append({
                'destino': idx,
                'centro_massa': dest['Centro de Massa (m)'],
                'isc_min_exigido': dest['ISC mínimo exigido'],
                'cft_necessario': cft_needed,
                'cft_recebido': cft_received,
                'cft_atendimento_pct': cft_fulfillment,
                'ca_necessario': ca_needed,
                'ca_recebido': ca_received,
                'ca_atendimento_pct': ca_fulfillment,
                'total_recebido': cft_received + ca_received
            })
        
        fulfill_df = pd.DataFrame(fulfillment_data)
        
        # Métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_cft_fulfillment = fulfill_df['cft_atendimento_pct'].mean()
            st.metric("Atendimento Médio CFT", f"{avg_cft_fulfillment:.1f}%")
        
        with col2:
            avg_ca_fulfillment = fulfill_df['ca_atendimento_pct'].mean()
            st.metric("Atendimento Médio CA", f"{avg_ca_fulfillment:.1f}%")
        
        with col3:
            fully_served = len(fulfill_df[
                (fulfill_df['cft_atendimento_pct'] >= 99) & 
                (fulfill_df['ca_atendimento_pct'] >= 99)
            ])
            st.metric("Destinos Totalmente Atendidos", fully_served)
        
        with col4:
            total_delivered = fulfill_df['total_recebido'].sum()
            st.metric("Volume Total Entregue", f"{total_delivered:.2f} m³")
        
        # Tabela de atendimento
        st.write("**Detalhes por Destino:**")
        st.dataframe(
            fulfill_df,
            use_container_width=True,
            column_config={
                'destino': st.column_config.NumberColumn('Destino', format='%d'),
                'centro_massa': st.column_config.NumberColumn('Centro Massa (m)', format='%.2f'),
                'isc_min_exigido': st.column_config.NumberColumn('ISC Mín', format='%.1f'),
                'cft_necessario': st.column_config.NumberColumn('CFT Necessário (m³)', format='%.2f'),
                'cft_recebido': st.column_config.NumberColumn('CFT Recebido (m³)', format='%.2f'),
                'cft_atendimento_pct': st.column_config.ProgressColumn('CFT Atendimento (%)', min_value=0, max_value=100),
                'ca_necessario': st.column_config.NumberColumn('CA Necessário (m³)', format='%.2f'),
                'ca_recebido': st.column_config.NumberColumn('CA Recebido (m³)', format='%.2f'),
                'ca_atendimento_pct': st.column_config.ProgressColumn('CA Atendimento (%)', min_value=0, max_value=100),
                'total_recebido': st.column_config.NumberColumn('Total Recebido (m³)', format='%.2f')
            }
        )
        
        return fulfill_df
    
    @staticmethod
    def display_cost_analysis(allocations_df: pd.DataFrame):
        """
        Exibe análise de custos de transporte
        
        Args:
            allocations_df: DataFrame com alocações
        """
        st.subheader("💰 Análise de Custos de Transporte")
        
        if allocations_df.empty:
            st.warning("Nenhuma alocação para análise de custos.")
            return
        
        # Estatísticas de custo
        total_cost = allocations_df['custo_transporte'].sum()
        avg_cost = allocations_df['custo_transporte'].mean()
        max_cost = allocations_df['custo_transporte'].max()
        min_cost = allocations_df['custo_transporte'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Custo Total", f"{total_cost:.2f} m³·m")
        
        with col2:
            st.metric("Custo Médio por Alocação", f"{avg_cost:.2f} m³·m")
        
        with col3:
            st.metric("Maior Custo Individual", f"{max_cost:.2f} m³·m")
        
        with col4:
            st.metric("Menor Custo Individual", f"{min_cost:.2f} m³·m")
        
        # Análise por tipo de material
        st.write("**Custos por Tipo de Material:**")
        
        cost_by_material = allocations_df.groupby('tipo_material').agg({
            'custo_transporte': ['sum', 'mean', 'count'],
            'volume': 'sum',
            'distancia': 'mean'
        }).round(2)
        
        cost_by_material.columns = ['Custo Total', 'Custo Médio', 'Num Alocações', 'Volume Total', 'Distância Média']
        st.dataframe(cost_by_material, use_container_width=True)
        
        # Top 10 alocações mais caras
        st.write("**Top 10 Alocações Mais Caras:**")
        top_costly = allocations_df.nlargest(10, 'custo_transporte')[
            ['origem', 'destino', 'tipo_material', 'volume', 'distancia', 'custo_transporte']
        ]
        st.dataframe(top_costly, use_container_width=True)


class PerformanceMetrics:
    """Classe para exibição de métricas de performance"""
    
    @staticmethod
    def display_optimization_performance(optimization_history: List[Dict]):
        """
        Exibe métricas de performance das otimizações
        
        Args:
            optimization_history: Lista com histórico de otimizações
        """
        if not optimization_history:
            st.info("Nenhum histórico de otimização disponível.")
            return
        
        st.subheader("⚡ Performance da Otimização")
        
        # Última otimização
        last_opt = optimization_history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tempo de Execução", f"{last_opt['execution_time']:.2f}s")
        
        with col2:
            st.metric("Status", last_opt['status'])
        
        with col3:
            st.metric("Número de Origens", last_opt['num_origins'])
        
        with col4:
            st.metric("Número de Destinos", last_opt['num_destinations'])
        
        # Histórico de tempos de execução
        if len(optimization_history) > 1:
            st.write("**Histórico de Performance:**")
            
            history_df = pd.DataFrame(optimization_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            fig = px.line(
                history_df, 
                x='timestamp', 
                y='execution_time',
                title='Tempo de Execução ao Longo do Tempo',
                labels={'execution_time': 'Tempo (s)', 'timestamp': 'Data/Hora'}
            )
            st.plotly_chart(fig, use_container_width=True)

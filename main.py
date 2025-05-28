"""
Aplicação principal de otimização de distribuição de terraplenagem
Versão refatorada e otimizada

Autor: Sistema de Otimização Automatizado
Data: 2025
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import traceback

# Imports dos módulos personalizados
from config import UI_CONFIG, OPTIMIZATION_CONFIG, ERROR_MESSAGES
from utils.data_validator import DataValidator, DataSanitizer
from optimization.optimizer import TerraplenagemOptimizer
from optimization.scipy_optimizer import ScipyOptimizer
from optimization.distance_calculator import DistanceCalculator
from ui.data_input import DataInputHandler, DataExportHandler
from ui.results_display import ResultsDisplay, PerformanceMetrics
from ui.visualization import OptimizationVisualizer, InteractiveAnalysis


# Configuração da página
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TerraplenageM:
    """Classe principal da aplicação"""
    
    def __init__(self):
        """Inicializa a aplicação"""
        self.initialize_session_state()
        
        # Tentar usar otimizador PuLP primeiro, depois scipy como fallback
        try:
            self.optimizer = TerraplenagemOptimizer()
            self.optimizer_type = "PuLP"
        except Exception as e:
            logger.warning(f"Problema com otimizador PuLP: {e}")
            self.optimizer = ScipyOptimizer()
            self.optimizer_type = "SciPy"
    
    def initialize_session_state(self):
        """Inicializa variáveis de estado da sessão"""
        if 'origins_df' not in st.session_state:
            st.session_state.origins_df = None
        
        if 'destinations_df' not in st.session_state:
            st.session_state.destinations_df = None
        
        if 'optimization_result' not in st.session_state:
            st.session_state.optimization_result = None
        
        if 'optimization_params' not in st.session_state:
            st.session_state.optimization_params = {}
        
        if 'page' not in st.session_state:
            st.session_state.page = 'entrada_dados'
    
    def run(self):
        """Executa a aplicação principal"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            
        except Exception as e:
            logger.error(f"Erro na aplicação principal: {str(e)}")
            st.error(f"Erro inesperado: {str(e)}")
            st.error("Verifique os logs para mais detalhes.")
            
            # Mostrar traceback em modo debug
            if st.checkbox("Mostrar detalhes técnicos do erro"):
                st.code(traceback.format_exc())
    
    def render_header(self):
        """Renderiza o cabeçalho da aplicação"""
        st.title("🏗️ " + UI_CONFIG['page_title'])
        st.markdown("""
        **Sistema avançado de otimização para distribuição de materiais em projetos de terraplenagem**
        
        Esta aplicação utiliza algoritmos de programação linear para encontrar a distribuição ótima 
        de materiais entre origens (cortes e empréstimos) e destinos (aterros), minimizando custos 
        de transporte enquanto atende todas as restrições técnicas.
        """)
        
        # Indicador de status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            origins_status = "✅ Carregadas" if st.session_state.origins_df is not None else "⏳ Pendente"
            st.metric("Origens", origins_status)
        
        with col2:
            destinations_status = "✅ Carregadas" if st.session_state.destinations_df is not None else "⏳ Pendente"
            st.metric("Destinos", destinations_status)
        
        with col3:
            opt_status = "✅ Concluída" if st.session_state.optimization_result is not None else "⏳ Pendente"
            st.metric("Otimização", opt_status)
        
        st.divider()
    
    def render_sidebar(self):
        """Renderiza a barra lateral com navegação e opções"""
        st.sidebar.title("🧭 Navegação")
        
        # Menu principal
        page_options = {
            'entrada_dados': '📥 Entrada de Dados',
            'validacao': '✅ Validação',
            'otimizacao': '⚙️ Otimização',
            'resultados': '📊 Resultados',
            'visualizacao': '📈 Visualização',
            'exportacao': '💾 Exportação'
        }
        
        selected_page = st.sidebar.radio(
            "Selecione a etapa:",
            list(page_options.keys()),
            format_func=lambda x: page_options[x],
            index=list(page_options.keys()).index(st.session_state.page)
        )
        
        st.session_state.page = selected_page
        
        st.sidebar.divider()
        
        # Informações da sessão
        st.sidebar.subheader("ℹ️ Informações da Sessão")
        
        if st.session_state.origins_df is not None:
            st.sidebar.write(f"**Origens:** {len(st.session_state.origins_df)} registros")
        
        if st.session_state.destinations_df is not None:
            st.sidebar.write(f"**Destinos:** {len(st.session_state.destinations_df)} registros")
        
        if st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            st.sidebar.write(f"**Status:** {result.get('status', 'Desconhecido')}")
            if 'execution_time' in result:
                st.sidebar.write(f"**Tempo:** {result['execution_time']:.2f}s")
        
        # Botões de ação rápida
        st.sidebar.divider()
        st.sidebar.subheader("🔧 Ações Rápidas")
        
        if st.sidebar.button("🔄 Limpar Sessão"):
            self.clear_session()
            st.rerun()
        
        if st.sidebar.button("📋 Exemplo de Dados"):
            self.load_example_data()
            st.rerun()
        
        # Configurações avançadas
        with st.sidebar.expander("⚙️ Configurações Avançadas"):
            st.write("**Parâmetros de Otimização:**")
            st.write(f"Tempo limite: {OPTIMIZATION_CONFIG['default_time_limit']}s")
            st.write(f"Solver: {OPTIMIZATION_CONFIG['solver']}")
            
            if st.button("Restaurar Configurações Padrão"):
                st.session_state.optimization_params = {}
                st.success("Configurações restauradas!")
    
    def render_main_content(self):
        """Renderiza o conteúdo principal baseado na página selecionada"""
        page = st.session_state.page
        
        if page == 'entrada_dados':
            self.render_data_input_page()
        elif page == 'validacao':
            self.render_validation_page()
        elif page == 'otimizacao':
            self.render_optimization_page()
        elif page == 'resultados':
            self.render_results_page()
        elif page == 'visualizacao':
            self.render_visualization_page()
        elif page == 'exportacao':
            self.render_export_page()
    
    def render_data_input_page(self):
        """Renderiza a página de entrada de dados"""
        st.header("📥 Entrada de Dados")
        
        tab1, tab2, tab3 = st.tabs(["📁 Upload de Arquivos", "✏️ Entrada Manual", "📋 Exemplos"])
        
        with tab1:
            self.render_file_upload_section()
        
        with tab2:
            self.render_manual_input_section()
        
        with tab3:
            self.render_examples_section()
    
    def render_file_upload_section(self):
        """Renderiza seção de upload de arquivos"""
        st.subheader("📁 Carregamento de Arquivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados das Origens**")
            origins_file = DataInputHandler.create_file_uploader(
                "Carregar arquivo de origens",
                "Arquivo com dados de cortes e empréstimos",
                key="origins_upload"
            )
            
            if origins_file is not None:
                st.session_state.origins_df = origins_file
                DataInputHandler.display_data_preview(origins_file, "Prévia - Origens")
        
        with col2:
            st.write("**Dados dos Destinos**")
            destinations_file = DataInputHandler.create_file_uploader(
                "Carregar arquivo de destinos",
                "Arquivo com dados de aterros",
                key="destinations_upload"
            )
            
            if destinations_file is not None:
                st.session_state.destinations_df = destinations_file
                DataInputHandler.display_data_preview(destinations_file, "Prévia - Destinos")
    
    def render_manual_input_section(self):
        """Renderiza seção de entrada manual"""
        st.subheader("✏️ Entrada Manual de Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Criação Manual de Origens**")
            manual_origins = DataInputHandler.create_manual_data_entry_form('origins')
            
            if manual_origins is not None:
                st.session_state.origins_df = manual_origins
                st.success("Dados de origens criados com sucesso!")
                DataInputHandler.display_data_preview(manual_origins, "Origens Criadas")
        
        with col2:
            st.write("**Criação Manual de Destinos**")
            manual_destinations = DataInputHandler.create_manual_data_entry_form('destinations')
            
            if manual_destinations is not None:
                st.session_state.destinations_df = manual_destinations
                st.success("Dados de destinos criados com sucesso!")
                DataInputHandler.display_data_preview(manual_destinations, "Destinos Criados")
    
    def render_examples_section(self):
        """Renderiza seção de exemplos"""
        st.subheader("📋 Dados de Exemplo")
        
        st.write("""
        Carregue dados de exemplo para testar a aplicação rapidamente.
        """)
        
        if st.button("🎯 Carregar Exemplo Simples"):
            self.load_simple_example()
            st.success("Dados de exemplo carregados!")
            st.rerun()
        
        if st.button("🎯 Carregar Exemplo Complexo"):
            self.load_complex_example()
            st.success("Dados de exemplo complexo carregados!")
            st.rerun()
    
    def render_validation_page(self):
        """Renderiza a página de validação"""
        st.header("✅ Validação de Dados")
        
        if st.session_state.origins_df is None or st.session_state.destinations_df is None:
            st.warning("⚠️ Carregue os dados de origens e destinos antes de validar.")
            return
        
        # Validação das origens
        st.subheader("🏗️ Validação das Origens")
        origins_valid = DataInputHandler.validate_and_display_errors(
            st.session_state.origins_df, 'origins'
        )
        
        # Validação dos destinos
        st.subheader("🎯 Validação dos Destinos")
        destinations_valid = DataInputHandler.validate_and_display_errors(
            st.session_state.destinations_df, 'destinations'
        )
        
        # Verificação de factibilidade
        st.subheader("🔍 Verificação de Factibilidade")
        if origins_valid and destinations_valid:
            is_feasible, message = DataValidator.check_feasibility(
                st.session_state.origins_df,
                st.session_state.destinations_df
            )
            
            if is_feasible:
                st.success(f"✅ {message}")
                
                # Mostrar estatísticas resumidas
                self.display_data_statistics()
                
            else:
                st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Corrija os erros de validação antes de verificar a factibilidade.")
    
    def render_optimization_page(self):
        """Renderiza a página de otimização"""
        st.header("⚙️ Otimização")
        
        if st.session_state.origins_df is None or st.session_state.destinations_df is None:
            st.warning("⚠️ Carregue e valide os dados antes de executar a otimização.")
            return
        
        # Parâmetros de otimização
        optimization_params = DataInputHandler.create_optimization_parameters_form()
        
        if optimization_params:
            st.session_state.optimization_params = optimization_params
        
        # Botão de otimização
        if st.button("🚀 Executar Otimização", type="primary"):
            self.run_optimization()
        
        # Mostrar histórico de otimizações
        if hasattr(self.optimizer, 'optimization_history') and self.optimizer.optimization_history:
            PerformanceMetrics.display_optimization_performance(
                self.optimizer.optimization_history
            )
    
    def render_results_page(self):
        """Renderiza a página de resultados"""
        st.header("📊 Resultados da Otimização")
        
        if st.session_state.optimization_result is None:
            st.warning("⚠️ Execute a otimização primeiro para ver os resultados.")
            return
        
        result = st.session_state.optimization_result
        
        if not result['success']:
            st.error(f"❌ Otimização falhou: {result.get('error', 'Erro desconhecido')}")
            return
        
        # Exibir resumo
        ResultsDisplay.display_optimization_summary(
            result['summary'], 
            result['status']
        )
        
        # Exibir tabela de alocações
        ResultsDisplay.display_allocations_table(result['allocations'])
        
        # Análise de utilização das origens
        util_df = ResultsDisplay.display_origin_utilization(
            result['allocations'],
            st.session_state.origins_df
        )
        
        # Análise de atendimento dos destinos
        fulfill_df = ResultsDisplay.display_destination_fulfillment(
            result['allocations'],
            st.session_state.destinations_df
        )
        
        # Análise de custos
        ResultsDisplay.display_cost_analysis(result['allocations'])
    
    def render_visualization_page(self):
        """Renderiza a página de visualização"""
        st.header("📈 Visualização de Resultados")
        
        if st.session_state.optimization_result is None:
            st.warning("⚠️ Execute a otimização primeiro para ver as visualizações.")
            return
        
        result = st.session_state.optimization_result
        
        if not result['success']:
            st.error("❌ Não é possível gerar visualizações - otimização falhou.")
            return
        
        allocations_df = result['allocations']
        
        # Dashboard executivo
        OptimizationVisualizer.create_summary_dashboard(
            allocations_df,
            result['summary'],
            st.session_state.origins_df,
            st.session_state.destinations_df
        )
        
        # Visualizações específicas
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌐 Rede de Transporte", 
            "📏 Distribuições", 
            "💰 Análise de Custos",
            "🔍 Análise What-If"
        ])
        
        with tab1:
            OptimizationVisualizer.create_transport_network_diagram(
                allocations_df,
                st.session_state.origins_df,
                st.session_state.destinations_df
            )
            
            OptimizationVisualizer.create_geographic_overview(
                st.session_state.origins_df,
                st.session_state.destinations_df,
                allocations_df
            )
        
        with tab2:
            OptimizationVisualizer.create_distance_distribution_chart(allocations_df)
            OptimizationVisualizer.create_volume_distribution_chart(allocations_df)
        
        with tab3:
            OptimizationVisualizer.create_cost_efficiency_chart(allocations_df)
        
        with tab4:
            InteractiveAnalysis.create_what_if_analysis(
                st.session_state.origins_df,
                st.session_state.destinations_df,
                allocations_df
            )
    
    def render_export_page(self):
        """Renderiza a página de exportação"""
        st.header("💾 Exportação de Resultados")
        
        if st.session_state.optimization_result is None:
            st.warning("⚠️ Execute a otimização primeiro para exportar os resultados.")
            return
        
        result = st.session_state.optimization_result
        
        if not result['success']:
            st.error("❌ Não é possível exportar - otimização falhou.")
            return
        
        st.subheader("📁 Opções de Exportação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Exportações Individuais:**")
            
            # Exportar apenas alocações
            DataExportHandler.create_download_button(
                result['allocations'],
                "alocacoes_otimizadas",
                file_format='xlsx',
                button_text="📊 Baixar Alocações (Excel)"
            )
            
            DataExportHandler.create_download_button(
                result['allocations'],
                "alocacoes_otimizadas",
                file_format='csv',
                button_text="📄 Baixar Alocações (CSV)"
            )
        
        with col2:
            st.write("**Exportação Completa:**")
            
            # Exportar tudo
            DataExportHandler.export_complete_results(
                result['allocations'],
                result['summary'],
                st.session_state.origins_df,
                st.session_state.destinations_df,
                filename=f"terraplenagem_resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Informações do resultado
        st.subheader("ℹ️ Informações do Resultado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Alocações", len(result['allocations']))
        
        with col2:
            st.metric("Volume Total", f"{result['summary']['total_volume_transported']:.2f} m³")
        
        with col3:
            st.metric("Tempo de Otimização", f"{result.get('execution_time', 0):.2f} s")
    
    def run_optimization(self):
        """Executa o processo de otimização com fallback automático"""
        try:
            with st.spinner(f"🔄 Executando otimização com {self.optimizer_type}..."):
                # Preparar parâmetros
                params = st.session_state.optimization_params.copy()
                
                # Primeira tentativa com otimizador atual
                result = self.optimizer.optimize_distribution(
                    st.session_state.origins_df,
                    st.session_state.destinations_df,
                    **params
                )
                
                # Se falhou e estamos usando PuLP, tentar scipy
                if not result['success'] and self.optimizer_type == "PuLP":
                    st.warning("⚠️ Otimização PuLP falhou, tentando com SciPy...")
                    
                    with st.spinner("🔄 Executando otimização com SciPy (fallback)..."):
                        fallback_optimizer = ScipyOptimizer()
                        result = fallback_optimizer.optimize_distribution(
                            st.session_state.origins_df,
                            st.session_state.destinations_df,
                            **params
                        )
                        
                        if result['success']:
                            st.info("✅ Otimização bem-sucedida usando SciPy como fallback!")
                            # Atualizar para usar scipy por padrão
                            self.optimizer = fallback_optimizer
                            self.optimizer_type = "SciPy"
                
                # Salvar resultado
                st.session_state.optimization_result = result
                
                if result['success']:
                    st.success(f"✅ Otimização concluída com sucesso usando {self.optimizer_type}!")
                    st.balloons()
                    
                    # Mostrar resumo rápido
                    summary = result['summary']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Volume Transportado", f"{summary['total_volume_transported']:.2f} m³")
                    
                    with col2:
                        st.metric("Distância Média", f"{summary['average_distance']:.2f} m")
                    
                    with col3:
                        st.metric("Número de Alocações", summary['num_allocations'])
                    
                else:
                    st.error(f"❌ Otimização falhou com ambos os métodos: {result.get('error', 'Erro desconhecido')}")
                    
                    # Mostrar informações de diagnóstico
                    with st.expander("🔧 Diagnóstico e Soluções"):
                        st.write("**Possíveis causas:**")
                        st.write("• Problema infactível (volume insuficiente)")
                        st.write("• Restrições contraditórias")
                        st.write("• Dados de entrada inválidos")
                        
                        st.write("**Soluções sugeridas:**")
                        st.write("1. Verifique se há volume suficiente nas origens")
                        st.write("2. Confirme se as restrições de ISC são atendíveis")
                        st.write("3. Execute o corretor de solver: `python fix_solver.py`")
                        st.write("4. Tente reduzir a complexidade do problema")
                
        except Exception as e:
            logger.error(f"Erro durante otimização: {str(e)}")
            st.error(f"Erro durante a otimização: {str(e)}")
            
            with st.expander("🔧 Informações Técnicas"):
                st.code(traceback.format_exc())
    
    def display_data_statistics(self):
        """Exibe estatísticas dos dados carregados"""
        st.subheader("📈 Estatísticas dos Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Estatísticas das Origens:**")
            origins_stats = {
                "Total de origens": len(st.session_state.origins_df),
                "Volume total disponível": f"{st.session_state.origins_df['Volume disponível (m³)'].sum():.2f} m³",
                "ISC médio": f"{st.session_state.origins_df['ISC'].mean():.2f}",
                "Extensão do projeto": f"{st.session_state.origins_df['Centro de Massa (m)'].max() - st.session_state.origins_df['Centro de Massa (m)'].min():.2f} m"
            }
            
            for key, value in origins_stats.items():
                st.metric(key, value)
        
        with col2:
            st.write("**Estatísticas dos Destinos:**")
            cft_total = st.session_state.destinations_df['Volume CFT (m³)'].fillna(0).sum()
            ca_total = st.session_state.destinations_df['Volume CA (m³)'].fillna(0).sum()
            
            destinations_stats = {
                "Total de destinos": len(st.session_state.destinations_df),
                "Volume CFT necessário": f"{cft_total:.2f} m³",
                "Volume CA necessário": f"{ca_total:.2f} m³",
                "ISC mínimo médio": f"{st.session_state.destinations_df['ISC mínimo exigido'].mean():.2f}"
            }
            
            for key, value in destinations_stats.items():
                st.metric(key, value)
    
    def clear_session(self):
        """Limpa todas as variáveis da sessão"""
        for key in ['origins_df', 'destinations_df', 'optimization_result', 'optimization_params']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.page = 'entrada_dados'
        st.success("🔄 Sessão limpa com sucesso!")
    
    def load_example_data(self):
        """Carrega dados de exemplo básicos"""
        self.load_simple_example()
    
    def load_simple_example(self):
        """Carrega exemplo simples"""
        # Dados de exemplo para origens
        origins_data = {
            'Tipo': ['Corte', 'Empréstimo Lateral', 'Empréstimo Concentrado'],
            'Centro de Massa (m)': [100, 300, 500],
            'Volume disponível (m³)': [2000, 1500, 2500],
            'ISC': [12, 8, 15],
            'DT Fixo (m)': [None, 200, 400]
        }
        
        st.session_state.origins_df = pd.DataFrame(origins_data)
        
        # Dados de exemplo para destinos
        destinations_data = {
            'Centro de Massa (m)': [150, 250, 400],
            'ISC mínimo exigido': [10, 6, 12],
            'Volume CFT (m³)': [800, 600, 700],
            'Volume CA (m³)': [600, 800, 900]
        }
        
        st.session_state.destinations_df = pd.DataFrame(destinations_data)
    
    def load_complex_example(self):
        """Carrega exemplo complexo"""
        # Dados mais complexos com mais origens e destinos
        origins_data = {
            'Tipo': ['Corte', 'Corte', 'Empréstimo Lateral', 'Empréstimo Lateral', 
                     'Empréstimo Concentrado', 'Empréstimo Concentrado'],
            'Centro de Massa (m)': [50, 150, 300, 450, 600, 800],
            'Volume disponível (m³)': [1500, 2200, 1800, 1200, 3000, 2500],
            'ISC': [15, 18, 10, 12, 8, 20],
            'DT Fixo (m)': [None, None, 300, 250, 500, 600]
        }
        
        st.session_state.origins_df = pd.DataFrame(origins_data)
        
        destinations_data = {
            'Centro de Massa (m)': [75, 200, 350, 500, 650, 750],
            'ISC mínimo exigido': [12, 8, 15, 10, 6, 18],
            'Volume CFT (m³)': [600, 800, 500, 700, 900, 400],
            'Volume CA (m³)': [400, 600, 800, 500, 700, 600]
        }
        
        st.session_state.destinations_df = pd.DataFrame(destinations_data)


def main():
    """Função principal da aplicação"""
    try:
        app = TerraplenageM()
        app.run()
        
    except Exception as e:
        st.error("💥 Erro crítico na aplicação!")
        st.error(f"Detalhes: {str(e)}")
        
        # Log do erro
        logger.critical(f"Erro crítico na aplicação: {str(e)}")
        
        # Opção de reiniciar
        if st.button("🔄 Reiniciar Aplicação"):
            st.rerun()


if __name__ == "__main__":
    main()
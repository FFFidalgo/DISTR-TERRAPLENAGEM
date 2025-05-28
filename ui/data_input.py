"""
Manipulador de entrada de dados via interface Streamlit
"""

import streamlit as st
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from io import BytesIO

from utils.data_validator import DataValidator, DataSanitizer
from config import DATA_CONFIG, UI_CONFIG, ERROR_MESSAGES


class DataInputHandler:
    """Classe para gerenciar entrada de dados via interface"""
    
    @staticmethod
    def create_file_uploader(
        label: str, 
        help_text: str = None,
        key: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Cria um widget de upload de arquivo e processa os dados
        
        Args:
            label: Rótulo do widget
            help_text: Texto de ajuda
            key: Chave única do widget
            
        Returns:
            Optional[pd.DataFrame]: DataFrame com os dados carregados ou None
        """
        uploaded_file = st.file_uploader(
            label,
            type=DATA_CONFIG['supported_file_types'],
            help=help_text or "Carregue um arquivo Excel (.xlsx, .xls) ou CSV (.csv)",
            key=key
        )
        
        if uploaded_file is not None:
            try:
                # Determinar tipo de arquivo e ler adequadamente
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Limpar e sanitizar dados
                df = DataSanitizer.sanitize_dataframe(df)
                
                return df
                
            except Exception as e:
                st.error(f"{ERROR_MESSAGES['file_upload_error']}: {str(e)}")
                return None
        
        return None
    
    @staticmethod
    def display_data_preview(
        df: pd.DataFrame, 
        title: str = "Prévia dos Dados",
        max_rows: int = 10
    ):
        """
        Exibe uma prévia dos dados carregados
        
        Args:
            df: DataFrame a ser exibido
            title: Título da seção
            max_rows: Número máximo de linhas a exibir
        """
        st.subheader(title)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Número de linhas", len(df))
        
        with col2:
            st.metric("Número de colunas", len(df.columns))
        
        # Exibir colunas disponíveis
        st.write("**Colunas disponíveis:**")
        st.write(", ".join(df.columns.tolist()))
        
        # Exibir prévia dos dados
        st.write("**Primeiras linhas:**")
        st.dataframe(df.head(max_rows))
        
        # Estatísticas básicas para colunas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Estatísticas das colunas numéricas:**")
            st.dataframe(df[numeric_cols].describe())
    
    @staticmethod
    def validate_and_display_errors(
        df: pd.DataFrame, 
        data_type: str
    ) -> bool:
        """
        Valida dados e exibe erros se houver
        
        Args:
            df: DataFrame a ser validado
            data_type: Tipo de dados ('origins' ou 'destinations')
            
        Returns:
            bool: True se válido, False caso contrário
        """
        if data_type == 'origins':
            is_valid, errors = DataValidator.validate_origins_data(df)
        elif data_type == 'destinations':
            is_valid, errors = DataValidator.validate_destinations_data(df)
        else:
            st.error("Tipo de dados inválido para validação")
            return False
        
        if not is_valid:
            st.error("❌ Erros encontrados nos dados:")
            for error in errors:
                st.error(f"• {error}")
            return False
        else:
            st.success("✅ Dados validados com sucesso!")
            return True
    
    @staticmethod
    def create_optimization_parameters_form() -> Dict[str, Any]:
        """
        Cria formulário para parâmetros de otimização
        
        Returns:
            Dict: Parâmetros de otimização selecionados
        """
        st.subheader("⚙️ Parâmetros de Otimização")
        
        with st.form("optimization_params"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Preferências de Material**")
                favor_cortes = st.checkbox(
                    "Favorecer cortes sobre empréstimos",
                    value=False,
                    help="Aplica peso menor para transporte de cortes"
                )
                
                st.write("**Limites de Tempo**")
                time_limit = st.number_input(
                    "Limite de tempo (segundos)",
                    min_value=60,
                    max_value=7200,
                    value=1800,
                    step=60,
                    help="Tempo máximo para otimização"
                )
            
            with col2:
                st.write("**Restrições de Distância**")
                use_distance_limits = st.checkbox("Aplicar limites de distância")
                
                max_dist_cortes = None
                max_dist_emprestimos = None
                
                if use_distance_limits:
                    max_dist_cortes = st.number_input(
                        "Distância máxima para cortes (m)",
                        min_value=100,
                        max_value=50000,
                        value=5000,
                        step=100
                    )
                    
                    max_dist_emprestimos = st.number_input(
                        "Distância máxima para empréstimos (m)",
                        min_value=100,
                        max_value=50000,
                        value=10000,
                        step=100
                    )
            
            submitted = st.form_submit_button("Aplicar Parâmetros")
            
            if submitted:
                return {
                    'favor_cortes': favor_cortes,
                    'time_limit': time_limit,
                    'max_dist_cortes': max_dist_cortes,
                    'max_dist_emprestimos': max_dist_emprestimos,
                    'use_distance_limits': use_distance_limits
                }
        
        return {}
    
    @staticmethod
    def create_manual_data_entry_form(data_type: str) -> Optional[pd.DataFrame]:
        """
        Cria formulário para entrada manual de dados
        
        Args:
            data_type: Tipo de dados ('origins' ou 'destinations')
            
        Returns:
            Optional[pd.DataFrame]: DataFrame com dados inseridos manualmente
        """
        if data_type == 'origins':
            return DataInputHandler._create_origins_form()
        elif data_type == 'destinations':
            return DataInputHandler._create_destinations_form()
        else:
            st.error("Tipo de dados inválido")
            return None
    
    @staticmethod
    def _create_origins_form() -> Optional[pd.DataFrame]:
        """Cria formulário para entrada manual de origens"""
        st.subheader("📝 Entrada Manual de Origens")
        
        with st.form("manual_origins"):
            num_origins = st.number_input(
                "Número de origens", 
                min_value=1, 
                max_value=20, 
                value=3
            )
            
            origins_data = []
            
            for i in range(num_origins):
                st.write(f"**Origem {i+1}**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    tipo = st.selectbox(
                        "Tipo",
                        ["Corte", "Empréstimo Lateral", "Empréstimo Concentrado"],
                        key=f"tipo_origem_{i}"
                    )
                
                with col2:
                    centro_massa = st.number_input(
                        "Centro de Massa (m)",
                        value=0.0,
                        key=f"centro_origem_{i}"
                    )
                
                with col3:
                    volume = st.number_input(
                        "Volume (m³)",
                        min_value=0.0,
                        value=1000.0,
                        key=f"volume_origem_{i}"
                    )
                
                with col4:
                    isc = st.number_input(
                        "ISC",
                        min_value=0.0,
                        max_value=100.0,
                        value=10.0,
                        key=f"isc_origem_{i}"
                    )
                
                # DT Fixo para empréstimos
                dt_fixo = 0.0
                if "Empréstimo" in tipo:
                    dt_fixo = st.number_input(
                        "DT Fixo (m)",
                        min_value=0.0,
                        value=500.0,
                        key=f"dt_fixo_origem_{i}"
                    )
                
                origins_data.append({
                    'Tipo': tipo,
                    'Centro de Massa (m)': centro_massa,
                    'Volume disponível (m³)': volume,
                    'ISC': isc,
                    'DT Fixo (m)': dt_fixo if dt_fixo > 0 else None
                })
            
            submitted = st.form_submit_button("Criar Dados de Origens")
            
            if submitted:
                df = pd.DataFrame(origins_data)
                return df
        
        return None
    
    @staticmethod
    def _create_destinations_form() -> Optional[pd.DataFrame]:
        """Cria formulário para entrada manual de destinos"""
        st.subheader("📝 Entrada Manual de Destinos")
        
        with st.form("manual_destinations"):
            num_destinations = st.number_input(
                "Número de destinos", 
                min_value=1, 
                max_value=20, 
                value=3
            )
            
            destinations_data = []
            
            for i in range(num_destinations):
                st.write(f"**Destino {i+1}**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    centro_massa = st.number_input(
                        "Centro de Massa (m)",
                        value=0.0,
                        key=f"centro_destino_{i}"
                    )
                
                with col2:
                    isc_min = st.number_input(
                        "ISC Mínimo",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.0,
                        key=f"isc_min_destino_{i}"
                    )
                
                with col3:
                    vol_cft = st.number_input(
                        "Volume CFT (m³)",
                        min_value=0.0,
                        value=500.0,
                        key=f"vol_cft_destino_{i}"
                    )
                
                with col4:
                    vol_ca = st.number_input(
                        "Volume CA (m³)",
                        min_value=0.0,
                        value=500.0,
                        key=f"vol_ca_destino_{i}"
                    )
                
                destinations_data.append({
                    'Centro de Massa (m)': centro_massa,
                    'ISC mínimo exigido': isc_min,
                    'Volume CFT (m³)': vol_cft if vol_cft > 0 else None,
                    'Volume CA (m³)': vol_ca if vol_ca > 0 else None
                })
            
            submitted = st.form_submit_button("Criar Dados de Destinos")
            
            if submitted:
                df = pd.DataFrame(destinations_data)
                return df
        
        return None


class DataExportHandler:
    """Classe para exportação de dados e resultados"""
    
    @staticmethod
    def create_download_button(
        data: pd.DataFrame,
        filename: str,
        file_format: str = 'xlsx',
        button_text: str = "Baixar Arquivo"
    ):
        """
        Cria botão de download para DataFrame
        
        Args:
            data: DataFrame a ser exportado
            filename: Nome do arquivo
            file_format: Formato do arquivo ('xlsx' ou 'csv')
            button_text: Texto do botão
        """
        if file_format == 'xlsx':
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False, sheet_name='Resultados')
            
            st.download_button(
                label=button_text,
                data=buffer.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        elif file_format == 'csv':
            csv_data = data.to_csv(index=False)
            st.download_button(
                label=button_text,
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
    
    @staticmethod
    def export_complete_results(
        allocations_df: pd.DataFrame,
        summary: Dict[str, Any],
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        filename: str = "resultado_completo"
    ):
        """
        Exporta resultados completos em um arquivo Excel com múltiplas abas
        
        Args:
            allocations_df: DataFrame com alocações
            summary: Dicionário com resumo
            origins_df: DataFrame original das origens
            destinations_df: DataFrame original dos destinos
            filename: Nome do arquivo
        """
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Aba com alocações
            allocations_df.to_excel(writer, sheet_name='Alocacoes', index=False)
            
            # Aba com resumo
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Resumo', index=False)
            
            # Aba com dados originais
            origins_df.to_excel(writer, sheet_name='Origens', index=False)
            destinations_df.to_excel(writer, sheet_name='Destinos', index=False)
        
        st.download_button(
            label="📁 Baixar Resultados Completos (Excel)",
            data=buffer.getvalue(),
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

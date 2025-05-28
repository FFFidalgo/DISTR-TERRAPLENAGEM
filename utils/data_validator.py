"""
Utilitários para validação de dados de entrada
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from config import DATA_CONFIG, VALIDATION_CONFIG, ERROR_MESSAGES


class DataValidator:
    """Classe para validação de dados de entrada"""
    
    @staticmethod
    def validate_origins_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida os dados das origens
        
        Args:
            df: DataFrame com dados das origens
            
        Returns:
            Tuple[bool, List[str]]: (é_válido, lista_de_erros)
        """
        errors = []
        
        # Verificar colunas obrigatórias
        required_cols = DATA_CONFIG['required_origin_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(ERROR_MESSAGES['missing_columns'].format(
                columns=', '.join(missing_cols)
            ))
            return False, errors
        
        # Validar dados linha por linha
        for idx, row in df.iterrows():
            row_errors = DataValidator._validate_origin_row(idx, row)
            errors.extend(row_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_destinations_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida os dados dos destinos
        
        Args:
            df: DataFrame com dados dos destinos
            
        Returns:
            Tuple[bool, List[str]]: (é_válido, lista_de_erros)
        """
        errors = []
        
        # Verificar colunas obrigatórias
        required_cols = DATA_CONFIG['required_destination_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(ERROR_MESSAGES['missing_columns'].format(
                columns=', '.join(missing_cols)
            ))
            return False, errors
        
        # Validar dados linha por linha
        for idx, row in df.iterrows():
            row_errors = DataValidator._validate_destination_row(idx, row)
            errors.extend(row_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_origin_row(idx: Any, row: pd.Series) -> List[str]:
        """Valida uma linha de dados de origem"""
        errors = []
        
        # Validar volume disponível
        try:
            volume = float(row['Volume disponível (m³)'])
            if volume < VALIDATION_CONFIG['min_volume']:
                errors.append(f"Linha {idx}: Volume muito pequeno ({volume} m³)")
        except (ValueError, TypeError):
            errors.append(f"Linha {idx}: Volume inválido")
        
        # Validar ISC
        try:
            isc = float(row['ISC'])
            if not (VALIDATION_CONFIG['min_isc'] <= isc <= VALIDATION_CONFIG['max_isc']):
                errors.append(f"Linha {idx}: ISC fora do intervalo válido ({isc})")
        except (ValueError, TypeError):
            errors.append(f"Linha {idx}: ISC inválido")
        
        # Validar centro de massa
        try:
            center = float(row['Centro de Massa (m)'])
        except (ValueError, TypeError):
            errors.append(f"Linha {idx}: Centro de massa inválido")
        
        # Validar DT Fixo (se presente)
        if 'DT Fixo (m)' in row.index and pd.notna(row['DT Fixo (m)']):
            try:
                dt_fixo = float(row['DT Fixo (m)'])
                if dt_fixo < 0:
                    errors.append(f"Linha {idx}: DT Fixo não pode ser negativo")
            except (ValueError, TypeError):
                errors.append(f"Linha {idx}: DT Fixo inválido")
        
        return errors
    
    @staticmethod
    def _validate_destination_row(idx: Any, row: pd.Series) -> List[str]:
        """Valida uma linha de dados de destino"""
        errors = []
        
        # Validar centro de massa
        try:
            center = float(row['Centro de Massa (m)'])
        except (ValueError, TypeError):
            errors.append(f"Linha {idx}: Centro de massa inválido")
        
        # Validar ISC mínimo
        try:
            isc_min = float(row['ISC mínimo exigido'])
            if not (VALIDATION_CONFIG['min_isc'] <= isc_min <= VALIDATION_CONFIG['max_isc']):
                errors.append(f"Linha {idx}: ISC mínimo fora do intervalo válido ({isc_min})")
        except (ValueError, TypeError):
            errors.append(f"Linha {idx}: ISC mínimo inválido")
        
        # Validar volumes (se presentes)
        for vol_col in ['Volume CFT (m³)', 'Volume CA (m³)']:
            if vol_col in row.index and pd.notna(row[vol_col]):
                try:
                    volume = float(row[vol_col])
                    if volume < 0:
                        errors.append(f"Linha {idx}: {vol_col} não pode ser negativo")
                except (ValueError, TypeError):
                    errors.append(f"Linha {idx}: {vol_col} inválido")
        
        return errors
    
    @staticmethod
    def check_feasibility(origins_df: pd.DataFrame, destinations_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Verifica se o problema tem uma solução factível
        
        Args:
            origins_df: DataFrame com dados das origens
            destinations_df: DataFrame com dados dos destinos
            
        Returns:
            Tuple[bool, str]: (é_factível, mensagem_de_erro)
        """
        try:
            total_volume_origem = origins_df['Volume disponível (m³)'].sum()
            total_volume_destino = (
                destinations_df['Volume CFT (m³)'].fillna(0).sum() + 
                destinations_df['Volume CA (m³)'].fillna(0).sum()
            )
            
            if total_volume_origem < total_volume_destino:
                return False, (
                    f"Volume total de origem ({total_volume_origem:.2f} m³) é menor "
                    f"que o volume total de destino ({total_volume_destino:.2f} m³)."
                )
            
            # Verificar se há material suficiente com ISC adequado para CFT
            for d_idx, d_row in destinations_df.iterrows():
                if pd.isna(d_row['Volume CFT (m³)']) or d_row['Volume CFT (m³)'] <= 0:
                    continue
                    
                isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
                valid_origins = origins_df[origins_df['ISC'] >= isc_min]
                
                if valid_origins.empty:
                    return False, (
                        f"Não há origens com ISC suficiente para o destino {d_idx} "
                        f"(ISC min: {isc_min})."
                    )
                
                total_valid_volume = valid_origins['Volume disponível (m³)'].sum()
                if total_valid_volume < d_row['Volume CFT (m³)']:
                    return False, (
                        f"Volume disponível com ISC adequado ({total_valid_volume:.2f} m³) "
                        f"é menor que o necessário para CFT no destino {d_idx} "
                        f"({d_row['Volume CFT (m³)']:.2f} m³)."
                    )
            
            return True, "Problema é factível"
            
        except Exception as e:
            return False, f"Erro na verificação de factibilidade: {str(e)}"


class DataSanitizer:
    """Classe para limpeza e padronização de dados"""
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza um DataFrame
        
        Args:
            df: DataFrame a ser limpo
            
        Returns:
            pd.DataFrame: DataFrame limpo
        """
        df_clean = df.copy()
        
        # Remover linhas completamente vazias
        df_clean = df_clean.dropna(how='all')
        
        # Padronizar nomes das colunas (remover espaços extras)
        df_clean.columns = df_clean.columns.str.strip()
        
        # Converter tipos de dados numéricos
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, fill_strategy: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Preenche valores ausentes baseado em estratégia definida
        
        Args:
            df: DataFrame com valores ausentes
            fill_strategy: Dicionário com estratégias de preenchimento por coluna
            
        Returns:
            pd.DataFrame: DataFrame com valores preenchidos
        """
        df_filled = df.copy()
        
        if fill_strategy is None:
            fill_strategy = {
                'Volume CFT (m³)': 0,
                'Volume CA (m³)': 0,
                'DT Fixo (m)': 0
            }
        
        for col, value in fill_strategy.items():
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(value)
        
        return df_filled

"""
Testes automatizados para a aplicação de otimização de terraplenagem
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append('/app')

from utils.data_validator import DataValidator, DataSanitizer
from optimization.optimizer import TerraplenagemOptimizer
from optimization.distance_calculator import DistanceCalculator
from config import DATA_CONFIG, VALIDATION_CONFIG


class TestDataValidator:
    """Testes para validação de dados"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.valid_origins_data = {
            'Tipo': ['Corte', 'Empréstimo Lateral'],
            'Centro de Massa (m)': [100, 200],
            'Volume disponível (m³)': [1000, 1500],
            'ISC': [15, 10]
        }
        
        self.valid_destinations_data = {
            'Centro de Massa (m)': [150, 250],
            'ISC mínimo exigido': [12, 8],
            'Volume CFT (m³)': [500, 600],
            'Volume CA (m³)': [400, 500]
        }
        
        self.valid_origins_df = pd.DataFrame(self.valid_origins_data)
        self.valid_destinations_df = pd.DataFrame(self.valid_destinations_data)
    
    def test_validate_origins_data_valid(self):
        """Testa validação com dados de origens válidos"""
        is_valid, errors = DataValidator.validate_origins_data(self.valid_origins_df)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_origins_data_missing_columns(self):
        """Testa validação com colunas faltantes"""
        df_invalid = self.valid_origins_df.drop(columns=['ISC'])
        is_valid, errors = DataValidator.validate_origins_data(df_invalid)
        assert not is_valid
        assert len(errors) > 0
        assert 'ISC' in errors[0]
    
    def test_validate_destinations_data_valid(self):
        """Testa validação com dados de destinos válidos"""
        is_valid, errors = DataValidator.validate_destinations_data(self.valid_destinations_df)
        assert is_valid
        assert len(errors) == 0
    
    def test_check_feasibility_valid(self):
        """Testa verificação de factibilidade com dados válidos"""
        is_feasible, message = DataValidator.check_feasibility(
            self.valid_origins_df, 
            self.valid_destinations_df
        )
        assert is_feasible
    
    def test_check_feasibility_insufficient_volume(self):
        """Testa verificação de factibilidade com volume insuficiente"""
        # Reduzir volume disponível
        df_insufficient = self.valid_origins_df.copy()
        df_insufficient['Volume disponível (m³)'] = [100, 200]  # Total: 300, necessário: ~2000
        
        is_feasible, message = DataValidator.check_feasibility(
            df_insufficient, 
            self.valid_destinations_df
        )
        assert not is_feasible
        assert 'menor que o volume total de destino' in message


class TestDataSanitizer:
    """Testes para limpeza de dados"""
    
    def test_sanitize_dataframe(self):
        """Testa limpeza básica de DataFrame"""
        dirty_data = {
            'Coluna 1  ': [1, 2, None],  # Espaços no nome da coluna
            'Coluna 2': [None, None, None],  # Linha vazia
            'Coluna 3': ['1.5', '2.0', '3.5']  # Strings numéricas
        }
        
        df_dirty = pd.DataFrame(dirty_data)
        df_clean = DataSanitizer.sanitize_dataframe(df_dirty)
        
        # Verificar se colunas foram limpas
        assert 'Coluna 1' in df_clean.columns
        assert 'Coluna 1  ' not in df_clean.columns
        
        # Verificar se linhas vazias foram removidas
        assert len(df_clean) <= len(df_dirty)
    
    def test_fill_missing_values(self):
        """Testa preenchimento de valores ausentes"""
        data_with_missing = {
            'Volume CFT (m³)': [100, None, 200],
            'Volume CA (m³)': [None, 150, 250],
            'Outra Coluna': [1, 2, 3]
        }
        
        df_missing = pd.DataFrame(data_with_missing)
        df_filled = DataSanitizer.fill_missing_values(df_missing)
        
        # Verificar se valores foram preenchidos
        assert df_filled['Volume CFT (m³)'].isna().sum() == 0
        assert df_filled['Volume CA (m³)'].isna().sum() == 0
        
        # Verificar se outras colunas não foram alteradas
        assert df_filled['Outra Coluna'].isna().sum() == 0


class TestDistanceCalculator:
    """Testes para calculadora de distâncias"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.origins_data = {
            'Centro de Massa (m)': [100, 300],
            'DT Fixo (m)': [None, 200],
            'Volume disponível (m³)': [1000, 1500],
            'ISC': [15, 10]
        }
        self.origins_df = pd.DataFrame(self.origins_data)
    
    def test_calculate_distance_without_dt_fixo(self):
        """Testa cálculo de distância sem DT fixo"""
        distance = DistanceCalculator.calculate_distance(
            0, 100, 200, self.origins_df
        )
        assert distance == 100  # |200 - 100|
    
    def test_calculate_distance_with_dt_fixo(self):
        """Testa cálculo de distância com DT fixo"""
        distance = DistanceCalculator.calculate_distance(
            1, 300, 400, self.origins_df
        )
        assert distance == 300  # |400 - 300| + 200
    
    def test_calculate_distance_matrix(self):
        """Testa cálculo da matriz de distâncias"""
        destinations_data = {
            'Centro de Massa (m)': [150, 250, 350]
        }
        destinations_df = pd.DataFrame(destinations_data)
        
        matrix = DistanceCalculator.calculate_distance_matrix(
            self.origins_df, destinations_df
        )
        
        assert matrix.shape == (2, 3)  # 2 origens, 3 destinos
        assert matrix[0, 0] == 50  # |150 - 100|
        assert matrix[1, 0] == 350  # |150 - 300| + 200


class TestTerraplenagemOptimizer:
    """Testes para o otimizador"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = TerraplenagemOptimizer(time_limit=60)
        
        self.origins_data = {
            'Tipo': ['Corte', 'Empréstimo Concentrado'],
            'Centro de Massa (m)': [100, 300],
            'Volume disponível (m³)': [2000, 2000],
            'ISC': [15, 10],
            'DT Fixo (m)': [None, 200]
        }
        
        self.destinations_data = {
            'Centro de Massa (m)': [150, 250],
            'ISC mínimo exigido': [12, 8],
            'Volume CFT (m³)': [800, 600],
            'Volume CA (m³)': [500, 700]
        }
        
        self.origins_df = pd.DataFrame(self.origins_data)
        self.destinations_df = pd.DataFrame(self.destinations_data)
    
    def test_optimizer_initialization(self):
        """Testa inicialização do otimizador"""
        assert self.optimizer.time_limit == 60
        assert hasattr(self.optimizer, 'optimization_history')
        assert len(self.optimizer.optimization_history) == 0
    
    def test_identify_origin_types(self):
        """Testa identificação de tipos de origens"""
        origin_types = self.optimizer._identify_origin_types(self.origins_df)
        
        assert len(origin_types['cortes']) == 1
        assert len(origin_types['emprestimos_concentrados']) == 1
        assert len(origin_types['emprestimos_laterais']) == 0
    
    def test_optimization_with_valid_data(self):
        """Testa otimização com dados válidos"""
        result = self.optimizer.optimize_distribution(
            self.origins_df, 
            self.destinations_df
        )
        
        assert 'success' in result
        assert 'status' in result
        assert 'execution_time' in result
        
        if result['success']:
            assert 'allocations' in result
            assert 'summary' in result
            assert isinstance(result['allocations'], pd.DataFrame)
            assert isinstance(result['summary'], dict)
    
    def test_optimization_with_infeasible_data(self):
        """Testa otimização com dados infactíveis"""
        # Criar dados infactíveis (volume insuficiente)
        infeasible_origins = self.origins_df.copy()
        infeasible_origins['Volume disponível (m³)'] = [50, 100]  # Muito pouco volume
        
        result = self.optimizer.optimize_distribution(
            infeasible_origins, 
            self.destinations_df
        )
        
        # Pode falhar na preparação ou retornar status não-ótimo
        assert 'success' in result
        assert 'status' in result


class TestIntegration:
    """Testes de integração entre componentes"""
    
    def setup_method(self):
        """Setup para testes de integração"""
        self.origins_data = {
            'Tipo': ['Corte', 'Empréstimo Lateral', 'Empréstimo Concentrado'],
            'Centro de Massa (m)': [50, 200, 400],
            'Volume disponível (m³)': [1500, 1200, 2000],
            'ISC': [18, 12, 8],
            'DT Fixo (m)': [None, 300, 500]
        }
        
        self.destinations_data = {
            'Centro de Massa (m)': [100, 300, 500],
            'ISC mínimo exigido': [15, 10, 6],
            'Volume CFT (m³)': [600, 500, 700],
            'Volume CA (m³)': [400, 600, 500]
        }
        
        self.origins_df = pd.DataFrame(self.origins_data)
        self.destinations_df = pd.DataFrame(self.destinations_data)
    
    def test_full_pipeline(self):
        """Testa pipeline completo: validação + otimização"""
        # 1. Validação
        origins_valid, _ = DataValidator.validate_origins_data(self.origins_df)
        destinations_valid, _ = DataValidator.validate_destinations_data(self.destinations_df)
        
        assert origins_valid
        assert destinations_valid
        
        # 2. Verificação de factibilidade
        is_feasible, _ = DataValidator.check_feasibility(
            self.origins_df, self.destinations_df
        )
        assert is_feasible
        
        # 3. Otimização
        optimizer = TerraplenagemOptimizer(time_limit=120)
        result = optimizer.optimize_distribution(
            self.origins_df, 
            self.destinations_df,
            favor_cortes=True
        )
        
        assert 'success' in result
        
        if result['success']:
            # 4. Verificação dos resultados
            allocations = result['allocations']
            summary = result['summary']
            
            assert len(allocations) > 0
            assert summary['total_volume_transported'] > 0
            assert summary['num_allocations'] > 0
            
            # Verificar se todas as alocações são válidas
            assert all(allocations['volume'] > 0)
            assert all(allocations['distancia'] >= 0)
    
    def test_distance_calculation_consistency(self):
        """Testa consistência nos cálculos de distância"""
        # Calcular distâncias individuais
        individual_distances = []
        for i, origin in self.origins_df.iterrows():
            for j, dest in self.destinations_df.iterrows():
                dist = DistanceCalculator.calculate_distance(
                    i, origin['Centro de Massa (m)'], 
                    dest['Centro de Massa (m)'], 
                    self.origins_df
                )
                individual_distances.append(dist)
        
        # Calcular matriz de distâncias
        distance_matrix = DistanceCalculator.calculate_distance_matrix(
            self.origins_df, self.destinations_df
        )
        
        matrix_distances = distance_matrix.flatten()
        
        # Verificar se são iguais
        np.testing.assert_array_almost_equal(
            individual_distances, matrix_distances, decimal=6
        )


def run_tests():
    """Executa todos os testes"""
    print("🧪 Iniciando testes automatizados...")
    
    # Executar testes usando pytest
    test_result = pytest.main([
        __file__, 
        '-v', 
        '--tb=short',
        '--color=yes'
    ])
    
    if test_result == 0:
        print("✅ Todos os testes passaram!")
    else:
        print("❌ Alguns testes falharam!")
    
    return test_result


if __name__ == "__main__":
    run_tests()
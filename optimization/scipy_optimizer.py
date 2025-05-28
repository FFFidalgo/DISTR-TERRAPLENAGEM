"""
Versão alternativa do otimizador usando scipy para evitar problemas com CBC
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from .distance_calculator import DistanceCalculator


class ScipyOptimizer:
    """Otimizador alternativo usando scipy.optimize para evitar problemas com solvers externos"""
    
    def __init__(self, time_limit: int = None):
        """
        Inicializa o otimizador com scipy
        
        Args:
            time_limit: Limite de tempo (não usado no scipy, mas mantido para compatibilidade)
        """
        self.time_limit = time_limit or 1800
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
    
    def optimize_distribution(
        self,
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa otimização usando scipy.optimize.linprog
        
        Args:
            origins_df: DataFrame com dados das origens
            destinations_df: DataFrame com dados dos destinos
            **kwargs: Parâmetros adicionais
            
        Returns:
            Dict: Resultado da otimização
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Iniciando otimização com scipy")
            
            # Preparar dados
            result = self._prepare_optimization_data(origins_df, destinations_df, **kwargs)
            if not result['success']:
                return result
            
            opt_data = result['data']
            
            # Converter para formato do scipy
            c, A_ub, b_ub, A_eq, b_eq, bounds = self._convert_to_scipy_format(opt_data)
            
            # Resolver com scipy
            scipy_result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',  # Método mais robusto
                options={'presolve': True, 'time_limit': self.time_limit}
            )
            
            # Processar resultado
            result = self._process_scipy_solution(scipy_result, opt_data)
            
            # Calcular tempo de execução
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            
            # Salvar no histórico
            self.optimization_history.append({
                'timestamp': start_time,
                'execution_time': execution_time,
                'status': result['status'],
                'objective_value': result.get('objective_value'),
                'num_origins': len(origins_df),
                'num_destinations': len(destinations_df),
                'solver': 'scipy'
            })
            
            self.logger.info(f"Otimização scipy concluída em {execution_time:.2f}s - Status: {result['status']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erro durante otimização scipy: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'status': 'Error',
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _prepare_optimization_data(
        self, 
        origins_df: pd.DataFrame, 
        destinations_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepara os dados para otimização scipy"""
        try:
            # Cópias dos DataFrames
            origins = origins_df.copy()
            destinations = destinations_df.copy()
            
            # Volumes disponíveis e necessários
            available_volumes = origins['Volume disponível (m³)'].copy()
            needed_cft = destinations['Volume CFT (m³)'].fillna(0).copy()
            needed_ca = destinations['Volume CA (m³)'].fillna(0).copy()
            
            # Calcular matriz de distâncias
            distance_matrix = DistanceCalculator.calculate_distance_matrix(origins, destinations)
            
            return {
                'success': True,
                'data': {
                    'origins_df': origins,
                    'destinations_df': destinations,
                    'available_volumes': available_volumes,
                    'needed_cft': needed_cft,
                    'needed_ca': needed_ca,
                    'distance_matrix': distance_matrix,
                    'num_origins': len(origins),
                    'num_destinations': len(destinations)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro na preparação dos dados: {str(e)}"
            }
    
    def _convert_to_scipy_format(self, opt_data: Dict) -> Tuple:
        """Converte o problema para formato do scipy.optimize.linprog"""
        
        num_origins = opt_data['num_origins']
        num_destinations = opt_data['num_destinations']
        distance_matrix = opt_data['distance_matrix']
        available_volumes = opt_data['available_volumes']
        needed_cft = opt_data['needed_cft']
        needed_ca = opt_data['needed_ca']
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        
        # Número total de variáveis: 2 * num_origins * num_destinations
        # (CFT e CA para cada par origem-destino)
        num_vars = 2 * num_origins * num_destinations
        
        # Função objetivo (minimizar custo total de transporte)
        c = np.zeros(num_vars)
        var_idx = 0
        
        for i in range(num_origins):
            for j in range(num_destinations):
                distance = distance_matrix[i, j]
                c[var_idx] = distance      # CFT
                c[var_idx + 1] = distance  # CA
                var_idx += 2
        
        # Preparar restrições de inequalidade (A_ub x <= b_ub)
        A_ub = []
        b_ub = []
        
        # Restrições de capacidade das origens
        for i in range(num_origins):
            constraint = np.zeros(num_vars)
            var_idx = 0
            
            for oi in range(num_origins):
                for j in range(num_destinations):
                    if oi == i:
                        constraint[var_idx] = 1      # CFT
                        constraint[var_idx + 1] = 1  # CA
                    var_idx += 2
            
            A_ub.append(constraint)
            b_ub.append(available_volumes.iloc[i])
        
        # Restrições de ISC para CFT
        for i in range(num_origins):
            for j in range(num_destinations):
                isc_origin = origins_df.iloc[i]['ISC']
                isc_min_dest = destinations_df.iloc[j]['ISC mínimo exigido']
                
                if pd.notna(isc_min_dest) and isc_origin < isc_min_dest:
                    # Esta origem não pode fornecer CFT para este destino
                    constraint = np.zeros(num_vars)
                    var_idx = 2 * (i * num_destinations + j)
                    constraint[var_idx] = 1  # Variável CFT desta combinação
                    
                    A_ub.append(constraint)
                    b_ub.append(0)  # <= 0, ou seja, = 0
        
        # Preparar restrições de igualdade (A_eq x = b_eq)
        A_eq = []
        b_eq = []
        
        # Restrições de demanda CFT nos destinos
        for j in range(num_destinations):
            if needed_cft.iloc[j] > 0:
                constraint = np.zeros(num_vars)
                var_idx = 0
                
                for i in range(num_origins):
                    for dj in range(num_destinations):
                        if dj == j:
                            constraint[var_idx] = 1  # Variável CFT
                        var_idx += 2
                
                A_eq.append(constraint)
                b_eq.append(needed_cft.iloc[j])
        
        # Restrições de demanda CA nos destinos
        for j in range(num_destinations):
            if needed_ca.iloc[j] > 0:
                constraint = np.zeros(num_vars)
                var_idx = 0
                
                for i in range(num_origins):
                    for dj in range(num_destinations):
                        if dj == j:
                            constraint[var_idx + 1] = 1  # Variável CA
                        var_idx += 2
                
                A_eq.append(constraint)
                b_eq.append(needed_ca.iloc[j])
        
        # Limites das variáveis (todas >= 0)
        bounds = [(0, None) for _ in range(num_vars)]
        
        # Converter para arrays numpy
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        return c, A_ub, b_ub, A_eq, b_eq, bounds
    
    def _process_scipy_solution(self, scipy_result, opt_data: Dict) -> Dict[str, Any]:
        """Processa a solução do scipy"""
        
        # Mapear status do scipy para nosso formato
        if scipy_result.success:
            status = 'Optimal'
            success = True
        else:
            status = 'Failed'
            success = False
        
        result = {
            'success': success,
            'status': status,
            'objective_value': scipy_result.fun if scipy_result.success else None
        }
        
        if success and scipy_result.x is not None:
            # Extrair alocações da solução
            allocations = self._extract_allocations_from_solution(
                scipy_result.x, opt_data
            )
            
            result['allocations'] = allocations['allocations_df']
            result['summary'] = allocations['summary']
        else:
            result['error'] = f"Otimização falhou: {scipy_result.message}"
        
        return result
    
    def _extract_allocations_from_solution(self, solution: np.ndarray, opt_data: Dict) -> Dict[str, Any]:
        """Extrai alocações da solução do scipy"""
        
        num_origins = opt_data['num_origins']
        num_destinations = opt_data['num_destinations']
        distance_matrix = opt_data['distance_matrix']
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        
        allocations = []
        total_distance = 0
        total_volume = 0
        
        var_idx = 0
        for i in range(num_origins):
            for j in range(num_destinations):
                cft_volume = solution[var_idx]
                ca_volume = solution[var_idx + 1]
                distance = distance_matrix[i, j]
                
                # CFT
                if cft_volume > 1e-6:
                    allocations.append({
                        'origem': origins_df.index[i],
                        'destino': destinations_df.index[j],
                        'tipo_material': 'CFT',
                        'volume': cft_volume,
                        'distancia': distance,
                        'custo_transporte': distance * cft_volume
                    })
                    
                    total_distance += distance * cft_volume
                    total_volume += cft_volume
                
                # CA
                if ca_volume > 1e-6:
                    allocations.append({
                        'origem': origins_df.index[i],
                        'destino': destinations_df.index[j],
                        'tipo_material': 'CA',
                        'volume': ca_volume,
                        'distancia': distance,
                        'custo_transporte': distance * ca_volume
                    })
                    
                    total_distance += distance * ca_volume
                    total_volume += ca_volume
                
                var_idx += 2
        
        # Criar DataFrame com resultados
        allocations_df = pd.DataFrame(allocations)
        
        # Estatísticas resumidas
        summary = {
            'total_volume_transported': total_volume,
            'total_distance_weighted': total_distance,
            'average_distance': total_distance / total_volume if total_volume > 0 else 0,
            'num_allocations': len(allocations),
            'volumes_by_material': {
                'CFT': allocations_df[allocations_df['tipo_material'] == 'CFT']['volume'].sum() if not allocations_df.empty else 0,
                'CA': allocations_df[allocations_df['tipo_material'] == 'CA']['volume'].sum() if not allocations_df.empty else 0
            }
        }
        
        return {
            'allocations_df': allocations_df,
            'summary': summary
        }
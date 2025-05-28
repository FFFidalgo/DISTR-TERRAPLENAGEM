"""
Otimizador principal para problemas de distribuição de terraplenagem
"""

import pulp as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from .distance_calculator import DistanceCalculator
from config import OPTIMIZATION_CONFIG, ERROR_MESSAGES


class TerraplenagemOptimizer:
    """Classe principal para otimização de distribuição de terraplenagem"""
    
    def __init__(self, time_limit: int = None, solver: str = None):
        """
        Inicializa o otimizador
        
        Args:
            time_limit: Limite de tempo em segundos
            solver: Nome do solver a utilizar
        """
        self.time_limit = time_limit or OPTIMIZATION_CONFIG['default_time_limit']
        self.solver_name = solver or OPTIMIZATION_CONFIG['solver']
        self.logger = logging.getLogger(__name__)
        
        # Configurar solver
        self.solver = self._setup_solver()
        
        # Histórico de otimizações
        self.optimization_history = []
    
    def _setup_solver(self):
        """Configura o solver de programação linear com fallbacks robustos"""
        # Lista de solvers para tentar em ordem de preferência
        solver_options = [
            ('PULP_CBC_CMD', lambda: pl.PULP_CBC_CMD(timeLimit=self.time_limit, msg=False)),
            ('COIN_CMD', lambda: pl.COIN_CMD(timeLimit=self.time_limit, msg=False)),
            ('GLPK_CMD', lambda: pl.GLPK_CMD(timeLimit=self.time_limit, msg=False)),
            ('CPLEX_CMD', lambda: pl.CPLEX_CMD(timeLimit=self.time_limit, msg=False)),
            ('GUROBI_CMD', lambda: pl.GUROBI_CMD(timeLimit=self.time_limit, msg=False))
        ]
        
        # Tentar configurar solver preferido primeiro
        if hasattr(self, 'solver_name') and self.solver_name:
            try:
                if self.solver_name == 'COIN_CMD':
                    solver = pl.COIN_CMD(timeLimit=self.time_limit, msg=False)
                elif self.solver_name == 'PULP_CBC_CMD':
                    solver = pl.PULP_CBC_CMD(timeLimit=self.time_limit, msg=False)
                elif self.solver_name == 'GLPK_CMD':
                    solver = pl.GLPK_CMD(timeLimit=self.time_limit, msg=False)
                else:
                    solver = pl.PULP_CBC_CMD(timeLimit=self.time_limit, msg=False)
                
                if solver.available():
                    self.logger.info(f"Solver configurado: {self.solver_name}")
                    return solver
                else:
                    self.logger.warning(f"Solver {self.solver_name} não disponível, tentando alternativas...")
            except Exception as e:
                self.logger.warning(f"Erro ao configurar solver {self.solver_name}: {e}")
        
        # Tentar solvers alternativos
        for solver_name, solver_factory in solver_options:
            try:
                solver = solver_factory()
                if solver.available():
                    self.logger.info(f"Usando solver alternativo: {solver_name}")
                    self.solver_name = solver_name  # Atualizar nome do solver
                    return solver
            except Exception as e:
                self.logger.debug(f"Solver {solver_name} não funcionou: {e}")
                continue
        
        # Se nenhum solver externo funcionar, usar o solver padrão do PuLP
        self.logger.warning("Nenhum solver externo disponível, usando solver padrão do PuLP")
        try:
            # Solver básico integrado (não precisa de executável externo)
            self.solver_name = 'DEFAULT'
            return None  # None fará o PuLP usar o solver padrão
        except Exception as e:
            self.logger.error(f"Falha crítica na configuração do solver: {e}")
            return None
    
    def optimize_distribution(
        self,
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa a otimização de distribuição de material
        
        Args:
            origins_df: DataFrame com dados das origens
            destinations_df: DataFrame com dados dos destinos
            **kwargs: Parâmetros adicionais de otimização
            
        Returns:
            Dict: Resultado da otimização
        """
        start_time = datetime.now()
        
        try:
            # Log início da otimização
            self.logger.info("Iniciando otimização de distribuição")
            
            # Criar problema de otimização
            problem = pl.LpProblem("Terraplenagem_Otimizacao", pl.LpMinimize)
            
            # Preparar dados
            result = self._prepare_optimization_data(origins_df, destinations_df, **kwargs)
            if not result['success']:
                return result
            
            opt_data = result['data']
            
            # Criar variáveis de decisão
            variables = self._create_decision_variables(opt_data)
            
            # Definir função objetivo
            problem += self._create_objective_function(variables, opt_data)
            
            # Adicionar restrições
            self._add_constraints(problem, variables, opt_data)
            
            # Resolver problema
            status = problem.solve(self.solver)
            
            # Processar resultado
            result = self._process_solution(problem, variables, opt_data, status)
            
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
                'num_destinations': len(destinations_df)
            })
            
            self.logger.info(f"Otimização concluída em {execution_time:.2f}s - Status: {result['status']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erro durante otimização: {str(e)}"
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
        """Prepara os dados para otimização"""
        try:
            # Cópias dos DataFrames
            origins = origins_df.copy()
            destinations = destinations_df.copy()
            
            # Extrair parâmetros
            favor_cortes = kwargs.get('favor_cortes', False)
            max_dist_cortes = kwargs.get('max_dist_cortes')
            max_dist_emprestimos = kwargs.get('max_dist_emprestimos')
            fixed_allocations = kwargs.get('fixed_allocations', [])
            
            # Volumes disponíveis e necessários
            available_volumes = origins['Volume disponível (m³)'].copy()
            needed_cft = destinations['Volume CFT (m³)'].fillna(0).copy()
            needed_ca = destinations['Volume CA (m³)'].fillna(0).copy()
            
            # Processar alocações fixas
            if fixed_allocations:
                available_volumes, needed_cft, needed_ca = self._process_fixed_allocations(
                    fixed_allocations, available_volumes, needed_cft, needed_ca
                )
            
            # Calcular matriz de distâncias
            distance_matrix = DistanceCalculator.calculate_distance_matrix(origins, destinations)
            
            # Identificar tipos de origens
            origin_types = self._identify_origin_types(origins)
            
            return {
                'success': True,
                'data': {
                    'origins_df': origins,
                    'destinations_df': destinations,
                    'available_volumes': available_volumes,
                    'needed_cft': needed_cft,
                    'needed_ca': needed_ca,
                    'distance_matrix': distance_matrix,
                    'origin_types': origin_types,
                    'favor_cortes': favor_cortes,
                    'max_dist_cortes': max_dist_cortes,
                    'max_dist_emprestimos': max_dist_emprestimos,
                    'fixed_allocations': fixed_allocations
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro na preparação dos dados: {str(e)}"
            }
    
    def _process_fixed_allocations(
        self,
        fixed_allocations: List[Dict],
        available_volumes: pd.Series,
        needed_cft: pd.Series,
        needed_ca: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Processa alocações fixas predefinidas"""
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            # Reduz volumes conforme alocações fixas
            if tipo == 'CFT':
                needed_cft.loc[d_idx] -= volume
            elif tipo == 'CA':
                needed_ca.loc[d_idx] -= volume
            
            # Reduz o volume disponível
            available_volumes[o_idx] -= volume
        
        return available_volumes, needed_cft, needed_ca
    
    def _identify_origin_types(self, origins_df: pd.DataFrame) -> Dict[str, List[int]]:
        """Identifica os tipos de origens"""
        origin_types = {
            'cortes': [],
            'emprestimos_laterais': [],
            'emprestimos_concentrados': []
        }
        
        for idx, row in origins_df.iterrows():
            tipo = row.get('Tipo', '').lower()
            
            if 'corte' in tipo or 'cut' in tipo:
                origin_types['cortes'].append(idx)
            elif 'empréstimo lateral' in tipo or 'lateral' in tipo:
                origin_types['emprestimos_laterais'].append(idx)
            elif 'empréstimo concentrado' in tipo or 'concentrado' in tipo:
                origin_types['emprestimos_concentrados'].append(idx)
            else:
                # Classificação padrão baseada em características
                if 'DT Fixo (m)' in origins_df.columns and pd.notna(row['DT Fixo (m)']):
                    origin_types['emprestimos_concentrados'].append(idx)
                else:
                    origin_types['cortes'].append(idx)
        
        return origin_types
    
    def _create_decision_variables(self, opt_data: Dict) -> Dict[str, Any]:
        """Cria variáveis de decisão para o problema de otimização"""
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        
        variables = {}
        
        # Variáveis para transporte de CFT
        variables['x_cft'] = {}
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                variables['x_cft'][(o_idx, d_idx)] = pl.LpVariable(
                    f"x_cft_{o_idx}_{d_idx}", 
                    lowBound=0, 
                    cat='Continuous'
                )
        
        # Variáveis para transporte de CA
        variables['x_ca'] = {}
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                variables['x_ca'][(o_idx, d_idx)] = pl.LpVariable(
                    f"x_ca_{o_idx}_{d_idx}", 
                    lowBound=0, 
                    cat='Continuous'
                )
        
        return variables
    
    def _create_objective_function(self, variables: Dict, opt_data: Dict):
        """Cria a função objetivo (minimizar distância total ponderada)"""
        objective = 0
        
        distance_matrix = opt_data['distance_matrix']
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        favor_cortes = opt_data['favor_cortes']
        
        for i, o_idx in enumerate(origins_df.index):
            for j, d_idx in enumerate(destinations_df.index):
                distance = distance_matrix[i, j]
                
                # Aplicar peso baseado no tipo de origem se favor_cortes=True
                weight = 1.0
                if favor_cortes:
                    tipo = origins_df.loc[o_idx, 'Tipo'].lower() if 'Tipo' in origins_df.columns else ''
                    if 'corte' in tipo:
                        weight = 0.8  # Preferência por cortes
                    elif 'empréstimo' in tipo:
                        weight = 1.2  # Penalização para empréstimos
                
                # Adicionar termos à função objetivo
                objective += weight * distance * variables['x_cft'][(o_idx, d_idx)]
                objective += weight * distance * variables['x_ca'][(o_idx, d_idx)]
        
        return objective
    
    def _add_constraints(self, problem: pl.LpProblem, variables: Dict, opt_data: Dict):
        """Adiciona restrições ao problema de otimização"""
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        available_volumes = opt_data['available_volumes']
        needed_cft = opt_data['needed_cft']
        needed_ca = opt_data['needed_ca']
        distance_matrix = opt_data['distance_matrix']
        max_dist_cortes = opt_data['max_dist_cortes']
        max_dist_emprestimos = opt_data['max_dist_emprestimos']
        origin_types = opt_data['origin_types']
        
        # 1. Restrições de capacidade das origens
        for o_idx in origins_df.index:
            total_from_origin = 0
            for d_idx in destinations_df.index:
                total_from_origin += variables['x_cft'][(o_idx, d_idx)]
                total_from_origin += variables['x_ca'][(o_idx, d_idx)]
            
            problem += total_from_origin <= available_volumes[o_idx], f"Capacidade_Origem_{o_idx}"
        
        # 2. Restrições de demanda CFT nos destinos
        for d_idx in destinations_df.index:
            if needed_cft[d_idx] > 0:
                total_cft_to_dest = 0
                for o_idx in origins_df.index:
                    total_cft_to_dest += variables['x_cft'][(o_idx, d_idx)]
                
                problem += total_cft_to_dest >= needed_cft[d_idx], f"Demanda_CFT_{d_idx}"
        
        # 3. Restrições de demanda CA nos destinos  
        for d_idx in destinations_df.index:
            if needed_ca[d_idx] > 0:
                total_ca_to_dest = 0
                for o_idx in origins_df.index:
                    total_ca_to_dest += variables['x_ca'][(o_idx, d_idx)]
                
                problem += total_ca_to_dest >= needed_ca[d_idx], f"Demanda_CA_{d_idx}"
        
        # 4. Restrições de ISC para CFT
        for d_idx in destinations_df.index:
            isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido']
            if pd.notna(isc_min) and needed_cft[d_idx] > 0:
                for o_idx in origins_df.index:
                    if origins_df.loc[o_idx, 'ISC'] < isc_min:
                        problem += variables['x_cft'][(o_idx, d_idx)] == 0, f"ISC_CFT_{o_idx}_{d_idx}"
        
        # 5. Restrições de distância máxima
        if max_dist_cortes or max_dist_emprestimos:
            for i, o_idx in enumerate(origins_df.index):
                for j, d_idx in enumerate(destinations_df.index):
                    distance = distance_matrix[i, j]
                    
                    # Verificar tipo de origem e aplicar restrição apropriada
                    if max_dist_cortes and o_idx in origin_types['cortes']:
                        if distance > max_dist_cortes:
                            problem += variables['x_cft'][(o_idx, d_idx)] == 0, f"MaxDist_Corte_CFT_{o_idx}_{d_idx}"
                            problem += variables['x_ca'][(o_idx, d_idx)] == 0, f"MaxDist_Corte_CA_{o_idx}_{d_idx}"
                    
                    if max_dist_emprestimos and (o_idx in origin_types['emprestimos_laterais'] or 
                                                 o_idx in origin_types['emprestimos_concentrados']):
                        if distance > max_dist_emprestimos:
                            problem += variables['x_cft'][(o_idx, d_idx)] == 0, f"MaxDist_Emp_CFT_{o_idx}_{d_idx}"
                            problem += variables['x_ca'][(o_idx, d_idx)] == 0, f"MaxDist_Emp_CA_{o_idx}_{d_idx}"
    
    def _process_solution(
        self, 
        problem: pl.LpProblem, 
        variables: Dict, 
        opt_data: Dict, 
        status: int
    ) -> Dict[str, Any]:
        """Processa a solução do problema de otimização"""
        
        status_mapping = {
            pl.LpStatusOptimal: 'Optimal',
            pl.LpStatusNotSolved: 'NotSolved', 
            pl.LpStatusInfeasible: 'Infeasible',
            pl.LpStatusUnbounded: 'Unbounded',
            pl.LpStatusUndefined: 'Undefined'
        }
        
        result = {
            'success': status == pl.LpStatusOptimal,
            'status': status_mapping.get(status, 'Unknown'),
            'objective_value': problem.objective.value() if status == pl.LpStatusOptimal else None
        }
        
        if status == pl.LpStatusOptimal:
            result.update(self._extract_solution_details(variables, opt_data))
        elif status == pl.LpStatusInfeasible:
            result['error'] = ERROR_MESSAGES['no_feasible_solution']
        else:
            result['error'] = f"Otimização falhou com status: {result['status']}"
        
        return result
    
    def _extract_solution_details(self, variables: Dict, opt_data: Dict) -> Dict[str, Any]:
        """Extrai detalhes da solução ótima"""
        origins_df = opt_data['origins_df']
        destinations_df = opt_data['destinations_df']
        distance_matrix = opt_data['distance_matrix']
        
        # Extrair alocações
        allocations = []
        total_distance = 0
        total_volume = 0
        
        for i, o_idx in enumerate(origins_df.index):
            for j, d_idx in enumerate(destinations_df.index):
                # CFT
                if variables['x_cft'][(o_idx, d_idx)].varValue > 1e-6:
                    volume = variables['x_cft'][(o_idx, d_idx)].varValue
                    distance = distance_matrix[i, j]
                    
                    allocations.append({
                        'origem': o_idx,
                        'destino': d_idx,
                        'tipo_material': 'CFT',
                        'volume': volume,
                        'distancia': distance,
                        'custo_transporte': distance * volume
                    })
                    
                    total_distance += distance * volume
                    total_volume += volume
                
                # CA
                if variables['x_ca'][(o_idx, d_idx)].varValue > 1e-6:
                    volume = variables['x_ca'][(o_idx, d_idx)].varValue
                    distance = distance_matrix[i, j]
                    
                    allocations.append({
                        'origem': o_idx,
                        'destino': d_idx,
                        'tipo_material': 'CA',
                        'volume': volume,
                        'distancia': distance,
                        'custo_transporte': distance * volume
                    })
                    
                    total_distance += distance * volume
                    total_volume += volume
        
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
            'allocations': allocations_df,
            'summary': summary
        }

"""
Calculadora de distâncias para otimização de terraplenagem
"""

import pandas as pd
import numpy as np
from typing import Union


class DistanceCalculator:
    """Classe responsável por calcular distâncias entre origens e destinos"""
    
    @staticmethod
    def calculate_distance(
        origin_idx: int, 
        origin_position: float, 
        dest_position: float, 
        origins_df: pd.DataFrame
    ) -> float:
        """
        Calcula a distância entre os centros de massa em metros, 
        adicionando DT fixo para empréstimos
        
        Args:
            origin_idx: Índice da origem
            origin_position: Posição do centro de massa da origem
            dest_position: Posição do centro de massa do destino
            origins_df: DataFrame com dados das origens
            
        Returns:
            float: Distância calculada em metros
        """
        # Distância básica entre os centros de massa
        base_distance = abs(dest_position - origin_position)
        
        # Verifica se a origem é um empréstimo e tem DT fixo definido
        if 'DT Fixo (m)' in origins_df.columns:
            dt_fixo = origins_df.loc[origin_idx, 'DT Fixo (m)']
            if pd.notna(dt_fixo) and dt_fixo > 0:
                # Adiciona o DT fixo à distância básica
                return base_distance + float(dt_fixo)
        
        return base_distance
    
    @staticmethod
    def calculate_distance_matrix(
        origins_df: pd.DataFrame, 
        destinations_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calcula a matriz de distâncias entre todas as origens e destinos
        
        Args:
            origins_df: DataFrame com dados das origens
            destinations_df: DataFrame com dados dos destinos
            
        Returns:
            np.ndarray: Matriz de distâncias [num_origins x num_destinations]
        """
        num_origins = len(origins_df)
        num_destinations = len(destinations_df)
        
        distance_matrix = np.zeros((num_origins, num_destinations))
        
        for o_idx in range(num_origins):
            origin_pos = origins_df.iloc[o_idx]['Centro de Massa (m)']
            
            for d_idx in range(num_destinations):
                dest_pos = destinations_df.iloc[d_idx]['Centro de Massa (m)']
                
                distance_matrix[o_idx, d_idx] = DistanceCalculator.calculate_distance(
                    origins_df.index[o_idx], origin_pos, dest_pos, origins_df
                )
        
        return distance_matrix
    
    @staticmethod
    def get_closest_origins(
        destination_idx: int,
        destinations_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        max_distance: float = None,
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Encontra as origens mais próximas a um destino específico
        
        Args:
            destination_idx: Índice do destino
            destinations_df: DataFrame com dados dos destinos
            origins_df: DataFrame com dados das origens
            max_distance: Distância máxima a considerar
            top_n: Número máximo de origens a retornar
            
        Returns:
            pd.DataFrame: Origens ordenadas por distância
        """
        dest_pos = destinations_df.loc[destination_idx, 'Centro de Massa (m)']
        
        distances = []
        for o_idx, o_row in origins_df.iterrows():
            origin_pos = o_row['Centro de Massa (m)']
            distance = DistanceCalculator.calculate_distance(
                o_idx, origin_pos, dest_pos, origins_df
            )
            distances.append({
                'origem_idx': o_idx,
                'distancia': distance,
                'volume_disponivel': o_row['Volume disponível (m³)'],
                'isc': o_row['ISC'],
                'tipo': o_row.get('Tipo', 'Desconhecido')
            })
        
        distances_df = pd.DataFrame(distances)
        distances_df = distances_df.sort_values('distancia')
        
        # Aplicar filtros se especificados
        if max_distance is not None:
            distances_df = distances_df[distances_df['distancia'] <= max_distance]
        
        if top_n is not None:
            distances_df = distances_df.head(top_n)
        
        return distances_df


class DistanceOptimizer:
    """Classe para otimizações específicas relacionadas a distância"""
    
    @staticmethod
    def calculate_transport_cost(
        distance: float, 
        volume: float, 
        cost_per_km_m3: float = 1.0
    ) -> float:
        """
        Calcula o custo de transporte baseado na distância e volume
        
        Args:
            distance: Distância em metros
            volume: Volume em m³
            cost_per_km_m3: Custo por km por m³
            
        Returns:
            float: Custo de transporte
        """
        distance_km = distance / 1000.0
        return distance_km * volume * cost_per_km_m3
    
    @staticmethod
    def find_optimal_allocation_greedy(
        origins_df: pd.DataFrame,
        destinations_df: pd.DataFrame,
        material_type: str = 'CFT'
    ) -> list:
        """
        Encontra uma alocação gulosa baseada na menor distância
        (útil como solução inicial para o otimizador)
        
        Args:
            origins_df: DataFrame com dados das origens
            destinations_df: DataFrame com dados dos destinos
            material_type: Tipo de material ('CFT' ou 'CA')
            
        Returns:
            list: Lista de alocações [(origem_idx, destino_idx, volume)]
        """
        allocations = []
        available_volumes = origins_df['Volume disponível (m³)'].copy()
        
        volume_col = f'Volume {material_type} (m³)'
        if volume_col not in destinations_df.columns:
            return allocations
        
        needed_volumes = destinations_df[volume_col].fillna(0).copy()
        
        # Para cada destino, encontrar a origem mais próxima disponível
        for d_idx, needed_volume in needed_volumes.items():
            if needed_volume <= 0:
                continue
            
            dest_pos = destinations_df.loc[d_idx, 'Centro de Massa (m)']
            isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido']
            
            remaining_need = needed_volume
            
            while remaining_need > 0:
                # Encontrar origem mais próxima com volume e ISC adequados
                best_origin = None
                best_distance = float('inf')
                
                for o_idx, o_row in origins_df.iterrows():
                    if available_volumes[o_idx] <= 0:
                        continue
                    
                    if material_type == 'CFT' and o_row['ISC'] < isc_min:
                        continue
                    
                    origin_pos = o_row['Centro de Massa (m)']
                    distance = DistanceCalculator.calculate_distance(
                        o_idx, origin_pos, dest_pos, origins_df
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_origin = o_idx
                
                if best_origin is None:
                    break  # Não há mais origens válidas
                
                # Alocar o máximo possível desta origem
                allocation_volume = min(remaining_need, available_volumes[best_origin])
                
                allocations.append((best_origin, d_idx, allocation_volume))
                available_volumes[best_origin] -= allocation_volume
                remaining_need -= allocation_volume
        
        return allocations

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import pulp as pl
import json
import uuid
import datetime

def calculate_distance(origin_idx, origin_position, dest_position, origins_df):
    """
    Calcula a distância entre os centros de massa em metros, 
    adicionando DT fixo para empréstimos
    """
    # Distância básica entre os centros de massa
    base_distance = abs(dest_position - origin_position)
    
    # Verifica se a origem é um empréstimo e tem DT fixo definido
    if 'DT Fixo (m)' in origins_df.columns:
        if pd.notna(origins_df.loc[origin_idx, 'DT Fixo (m)']):
            # Adiciona o DT fixo à distância básica
            dt_fixo = origins_df.loc[origin_idx, 'DT Fixo (m)']
            return base_distance + dt_fixo
    
    return base_distance

def check_feasibility(origins_df, destinations_df):
    """
    Verifica se o problema tem uma solução factível
    """
    total_volume_origem = origins_df['Volume disponível (m³)'].sum()
    total_volume_destino = destinations_df['Volume CFT (m³)'].fillna(0).sum() + destinations_df['Volume CA (m³)'].fillna(0).sum()
    
    if total_volume_origem < total_volume_destino:
        return False, f"Volume total de origem ({total_volume_origem:.2f} m³) é menor que o volume total de destino ({total_volume_destino:.2f} m³)."
    
    # Verifica se há material suficiente com ISC adequado para CFT
    for d_idx, d_row in destinations_df.iterrows():
        if pd.isna(d_row['Volume CFT (m³)']) or d_row['Volume CFT (m³)'] <= 0:
            continue
            
        isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
        valid_origins = origins_df[origins_df['ISC'] >= isc_min]
        
        if valid_origins.empty:
            return False, f"Não há origens com ISC suficiente para o destino {d_idx} (ISC min: {isc_min})."
        
        total_valid_volume = valid_origins['Volume disponível (m³)'].sum()
        if total_valid_volume < d_row['Volume CFT (m³)']:
            return False, f"Volume disponível com ISC adequado ({total_valid_volume:.2f} m³) é menor que o necessário para CFT no destino {d_idx} ({d_row['Volume CFT (m³)']:.2f} m³)."
    
    return True, "O problema parece ter uma solução factível."

def identify_emprestimo_types(origins_df):
    """
    Identifica os tipos de empréstimos (laterais ou concentrados) baseado no tipo
    """
    emprestimos_laterais_idx = origins_df[
        origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True)
    ].index.tolist()
    
    emprestimos_concentrados_idx = origins_df[
        (origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)) & 
        (~origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
    ].index.tolist()
    
    cortes_idx = [idx for idx in origins_df.index 
                 if idx not in emprestimos_laterais_idx and idx not in emprestimos_concentrados_idx]
    
    return cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx

def optimize_distribution(origins_df, destinations_df, time_limit=1800, favor_cortes=False, 
                          max_dist_cortes=None, max_dist_emprestimos=None, fixed_allocations=None,
                          usar_emprestimos_bota_fora=False, priorizar_bf_proximo=True,
                          max_dist_bota_fora=None):
    """
    Otimiza a distribuição de materiais usando programação linear
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
        time_limit: Tempo limite para o solver (em segundos)
        favor_cortes: Se True, favorece materiais de corte sobre empréstimos
        max_dist_cortes: Distância máxima para transporte de material de corte (em metros)
        max_dist_emprestimos: Distância máxima para transporte de material de empréstimos (em metros)
        fixed_allocations: Lista de dicionários com alocações fixas [{'origem': 'id_origem', 
                          'destino': 'id_destino', 'volume': valor, 'tipo': 'CFT|CA'}]
        usar_emprestimos_bota_fora: Se True, permite usar empréstimos como locais de bota-fora
        priorizar_bf_proximo: Se True, prioriza o bota-fora mais próximo
        max_dist_bota_fora: Distância máxima para transporte de material para bota-fora em empréstimos
    """
    # Cria o problema de otimização
    problem = pl.LpProblem("Terraplenagem_Otimizacao", pl.LpMinimize)
    
    # Cópias dos volumes para trabalho
    available_volumes = origins_df['Volume disponível (m³)'].copy()
    needed_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    needed_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    # Processa alocações fixas, se fornecidas
    fixed_volumes_origin = {}  # Para rastrear volumes já alocados fixos por origem
    fixed_volumes_dest_cft = {}  # Para destinos CFT
    fixed_volumes_dest_ca = {}  # Para destinos CA
    
    if fixed_allocations:
        print("Aplicando alocações fixas:")
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            print(f"  Alocação fixa: {o_idx} → {d_idx}, {volume} m³ ({tipo})")
            
            # Registra o volume fixo
            if o_idx not in fixed_volumes_origin:
                fixed_volumes_origin[o_idx] = 0
            fixed_volumes_origin[o_idx] += volume
            
            # Registra por tipo
            if tipo == 'CFT':
                if d_idx not in fixed_volumes_dest_cft:
                    fixed_volumes_dest_cft[d_idx] = 0
                fixed_volumes_dest_cft[d_idx] += volume
                
                # Reduz o volume necessário
                needed_cft[d_idx] -= volume
            elif tipo == 'CA':
                if d_idx not in fixed_volumes_dest_ca:
                    fixed_volumes_dest_ca[d_idx] = 0
                fixed_volumes_dest_ca[d_idx] += volume
                
                # Reduz o volume necessário
                needed_ca[d_idx] -= volume
            
            # Reduz o volume disponível
            available_volumes[o_idx] -= volume
    
    # Identificar origens que são empréstimos
    emprestimos_idx = origins_df[
        origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)
    ].index.tolist()
    
    # Identificar origens que são cortes (não-empréstimos)
    cortes_idx = [idx for idx in origins_df.index if idx not in emprestimos_idx]
    
    print(f"Empréstimos identificados: {emprestimos_idx}")
    print(f"Cortes identificados: {cortes_idx}")
    
    # Calcula matriz de distâncias considerando DT fixo para empréstimos
    distances = {}
    adjusted_distances = {}  # Distâncias ajustadas considerando distâncias máximas
    
    for o_idx, o_row in origins_df.iterrows():
        for d_idx, d_row in destinations_df.iterrows():
            # Calcula distância básica
            dist = calculate_distance(
                o_idx,
                o_row['Centro de Massa (m)'], 
                d_row['Centro de Massa (m)'],
                origins_df
            )
            distances[(o_idx, d_idx)] = dist
            
            # Por padrão, usa a distância calculada
            adjusted_dist = dist
            
            # Aplica penalização para cortes que excedem a distância máxima
            if max_dist_cortes is not None and o_idx in cortes_idx and dist > max_dist_cortes:
                adjusted_dist = dist * 2.0  # Penalização para cortes
                print(f"Distância ajustada para corte {o_idx}->{d_idx}: {dist} -> {adjusted_dist} (excede máximo de {max_dist_cortes}m)")
            
            # Aplica penalização para empréstimos que excedem a distância máxima
            if max_dist_emprestimos is not None and o_idx in emprestimos_idx and dist > max_dist_emprestimos:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos
                print(f"Distância ajustada para empréstimo {o_idx}->{d_idx}: {dist} -> {adjusted_dist} (excede máximo de {max_dist_emprestimos}m)")
            
            adjusted_distances[(o_idx, d_idx)] = adjusted_dist
    
    # Calcular distâncias entre cortes e empréstimos para bota-fora, se usar_emprestimos_bota_fora=True
    if usar_emprestimos_bota_fora:
        bf_distances = {}
        bf_adjusted_distances = {}
        
        for o_idx in cortes_idx:  # Origem do material (cortes)
            o_row = origins_df.loc[o_idx]
            for emp_idx in emprestimos_idx:  # Destino do bota-fora (empréstimos)
                emp_row = origins_df.loc[emp_idx]
                
                # Calcula distância básica entre corte e empréstimo
                dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                
                # Adiciona DT fixo do empréstimo, se existir
                if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                    dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                
                bf_distances[(o_idx, emp_idx)] = dist
                
                # Por padrão, usa a distância calculada
                adjusted_dist = dist
                
                # Aplica penalização se exceder a distância máxima para bota-fora
                if max_dist_bota_fora is not None and dist > max_dist_bota_fora:
                    adjusted_dist = dist * 2.0  # Penalização
                    print(f"Distância ajustada para bota-fora {o_idx}->{emp_idx}: {dist} -> {adjusted_dist} (excede máximo de {max_dist_bota_fora}m)")
                
                bf_adjusted_distances[(o_idx, emp_idx)] = adjusted_dist
    
    # Normaliza as distâncias ajustadas para evitar problemas numéricos
    max_adj_distance = max(adjusted_distances.values()) if adjusted_distances else 1
    for key in adjusted_distances:
        adjusted_distances[key] /= max_adj_distance  # Normaliza para [0,1]
    
    # Normaliza as distâncias de bota-fora, se aplicável
    if usar_emprestimos_bota_fora and bf_adjusted_distances:
        max_bf_adj_distance = max(bf_adjusted_distances.values()) if bf_adjusted_distances else 1
        for key in bf_adjusted_distances:
            bf_adjusted_distances[key] /= max_bf_adj_distance  # Normaliza para [0,1]
    
    # Verifica se há alguma configuração de ISC impossível
    feasible = True
    for d_idx, d_row in destinations_df.iterrows():
        if needed_cft[d_idx] > 0:
            isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
            valid_origins = origins_df[origins_df['ISC'] >= isc_min]
            
            if valid_origins.empty:
                st.warning(f"Não há origem com ISC adequado para o destino {d_idx} (ISC min: {isc_min})")
                feasible = False
    
    if not feasible:
        st.error("Problema infactível devido a restrições de ISC incompatíveis.")
        return None  # Retorna None para indicar que o problema é infactível
    
    # Cria variáveis de decisão para CFT
    cft_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_cft[d_idx] > 0:
                # Verifica compatibilidade de ISC
                isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido']
                if pd.isna(isc_min) or origins_df.loc[o_idx, 'ISC'] >= isc_min:
                    cft_vars[(o_idx, d_idx)] = pl.LpVariable(
                        f"CFT_{o_idx}_{d_idx}", 
                        lowBound=0
                    )
    
    # Cria variáveis de decisão para CA
    ca_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_ca[d_idx] > 0:
                ca_vars[(o_idx, d_idx)] = pl.LpVariable(
                    f"CA_{o_idx}_{d_idx}", 
                    lowBound=0
                )
    
    # Cria variáveis para bota-fora convencional, mas somente para origens que não são empréstimos
    bota_fora_vars = {}
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx:  # Somente origens que não são empréstimos
            bota_fora_vars[o_idx] = pl.LpVariable(f"BF_{o_idx}", lowBound=0)
    
    # Cria variáveis para bota-fora em empréstimos, se a opção estiver habilitada
    bota_fora_emprestimo_vars = {}
    if usar_emprestimos_bota_fora:
        for o_idx in cortes_idx:  # Origem do material (cortes)
            for emp_idx in emprestimos_idx:  # Destino do bota-fora (empréstimo)
                bota_fora_emprestimo_vars[(o_idx, emp_idx)] = pl.LpVariable(
                    f"BF_EMP_{o_idx}_{emp_idx}", 
                    lowBound=0
                )
    
    # Cria variáveis de folga para destinos (para identificar onde não está atendendo)
    cft_slack_vars = {}
    ca_slack_vars = {}
    
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            cft_slack_vars[d_idx] = pl.LpVariable(f"CFT_SLACK_{d_idx}", lowBound=0)
        if needed_ca[d_idx] > 0:
            ca_slack_vars[d_idx] = pl.LpVariable(f"CA_SLACK_{d_idx}", lowBound=0)
    
    # Variáveis para volume não utilizado de cada empréstimo
    emprestimo_nao_utilizado = {}
    for o_idx in emprestimos_idx:
        emprestimo_nao_utilizado[o_idx] = pl.LpVariable(f"NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    # Função objetivo depende da escolha de favorecimento
    if favor_cortes:
        # Função objetivo que favorece materiais de corte
        objective_terms = [
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]),
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]),
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]),
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()]),
            0.1 * pl.lpSum([bota_fora_vars[o_idx] for o_idx in bota_fora_vars.keys()])  # Penalização para bota-fora
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            if priorizar_bf_proximo:
                # Usa distância como fator para priorizar o mais próximo
                objective_terms.append(
                    0.05 * pl.lpSum([
                        bf_adjusted_distances[(o_idx, emp_idx)] * bota_fora_emprestimo_vars[(o_idx, emp_idx)]
                        for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()
                    ])
                )
            else:
                # Penalização fixa, menor que bota-fora convencional
                objective_terms.append(
                    0.05 * pl.lpSum([bota_fora_emprestimo_vars[(o_idx, emp_idx)] 
                                    for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()])
                )
        
        problem += sum(objective_terms)
    else:
        # Função objetivo que prioriza apenas menor distância
        objective_terms = [
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]),
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]),
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]),
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()])
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            if priorizar_bf_proximo:
                # Usa distância como fator para priorizar o mais próximo
                objective_terms.append(
                    pl.lpSum([
                        bf_adjusted_distances[(o_idx, emp_idx)] * bota_fora_emprestimo_vars[(o_idx, emp_idx)]
                        for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()
                    ])
                )
        
        problem += sum(objective_terms)
    
    # Restrição para cortes: volume total distribuído + bota-fora (convencional e em empréstimos) = volume disponível
    for o_idx in cortes_idx:
        distribuicao_terms = [
            pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]),
            pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]),
            bota_fora_vars[o_idx]
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            distribuicao_terms.append(
                pl.lpSum([bota_fora_emprestimo_vars.get((o_idx, emp_idx), 0) 
                         for emp_idx in emprestimos_idx 
                         if (o_idx, emp_idx) in bota_fora_emprestimo_vars])
            )
        
        problem += (
            sum(distribuicao_terms) == available_volumes[o_idx],
            f"Conservacao_Volume_Corte_{o_idx}"
        )
    
    # Restrição para empréstimos: volume total distribuído + não utilizado = volume disponível
    for o_idx in emprestimos_idx:
        problem += (
            pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
            pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
            emprestimo_nao_utilizado[o_idx] == available_volumes[o_idx],
            f"Conservacao_Volume_Emprestimo_{o_idx}"
        )
    
    # Se usar empréstimos como bota-fora, adiciona restrição de que volume recebido não pode superar o retirado
    if usar_emprestimos_bota_fora:
        for emp_idx in emprestimos_idx:
            # Volume retirado do empréstimo
            volume_retirado = (
                pl.lpSum([cft_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in ca_vars])
            )
            
            # Volume de bota-fora recebido pelo empréstimo
            volume_botafora_recebido = pl.lpSum([
                bota_fora_emprestimo_vars.get((o_idx, emp_idx), 0) 
                for o_idx in cortes_idx 
                if (o_idx, emp_idx) in bota_fora_emprestimo_vars
            ])
            
            # Restrição: volume recebido não pode superar volume retirado
            problem += (
                volume_botafora_recebido <= volume_retirado,
                f"Limite_Botafora_Emprestimo_{emp_idx}"
            )
    
    # Restrição: volume recebido de CFT em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in cft_vars]) + 
                cft_slack_vars[d_idx] == needed_cft[d_idx],
                f"Atendimento_CFT_{d_idx}"
            )
    
    # Restrição: volume recebido de CA em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_ca[d_idx] > 0:
            problem += (
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in ca_vars]) + 
                ca_slack_vars[d_idx] == needed_ca[d_idx],
                f"Atendimento_CA_{d_idx}"
            )
    
    # Configurações do solver
    if len(origins_df) * len(destinations_df) > 10000:  # Problemas muito grandes
        st.warning("Problema muito grande detectado. Ajustando parâmetros do solver...")
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            gapRel=0.05,
            options=['presolve on', 'strong branching on', 'gomory on']
        )
    else:
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit,
            gapRel=0.01,
            options=['presolve on']
        )
    
    # Resolve o problema
    print("Iniciando otimização...")
    status = problem.solve(solver)
    print("Otimização concluída!")
    
    # Verifica o status da solução
    status_text = pl.LpStatus[status]
    print(f"Status da otimização: {status_text}")
    print(f"Valor da função objetivo: {pl.value(problem.objective)}")
    
    # Extrai os resultados
    cft_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    ca_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    bota_fora = pd.Series(0, index=origins_df.index)
    
    # Para bota-fora em empréstimos, se habilitado
    bota_fora_emprestimo = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx) if usar_emprestimos_bota_fora else None
    
    # Adiciona primeiro as alocações fixas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            if tipo == 'CFT':
                cft_distribution.loc[o_idx, d_idx] = volume
            elif tipo == 'CA':
                ca_distribution.loc[o_idx, d_idx] = volume
    
    # Adiciona as alocações calculadas pelo otimizador
    for (o_idx, d_idx), var in cft_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            cft_distribution.loc[o_idx, d_idx] += var.value()
    
    for (o_idx, d_idx), var in ca_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            ca_distribution.loc[o_idx, d_idx] += var.value()
    
    # Somente origens não-empréstimo vão para bota-fora convencional
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx and o_idx in bota_fora_vars:
            var = bota_fora_vars[o_idx]
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora[o_idx] = var.value()
    
    # Extrair resultados de bota-fora em empréstimos, se habilitado
    if usar_emprestimos_bota_fora:
        for (o_idx, emp_idx), var in bota_fora_emprestimo_vars.items():
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora_emprestimo.loc[o_idx, emp_idx] = var.value()
    
    # Verifica valores de folga (não atendimento)
    cft_slack = {}
    ca_slack = {}
    for d_idx, var in cft_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            cft_slack[d_idx] = var.value()
            print(f"CFT não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    for d_idx, var in ca_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            ca_slack[d_idx] = var.value()
            print(f"CA não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    # Volume não utilizado de empréstimos
    emprestimos_nao_utilizados = {}
    for o_idx, var in emprestimo_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo {o_idx}: {var.value():.2f} m³")
    
    # Se houver problemas sérios com o não atendimento, retorna None
    if (sum(cft_slack.values()) > 0.05 * destinations_df['Volume CFT (m³)'].fillna(0).sum() or 
        sum(ca_slack.values()) > 0.05 * destinations_df['Volume CA (m³)'].fillna(0).sum()):
        print("Não atendimento significativo.")
        st.warning("O solver não conseguiu atender a um percentual significativo dos volumes necessários.")
    
    # Calcula volumes não atendidos
    # Considerando os volumes originais (incluindo os fixados)
    original_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    original_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    distributed_cft = cft_distribution.sum(axis=0)
    distributed_ca = ca_distribution.sum(axis=0)
    
    remaining_cft = original_cft - distributed_cft
    remaining_ca = original_ca - distributed_ca
    
        # Substituindo valores negativos pequenos por zero
    remaining_cft = remaining_cft.map(lambda x: max(0, x))
    remaining_ca = remaining_ca.map(lambda x: max(0, x))
    
    # Desnormaliza as distâncias originais para cálculos de resultado
    distances_df = pd.DataFrame(index=origins_df.index, columns=destinations_df.index)
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            distances_df.loc[o_idx, d_idx] = distances[(o_idx, d_idx)]
    
    # Calcular momento de transporte (m³·m) usando distâncias REAIS
    momento_cft = sum(cft_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                     for o_idx in origins_df.index
                     for d_idx in destinations_df.index
                     if cft_distribution.loc[o_idx, d_idx] > 0)
    
    momento_ca = sum(ca_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                    for o_idx in origins_df.index
                    for d_idx in destinations_df.index
                    if ca_distribution.loc[o_idx, d_idx] > 0)
    
    momento_total = momento_cft + momento_ca
    
    # Momento de bota-fora em empréstimos, se aplicável
    momento_bf_emprestimo = 0
    if usar_emprestimos_bota_fora and bota_fora_emprestimo is not None:
        bf_distances_df = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
        for o_idx in cortes_idx:
            for emp_idx in emprestimos_idx:
                if (o_idx, emp_idx) in bf_distances:
                    bf_distances_df.loc[o_idx, emp_idx] = bf_distances[(o_idx, emp_idx)]
        
        momento_bf_emprestimo = sum(bota_fora_emprestimo.loc[o_idx, emp_idx] * bf_distances_df.loc[o_idx, emp_idx]
                                   for o_idx in cortes_idx
                                   for emp_idx in emprestimos_idx
                                   if bota_fora_emprestimo.loc[o_idx, emp_idx] > 0)
    
    # Distância média de transporte (m)
    volume_total_distribuido = cft_distribution.sum().sum() + ca_distribution.sum().sum()
    if volume_total_distribuido > 0:
        dmt = momento_total / volume_total_distribuido
    else:
        dmt = 0
    
    # Volume total não utilizado de empréstimos
    total_emprestimo_nao_utilizado = sum(emprestimos_nao_utilizados.values())
    
    # Volume total em bota-fora em empréstimos, se aplicável
    total_bf_emprestimo = 0
    if usar_emprestimos_bota_fora and bota_fora_emprestimo is not None:
        total_bf_emprestimo = bota_fora_emprestimo.sum().sum()
    
    result = {
        'cft': cft_distribution,
        'ca': ca_distribution,
        'bota_fora': bota_fora,
        'distances': distances_df,
        'remaining_cft': remaining_cft,
        'remaining_ca': remaining_ca,
        'momento_total': momento_total,
        'dmt': dmt,
        'status': status_text,
        'emprestimos_nao_utilizados': emprestimos_nao_utilizados,
        'total_emprestimo_nao_utilizado': total_emprestimo_nao_utilizado,
        'favor_cortes': favor_cortes,
        'max_dist_cortes': max_dist_cortes,
        'max_dist_emprestimos': max_dist_emprestimos,
        'fixed_allocations': fixed_allocations if fixed_allocations else []
    }
    
    # Adiciona informações sobre bota-fora em empréstimos, se aplicável
    if usar_emprestimos_bota_fora:
        result['bota_fora_emprestimo'] = bota_fora_emprestimo
        result['total_bf_emprestimo'] = total_bf_emprestimo
        result['momento_bf_emprestimo'] = momento_bf_emprestimo
        result['usar_emprestimos_bota_fora'] = True
        result['priorizar_bf_proximo'] = priorizar_bf_proximo
        result['max_dist_bota_fora'] = max_dist_bota_fora

    return result

def optimize_distribution_advanced(origins_df, destinations_df, time_limit=1800, favor_cortes=False, 
                                  max_dist_cortes=None, max_dist_emprestimos_laterais=None, 
                                  max_dist_emprestimos_concentrados=None, fixed_allocations=None,
                                  cortes_idx=None, emprestimos_laterais_idx=None, emprestimos_concentrados_idx=None,
                                  usar_emprestimos_bota_fora=False, priorizar_bf_proximo=True,
                                  max_dist_bota_fora=None):
    """
    Versão avançada da função optimize_distribution que considera diferentes tipos de empréstimos
    e permite usar empréstimos como bota-fora
    """
    # Código de diagnóstico (opcional - pode ser removido após a depuração)
    print("=== DIAGNÓSTICO DE TIPOS DE DADOS ===")
    print(f"Tipo de origins_df: {type(origins_df)}")
    print(f"Tipo de destinations_df: {type(destinations_df)}")
    
    # Cria o problema de otimização
    problem = pl.LpProblem("Terraplenagem_Otimizacao", pl.LpMinimize)
    
    # Cópias dos volumes para trabalho
    available_volumes = origins_df['Volume disponível (m³)'].copy()
    needed_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    needed_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    # Código de diagnóstico (opcional)
    print("=== DADOS DE VOLUMES ===")
    print(f"Tipo de needed_cft: {type(needed_cft)}")
    print(f"Tipo de needed_ca: {type(needed_ca)}")
    
    # Processa alocações fixas, se fornecidas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            # Reduz volumes conforme alocações fixas
            if tipo == 'CFT':
                if isinstance(needed_cft, pd.Series):
                    needed_cft.loc[d_idx] -= volume
                else:
                    needed_cft[d_idx] -= volume
            elif tipo == 'CA':
                if isinstance(needed_ca, pd.Series):
                    needed_ca.loc[d_idx] -= volume
                else:
                    needed_ca[d_idx] -= volume
            
            # Reduz o volume disponível
            available_volumes[o_idx] -= volume
    
    # Se não foram fornecidos, identifica os tipos de origens
    if cortes_idx is None or emprestimos_laterais_idx is None or emprestimos_concentrados_idx is None:
        cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx = identify_emprestimo_types(origins_df)
    
    # União de todos os tipos de empréstimos para regras gerais
    emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
    
    # Calcula matriz de distâncias considerando DT fixo para empréstimos
    distances = {}
    adjusted_distances = {}  # Distâncias ajustadas considerando distâncias máximas
    
    for o_idx, o_row in origins_df.iterrows():
        for d_idx, d_row in destinations_df.iterrows():
            # Calcula distância básica
            dist = calculate_distance(
                o_idx,
                o_row['Centro de Massa (m)'], 
                d_row['Centro de Massa (m)'],
                origins_df
            )
            distances[(o_idx, d_idx)] = dist
            
            # Por padrão, usa a distância calculada
            adjusted_dist = dist
            
            # Aplica penalização para cortes que excedem a distância máxima
            if max_dist_cortes is not None and o_idx in cortes_idx and dist > max_dist_cortes:
                adjusted_dist = dist * 2.0  # Penalização para cortes
            
            # Aplica penalização para empréstimos laterais que excedem a distância máxima
            if max_dist_emprestimos_laterais is not None and o_idx in emprestimos_laterais_idx and dist > max_dist_emprestimos_laterais:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos laterais
            
            # Aplica penalização para empréstimos concentrados que excedem a distância máxima
            if max_dist_emprestimos_concentrados is not None and o_idx in emprestimos_concentrados_idx and dist > max_dist_emprestimos_concentrados:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos concentrados
            
            adjusted_distances[(o_idx, d_idx)] = adjusted_dist
    
    # Calcula distâncias entre cortes e empréstimos para bota-fora, se aplicável
    bf_distances = {}
    bf_adjusted_distances = {}
    if usar_emprestimos_bota_fora:
        for o_idx in cortes_idx:  # Origem do material (cortes)
            o_row = origins_df.loc[o_idx]
            for emp_idx in emprestimos_idx:  # Destino do bota-fora (empréstimos)
                emp_row = origins_df.loc[emp_idx]
                
                # Calcula distância básica entre corte e empréstimo
                dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                
                # Adiciona DT fixo do empréstimo, se existir
                if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                    dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                
                bf_distances[(o_idx, emp_idx)] = dist
                
                # Por padrão, usa a distância calculada
                adjusted_dist = dist
                
                # Aplica penalização se exceder a distância máxima para bota-fora
                if max_dist_bota_fora is not None and dist > max_dist_bota_fora:
                    adjusted_dist = dist * 2.0  # Penalização
                
                bf_adjusted_distances[(o_idx, emp_idx)] = adjusted_dist
    
    # Normaliza as distâncias ajustadas para evitar problemas numéricos
    max_adj_distance = max(adjusted_distances.values()) if adjusted_distances else 1
    for key in adjusted_distances:
        adjusted_distances[key] /= max_adj_distance  # Normaliza para [0,1]
    
    # Normaliza as distâncias de bota-fora, se aplicável
    if usar_emprestimos_bota_fora and bf_adjusted_distances:
        max_bf_adj_distance = max(bf_adjusted_distances.values()) if bf_adjusted_distances else 1
        for key in bf_adjusted_distances:
            bf_adjusted_distances[key] /= max_bf_adj_distance  # Normaliza para [0,1]
    
    # Verifica se há alguma configuração de ISC impossível
    feasible = True
    for d_idx, d_row in destinations_df.iterrows():
        try:
            # Verifica se há volume CFT necessário para este destino
            if isinstance(needed_cft, pd.Series):
                cft_value = needed_cft.loc[d_idx] if d_idx in needed_cft.index else 0
            else:
                cft_value = needed_cft.get(d_idx, 0)
                
            # Garantir que cft_value é um escalar
            if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                cft_value = cft_value.iloc[0] if not cft_value.empty else 0
                
            if float(cft_value) > 0:
                isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
                valid_origins = origins_df[origins_df['ISC'] >= isc_min]
                
                if valid_origins.empty:
                    feasible = False
        except Exception as e:
            print(f"Erro ao verificar factibilidade para destino {d_idx}: {str(e)}")
            continue
    
    if not feasible:
        return None  # Retorna None para indicar que o problema é infactível
    
    # Cria variáveis de decisão para CFT
    cft_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            try:
                # Acesso seguro ao valor de needed_cft para este destino
                if isinstance(needed_cft, pd.Series):
                    cft_value = needed_cft.loc[d_idx] if d_idx in needed_cft.index else 0
                else:
                    cft_value = needed_cft.get(d_idx, 0)
                    
                # Certifique-se de que cft_value é um escalar
                if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                    cft_value = cft_value.iloc[0] if not cft_value.empty else 0
                    
                if float(cft_value) > 0:
                    # Verifica compatibilidade de ISC
                    isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido'] if pd.notna(destinations_df.loc[d_idx, 'ISC mínimo exigido']) else 0
                    origin_isc = origins_df.loc[o_idx, 'ISC']
                    
                    if pd.isna(isc_min) or float(origin_isc) >= float(isc_min):
                        cft_vars[(o_idx, d_idx)] = pl.LpVariable(
                            f"CFT_{o_idx}_{d_idx}", 
                            lowBound=0
                        )
            except Exception as e:
                print(f"Erro ao processar CFT para origem {o_idx}, destino {d_idx}: {str(e)}")
                continue
    
    # Cria variáveis de decisão para CA
    ca_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            try:
                # Acesso seguro ao valor de needed_ca para este destino
                if isinstance(needed_ca, pd.Series):
                    ca_value = needed_ca.loc[d_idx] if d_idx in needed_ca.index else 0
                else:
                    ca_value = needed_ca.get(d_idx, 0)
                    
                # Certifique-se de que ca_value é um escalar
                if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                    ca_value = ca_value.iloc[0] if not ca_value.empty else 0
                    
                if float(ca_value) > 0:
                    ca_vars[(o_idx, d_idx)] = pl.LpVariable(
                        f"CA_{o_idx}_{d_idx}", 
                        lowBound=0
                    )
            except Exception as e:
                print(f"Erro ao processar CA para origem {o_idx}, destino {d_idx}: {str(e)}")
                continue
    
    # Cria variáveis para bota-fora, mas somente para origens que não são empréstimos
    bota_fora_vars = {}
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx:  # Somente origens que não são empréstimos
            bota_fora_vars[o_idx] = pl.LpVariable(f"BF_{o_idx}", lowBound=0)
    
    # Cria variáveis para bota-fora em empréstimos, se a opção estiver habilitada
    bota_fora_emprestimo_vars = {}
    if usar_emprestimos_bota_fora:
        for o_idx in cortes_idx:  # Origem do material (cortes)
            for emp_idx in emprestimos_idx:  # Destino do bota-fora (empréstimo)
                bota_fora_emprestimo_vars[(o_idx, emp_idx)] = pl.LpVariable(
                    f"BF_EMP_{o_idx}_{emp_idx}", 
                    lowBound=0
                )
    
    # Cria variáveis de folga para destinos (para identificar onde não está atendendo)
    cft_slack_vars = {}
    ca_slack_vars = {}
    
    for d_idx in destinations_df.index:
        try:
            # Acesso seguro aos valores
            if isinstance(needed_cft, pd.Series):
                cft_value = needed_cft.loc[d_idx] if d_idx in needed_cft.index else 0
            else:
                cft_value = needed_cft.get(d_idx, 0)
                
            if isinstance(needed_ca, pd.Series):
                ca_value = needed_ca.loc[d_idx] if d_idx in needed_ca.index else 0
            else:
                ca_value = needed_ca.get(d_idx, 0)
            
            # Certifique-se de que os valores são escalares
            if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                cft_value = cft_value.iloc[0] if not cft_value.empty else 0
                
            if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                ca_value = ca_value.iloc[0] if not ca_value.empty else 0
            
            if float(cft_value) > 0:
                cft_slack_vars[d_idx] = pl.LpVariable(f"CFT_SLACK_{d_idx}", lowBound=0)
            if float(ca_value) > 0:
                ca_slack_vars[d_idx] = pl.LpVariable(f"CA_SLACK_{d_idx}", lowBound=0)
        except Exception as e:
            print(f"Erro ao processar variáveis de folga para destino {d_idx}: {str(e)}")
            continue
    
    # Variáveis para volume não utilizado de cada empréstimo, separado por tipo
    emprestimo_lateral_nao_utilizado = {}
    for o_idx in emprestimos_laterais_idx:
        emprestimo_lateral_nao_utilizado[o_idx] = pl.LpVariable(f"LATERAL_NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    emprestimo_concentrado_nao_utilizado = {}
    for o_idx in emprestimos_concentrados_idx:
        emprestimo_concentrado_nao_utilizado[o_idx] = pl.LpVariable(f"CONCENTRADO_NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    # Função objetivo depende da escolha de favorecimento
    if favor_cortes:
        # Função objetivo que favorece materiais de corte
        objective_terms = [
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]),
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]),
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]),
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()]),
            0.1 * pl.lpSum([bota_fora_vars[o_idx] for o_idx in bota_fora_vars.keys()])  # Penalização para bota-fora
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            if priorizar_bf_proximo:
                # Usa distância como fator para priorizar o mais próximo
                objective_terms.append(
                    0.05 * pl.lpSum([
                        bf_adjusted_distances[(o_idx, emp_idx)] * bota_fora_emprestimo_vars[(o_idx, emp_idx)]
                        for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()
                    ])
                )
            else:
                # Penalização fixa, menor que bota-fora convencional
                objective_terms.append(
                    0.05 * pl.lpSum([bota_fora_emprestimo_vars[(o_idx, emp_idx)] 
                                    for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()])
                )
        
        problem += sum(objective_terms)
    else:
        # Função objetivo que prioriza apenas menor distância
        objective_terms = [
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]),
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]),
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]),
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()])
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            if priorizar_bf_proximo:
                # Usa distância como fator para priorizar o mais próximo
                objective_terms.append(
                    pl.lpSum([
                        bf_adjusted_distances[(o_idx, emp_idx)] * bota_fora_emprestimo_vars[(o_idx, emp_idx)]
                        for (o_idx, emp_idx) in bota_fora_emprestimo_vars.keys()
                    ])
                )
        
        problem += sum(objective_terms)
    
    # Restrição para cortes: volume total distribuído + bota-fora (convencional e em empréstimos) = volume disponível
    for o_idx in cortes_idx:
        distribuicao_terms = [
            pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]),
            pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]),
            bota_fora_vars[o_idx]
        ]
        
        # Adiciona termo para bota-fora em empréstimos, se aplicável
        if usar_emprestimos_bota_fora:
            distribuicao_terms.append(
                pl.lpSum([bota_fora_emprestimo_vars.get((o_idx, emp_idx), 0) 
                         for emp_idx in emprestimos_idx 
                         if (o_idx, emp_idx) in bota_fora_emprestimo_vars])
            )
        
        problem += (
            sum(distribuicao_terms) == available_volumes[o_idx],
            f"Conservacao_Volume_Corte_{o_idx}"
        )
    
    # Restrição para empréstimos laterais: volume total distribuído + não utilizado = volume disponível
    for o_idx in emprestimos_laterais_idx:
        problem += (
            pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
            pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
            emprestimo_lateral_nao_utilizado[o_idx] == available_volumes[o_idx],
            f"Conservacao_Volume_Emp_Lateral_{o_idx}"
        )
    
    # Restrição para empréstimos concentrados: volume total distribuído + não utilizado = volume disponível
    for o_idx in emprestimos_concentrados_idx:
        problem += (
            pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
            pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
            emprestimo_concentrado_nao_utilizado[o_idx] == available_volumes[o_idx],
            f"Conservacao_Volume_Emp_Concentrado_{o_idx}"
        )
    
    # Se usar empréstimos como bota-fora, adiciona restrição de que volume recebido não pode superar o retirado
    if usar_emprestimos_bota_fora:
        # Para empréstimos laterais
        for emp_idx in emprestimos_laterais_idx:
            # Volume retirado do empréstimo
            volume_retirado = (
                pl.lpSum([cft_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in ca_vars])
            )
            
            # Volume de bota-fora recebido pelo empréstimo
            volume_botafora_recebido = pl.lpSum([
                bota_fora_emprestimo_vars.get((o_idx, emp_idx), 0) 
                for o_idx in cortes_idx 
                if (o_idx, emp_idx) in bota_fora_emprestimo_vars
            ])
            
            # Restrição: volume recebido não pode superar volume retirado
            problem += (
                volume_botafora_recebido <= volume_retirado,
                f"Limite_Botafora_Emprestimo_Lateral_{emp_idx}"
            )
        
        # Para empréstimos concentrados
        for emp_idx in emprestimos_concentrados_idx:
            # Volume retirado do empréstimo
            volume_retirado = (
                pl.lpSum([cft_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((emp_idx, d_idx), 0) for d_idx in destinations_df.index if (emp_idx, d_idx) in ca_vars])
            )
            
            # Volume de bota-fora recebido pelo empréstimo
            volume_botafora_recebido = pl.lpSum([
                bota_fora_emprestimo_vars.get((o_idx, emp_idx), 0) 
                for o_idx in cortes_idx 
                if (o_idx, emp_idx) in bota_fora_emprestimo_vars
            ])
            
            # Restrição: volume recebido não pode superar volume retirado
            problem += (
                volume_botafora_recebido <= volume_retirado,
                f"Limite_Botafora_Emprestimo_Concentrado_{emp_idx}"
            )
    
    # Restrição: volume recebido de CFT em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        try:
            # Acesso seguro ao valor
            if isinstance(needed_cft, pd.Series):
                cft_value = needed_cft.loc[d_idx] if d_idx in needed_cft.index else 0
            else:
                cft_value = needed_cft.get(d_idx, 0)
                
            # Certifique-se de que é um escalar
            if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                cft_value = cft_value.iloc[0] if not cft_value.empty else 0
                
            cft_value = float(cft_value)
            
            if cft_value > 0:
                problem += (
                    pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in cft_vars]) + 
                    cft_slack_vars[d_idx] == cft_value,
                    f"Atendimento_CFT_{d_idx}"
                )
        except Exception as e:
            print(f"Erro ao criar restrição CFT para destino {d_idx}: {str(e)}")
            continue
    
    # Restrição: volume recebido de CA em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        try:
            # Acesso seguro ao valor
            if isinstance(needed_ca, pd.Series):
                ca_value = needed_ca.loc[d_idx] if d_idx in needed_ca.index else 0
            else:
                ca_value = needed_ca.get(d_idx, 0)
                
            # Certifique-se de que é um escalar
            if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                ca_value = ca_value.iloc[0] if not ca_value.empty else 0
                
            ca_value = float(ca_value)
            
            if ca_value > 0:
                problem += (
                    pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in ca_vars]) + 
                    ca_slack_vars[d_idx] == ca_value,
                    f"Atendimento_CA_{d_idx}"
                )
        except Exception as e:
            print(f"Erro ao criar restrição CA para destino {d_idx}: {str(e)}")
            continue
    
    # Configurações do solver
    if len(origins_df) * len(destinations_df) > 10000:  # Problemas muito grandes
        print("Problema muito grande detectado. Ajustando parâmetros do solver...")
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            gapRel=0.05,
            options=['presolve on', 'strong branching on', 'gomory on']
        )
    else:
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit,
            gapRel=0.01,
            options=['presolve on']
        )
    
    # Resolve o problema
    print("Iniciando otimização avançada...")
    status = problem.solve(solver)
    print("Otimização concluída!")
    
    # Verifica o status da solução
    status_text = pl.LpStatus[status]
    print(f"Status da otimização: {status_text}")
    print(f"Valor da função objetivo: {pl.value(problem.objective)}")
    
    # Extrai os resultados
    cft_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    ca_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    bota_fora = pd.Series(0, index=origins_df.index)
    
    # Para bota-fora em empréstimos, se habilitado
    bota_fora_emprestimo = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx) if usar_emprestimos_bota_fora else None
    
    # Adiciona primeiro as alocações fixas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            if tipo == 'CFT':
                cft_distribution.loc[o_idx, d_idx] = volume
            elif tipo == 'CA':
                ca_distribution.loc[o_idx, d_idx] = volume
    
    # Adiciona as alocações calculadas pelo otimizador
    for (o_idx, d_idx), var in cft_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            cft_distribution.loc[o_idx, d_idx] += var.value()
    
    for (o_idx, d_idx), var in ca_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            ca_distribution.loc[o_idx, d_idx] += var.value()
    
    # Somente origens não-empréstimo vão para bota-fora
    for o_idx in cortes_idx:
        if o_idx in bota_fora_vars:
            var = bota_fora_vars[o_idx]
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora[o_idx] = var.value()
    
    # Extrair resultados de bota-fora em empréstimos, se habilitado
    if usar_emprestimos_bota_fora:
        for (o_idx, emp_idx), var in bota_fora_emprestimo_vars.items():
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora_emprestimo.loc[o_idx, emp_idx] = var.value()
    
        # Verifica valores de folga (não atendimento)
    cft_slack = {}
    ca_slack = {}
    for d_idx, var in cft_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            cft_slack[d_idx] = var.value()
            print(f"CFT não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    for d_idx, var in ca_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            ca_slack[d_idx] = var.value()
            print(f"CA não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    # Volume não utilizado de empréstimos separados por tipo
    emprestimos_laterais_nao_utilizados = {}
    for o_idx, var in emprestimo_lateral_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_laterais_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo lateral {o_idx}: {var.value():.2f} m³")
    
    emprestimos_concentrados_nao_utilizados = {}
    for o_idx, var in emprestimo_concentrado_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_concentrados_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo concentrado {o_idx}: {var.value():.2f} m³")
    
    # Unindo todos os empréstimos não utilizados para análise
    emprestimos_nao_utilizados = {**emprestimos_laterais_nao_utilizados, **emprestimos_concentrados_nao_utilizados}
    
    # Se houver problemas sérios com o não atendimento, retorna aviso
    if (sum(cft_slack.values()) > 0.05 * destinations_df['Volume CFT (m³)'].fillna(0).sum() or 
        sum(ca_slack.values()) > 0.05 * destinations_df['Volume CA (m³)'].fillna(0).sum()):
        print("Não atendimento significativo.")
    
    # Calcula volumes não atendidos
    # Considerando os volumes originais (incluindo os fixados)
    original_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    original_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    distributed_cft = cft_distribution.sum(axis=0)
    distributed_ca = ca_distribution.sum(axis=0)
    
    remaining_cft = original_cft - distributed_cft
    remaining_ca = original_ca - distributed_ca
    
    # Substituindo valores negativos pequenos por zero
    remaining_cft = remaining_cft.map(lambda x: max(0, x))
    remaining_ca = remaining_ca.map(lambda x: max(0, x))
    
    # Desnormaliza as distâncias originais para cálculos de resultado
    distances_df = pd.DataFrame(index=origins_df.index, columns=destinations_df.index)
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            distances_df.loc[o_idx, d_idx] = distances[(o_idx, d_idx)]
    
    # Calcular momento de transporte (m³·m) usando distâncias REAIS - VERSÃO CORRIGIDA
    # Método seguro com loops e verificação de tipos
    momento_cft = 0
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            try:
                # Acessar o valor de forma segura e converter para escalar
                value = cft_distribution.loc[o_idx, d_idx]
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    value = value.iloc[0] if not value.empty else 0
                value = float(value)
                
                # Somente calcular para valores positivos
                if value > 0:
                    # Acessar a distância de forma segura
                    dist = distances_df.loc[o_idx, d_idx]
                    if isinstance(dist, (pd.Series, pd.DataFrame)):
                        dist = dist.iloc[0] if not dist.empty else 0
                    dist = float(dist)
                    
                    # Adicionar ao momento total
                    momento_cft += value * dist
            except Exception as e:
                print(f"Erro ao calcular momento CFT para origem {o_idx}, destino {d_idx}: {str(e)}")
                continue

    # Cálculo similar para CA
    momento_ca = 0
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            try:
                # Acessar o valor de forma segura e converter para escalar
                value = ca_distribution.loc[o_idx, d_idx]
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    value = value.iloc[0] if not value.empty else 0
                value = float(value)
                
                # Somente calcular para valores positivos
                if value > 0:
                    # Acessar a distância de forma segura
                    dist = distances_df.loc[o_idx, d_idx]
                    if isinstance(dist, (pd.Series, pd.DataFrame)):
                        dist = dist.iloc[0] if not dist.empty else 0
                    dist = float(dist)
                    
                    # Adicionar ao momento total
                    momento_ca += value * dist
            except Exception as e:
                print(f"Erro ao calcular momento CA para origem {o_idx}, destino {d_idx}: {str(e)}")
                continue
    
    momento_total = momento_cft + momento_ca
    
    # Momento de bota-fora em empréstimos, se aplicável
    momento_bf_emprestimo = 0
    if usar_emprestimos_bota_fora and bota_fora_emprestimo is not None:
        bf_distances_df = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
        for o_idx in cortes_idx:
            for emp_idx in emprestimos_idx:
                if (o_idx, emp_idx) in bf_distances:
                    bf_distances_df.loc[o_idx, emp_idx] = bf_distances[(o_idx, emp_idx)]
        
        # Cálculo seguro do momento de bota-fora em empréstimos
        for o_idx in cortes_idx:
            for emp_idx in emprestimos_idx:
                try:
                    # Acessar o valor de forma segura e converter para escalar
                    value = bota_fora_emprestimo.loc[o_idx, emp_idx]
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        value = value.iloc[0] if not value.empty else 0
                    value = float(value)
                    
                    # Somente calcular para valores positivos
                    if value > 0:
                        # Acessar a distância de forma segura
                        dist = bf_distances_df.loc[o_idx, emp_idx]
                        if isinstance(dist, (pd.Series, pd.DataFrame)):
                            dist = dist.iloc[0] if not dist.empty else 0
                        dist = float(dist)
                        
                        # Adicionar ao momento total
                        momento_bf_emprestimo += value * dist
                except Exception as e:
                    print(f"Erro ao calcular momento BF-Empréstimo para origem {o_idx}, empréstimo {emp_idx}: {str(e)}")
                    continue
    
    # Distância média de transporte (m)
    volume_total_distribuido = cft_distribution.sum().sum() + ca_distribution.sum().sum()
    if volume_total_distribuido > 0:
        dmt = momento_total / volume_total_distribuido
    else:
        dmt = 0
    
    # Volumes totais não utilizados por tipo de empréstimo
    total_emprestimo_lateral_nao_utilizado = sum(emprestimos_laterais_nao_utilizados.values())
    total_emprestimo_concentrado_nao_utilizado = sum(emprestimos_concentrados_nao_utilizados.values())
    total_emprestimo_nao_utilizado = total_emprestimo_lateral_nao_utilizado + total_emprestimo_concentrado_nao_utilizado
    
    # Volume total em bota-fora em empréstimos, se aplicável
    total_bf_emprestimo = 0
    if usar_emprestimos_bota_fora and bota_fora_emprestimo is not None:
        total_bf_emprestimo = bota_fora_emprestimo.sum().sum()
    
    # Análise por tipo de origem
    volume_por_tipo = {
        'cortes': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                           for idx in cortes_idx),
            'bota_fora': sum(bota_fora[idx] for idx in cortes_idx),
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in cortes_idx)
        },
        'emprestimos_laterais': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                            for idx in emprestimos_laterais_idx),
            'nao_utilizado': total_emprestimo_lateral_nao_utilizado,
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in emprestimos_laterais_idx)
        },
        'emprestimos_concentrados': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                             for idx in emprestimos_concentrados_idx),
            'nao_utilizado': total_emprestimo_concentrado_nao_utilizado,
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in emprestimos_concentrados_idx)
        }
    }
    
    # Adiciona dados de bota-fora em empréstimos, se aplicável
    if usar_emprestimos_bota_fora:
        volume_bf_emprestimo_lateral = sum(bota_fora_emprestimo[emp_idx].sum() 
                                         for emp_idx in emprestimos_laterais_idx
                                         if emp_idx in bota_fora_emprestimo.columns)
        
        volume_bf_emprestimo_concentrado = sum(bota_fora_emprestimo[emp_idx].sum() 
                                             for emp_idx in emprestimos_concentrados_idx
                                             if emp_idx in bota_fora_emprestimo.columns)
        
        volume_por_tipo['emprestimos_laterais']['bota_fora_recebido'] = volume_bf_emprestimo_lateral
        volume_por_tipo['emprestimos_concentrados']['bota_fora_recebido'] = volume_bf_emprestimo_concentrado
    
    result = {
        'cft': cft_distribution,
        'ca': ca_distribution,
        'bota_fora': bota_fora,
        'distances': distances_df,
        'remaining_cft': remaining_cft,
        'remaining_ca': remaining_ca,
        'momento_total': momento_total,
        'dmt': dmt,
        'status': status_text,
        'emprestimos_laterais_nao_utilizados': emprestimos_laterais_nao_utilizados,
        'emprestimos_concentrados_nao_utilizados': emprestimos_concentrados_nao_utilizados,
        'total_emprestimo_lateral_nao_utilizado': total_emprestimo_lateral_nao_utilizado,
        'total_emprestimo_concentrado_nao_utilizado': total_emprestimo_concentrado_nao_utilizado,
        'total_emprestimo_nao_utilizado': total_emprestimo_nao_utilizado,
        'volume_por_tipo': volume_por_tipo,
        'favor_cortes': favor_cortes,
        'max_dist_cortes': max_dist_cortes,
        'max_dist_emprestimos_laterais': max_dist_emprestimos_laterais,
        'max_dist_emprestimos_concentrados': max_dist_emprestimos_concentrados,
        'fixed_allocations': fixed_allocations if fixed_allocations else []
    }
    
    # Adiciona informações sobre bota-fora em empréstimos, se aplicável
    if usar_emprestimos_bota_fora:
        result['bota_fora_emprestimo'] = bota_fora_emprestimo
        result['total_bf_emprestimo'] = total_bf_emprestimo
        result['momento_bf_emprestimo'] = momento_bf_emprestimo
        result['usar_emprestimos_bota_fora'] = True
        result['priorizar_bf_proximo'] = priorizar_bf_proximo
        result['max_dist_bota_fora'] = max_dist_bota_fora
    
    return result

def optimize_with_two_steps(origins_df, destinations_df, time_limit=1800, favor_cortes=False, 
                           max_dist_cortes=None, max_dist_emprestimos=None, 
                           max_dist_emprestimos_laterais=None, max_dist_emprestimos_concentrados=None,
                           usar_emprestimos_bota_fora=False, priorizar_bf_proximo=True, 
                           max_dist_bota_fora=None, fixed_allocations=None):
    """
    Executa otimização em duas etapas:
    1. Otimiza a distribuição principal
    2. Otimiza o destino do material excedente para bota-fora em empréstimos
    """
    # Código para diagnóstico
    print("Empréstimos identificados:")
    emprestimos_identificados = origins_df[origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)].index.tolist()
    print(f"Total de empréstimos: {len(emprestimos_identificados)}")
    print(f"IDs dos empréstimos: {emprestimos_identificados}")

    # Verificar os tipos existentes na coluna 'Tipo'
    tipos_unicos = origins_df['Tipo'].unique()
    print(f"Tipos únicos na coluna 'Tipo': {tipos_unicos}")


    # ETAPA 1: Otimização principal da distribuição 
    print("Etapa 1: Otimizando distribuição principal...")
    
    # Identifica tipos de empréstimos
    cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx = identify_emprestimo_types(origins_df)
    emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
    
    # Se temos diferentes tipos de empréstimos
    if emprestimos_laterais_idx and emprestimos_concentrados_idx:
        result_step1 = optimize_distribution_advanced(
            origins_df, destinations_df, 
            time_limit=int(time_limit*0.7),  # 70% do tempo para primeira etapa
            favor_cortes=favor_cortes,
            max_dist_cortes=max_dist_cortes,
            max_dist_emprestimos_laterais=max_dist_emprestimos_laterais,
            max_dist_emprestimos_concentrados=max_dist_emprestimos_concentrados,
            fixed_allocations=fixed_allocations,
            cortes_idx=cortes_idx,
            emprestimos_laterais_idx=emprestimos_laterais_idx,
            emprestimos_concentrados_idx=emprestimos_concentrados_idx,
            usar_emprestimos_bota_fora=False  # Desativado na primeira etapa
        )
    else:
        # Versão padrão
        result_step1 = optimize_distribution(
            origins_df, destinations_df, 
            time_limit=int(time_limit*0.7),  # 70% do tempo para primeira etapa
            favor_cortes=favor_cortes,
            max_dist_cortes=max_dist_cortes,
            max_dist_emprestimos=max_dist_emprestimos,
            fixed_allocations=fixed_allocations,
            usar_emprestimos_bota_fora=False  # Desativado na primeira etapa
        )
    
    # Se a primeira etapa falhou, retorna None
    if result_step1 is None:
        print("A primeira etapa de otimização falhou. Retornando None.")
        return None
    
    # Se não usar empréstimos como bota-fora, retorna o resultado da primeira etapa
    if not usar_emprestimos_bota_fora:
        print("Opção de usar empréstimos como bota-fora está desativada. Pulando segunda etapa.")
        result_step1['usar_emprestimos_bota_fora'] = False
        return result_step1
    
    print("Etapa 2: Otimizando distribuição de bota-fora em empréstimos...")
    print(f"usar_emprestimos_bota_fora = {usar_emprestimos_bota_fora}")
    
    # Extrai informações da primeira etapa
    bota_fora_volumes = result_step1['bota_fora'].copy()
    
    # Verifica se há material para bota-fora
    if bota_fora_volumes.sum() <= 0:
        print("Não há material para bota-fora. Pulando segunda etapa.")
        result_step1['usar_emprestimos_bota_fora'] = True  # Mantém a opção ativa mesmo sem material
        result_step1['bota_fora_emprestimo'] = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
        result_step1['total_bf_emprestimo'] = 0
        result_step1['momento_bf_emprestimo'] = 0
        return result_step1
    
    # Calcula quanto foi retirado de cada empréstimo - ESTRITAMENTE CONFORME A REGRA
    emprestimos_utilizacao = {}
    for emp_idx in emprestimos_idx:
        vol_retirado = result_step1['cft'].loc[emp_idx].sum() + result_step1['ca'].loc[emp_idx].sum()
        emprestimos_utilizacao[emp_idx] = vol_retirado
    
    # Verifica se há empréstimos utilizados
    if sum(emprestimos_utilizacao.values()) <= 0:
        print("Não há empréstimos utilizados. Não é possível enviar bota-fora para empréstimos.")
        result_step1['usar_emprestimos_bota_fora'] = True  # Mantém a opção ativa mesmo sem empréstimos utilizados
        result_step1['bota_fora_emprestimo'] = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
        result_step1['total_bf_emprestimo'] = 0
        result_step1['momento_bf_emprestimo'] = 0
        return result_step1
    
    # Imprime informações detalhadas para debug
    print("\nEmpréstimos utilizados:")
    for emp_idx, vol in emprestimos_utilizacao.items():
        if vol > 0:
            print(f"Empréstimo {emp_idx}: {vol:.2f} m³")
    
    print("\nVolumes de bota-fora a distribuir:")
    for o_idx, vol in bota_fora_volumes.items():
        if vol > 0:
            print(f"Origem {o_idx}: {vol:.2f} m³")
    
    # Cria um problema de otimização só para o bota-fora
    bf_problem = pl.LpProblem("Bota_Fora_Optimization", pl.LpMinimize)
    
    # Calcula distâncias entre cortes com bota-fora e empréstimos
    bf_distances = {}
    for o_idx, bf_volume in bota_fora_volumes.items():
        if bf_volume > 0:
            o_row = origins_df.loc[o_idx]
            for emp_idx in emprestimos_idx:
                if emprestimos_utilizacao.get(emp_idx, 0) > 0:  # Apenas empréstimos utilizados
                    emp_row = origins_df.loc[emp_idx]
                    # Calcula distância básica
                    dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                    # Adiciona DT fixo do empréstimo, se existir
                    if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                        dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                    bf_distances[(o_idx, emp_idx)] = dist
    
    # Aplica distância máxima, se definida
    bf_adjusted_distances = bf_distances.copy()
    if max_dist_bota_fora is not None:
        for key, dist in bf_distances.items():
            if dist > max_dist_bota_fora:
                bf_adjusted_distances[key] = dist * 2.0  # Penalização
                print(f"Distância ajustada para bota-fora {key[0]}->{key[1]}: {dist} -> {bf_adjusted_distances[key]} (excede máximo de {max_dist_bota_fora}m)")
    
    # Normaliza distâncias apenas se houver alguma distância
    if bf_adjusted_distances:
        max_dist = max(bf_adjusted_distances.values())
        for key in bf_adjusted_distances:
            bf_adjusted_distances[key] /= max_dist
    
    # Variáveis de decisão: quanto de cada bota-fora vai para cada empréstimo
    bf_to_emp_vars = {}
    for o_idx, bf_volume in bota_fora_volumes.items():
        if bf_volume > 0:
            for emp_idx in emprestimos_idx:
                if emprestimos_utilizacao.get(emp_idx, 0) > 0:  # Apenas empréstimos utilizados
                    bf_to_emp_vars[(o_idx, emp_idx)] = pl.LpVariable(
                        f"BF_TO_EMP_{o_idx}_{emp_idx}",
                        lowBound=0
                    )
    
    # Variáveis para bota-fora convencional remanescente
    bf_remaining_vars = {}
    for o_idx, bf_volume in bota_fora_volumes.items():
        if bf_volume > 0:
            bf_remaining_vars[o_idx] = pl.LpVariable(
                f"BF_REMAIN_{o_idx}",
                lowBound=0
            )
    # Adicione após a criação das variáveis
    print(f"Número de variáveis de decisão criadas para bota-fora em empréstimos: {len(bf_to_emp_vars)}")
    for (o_idx, emp_idx) in list(bf_to_emp_vars.keys())[:5]:  # Mostrar as 5 primeiras
     print(f"Variável: origem {o_idx} -> empréstimo {emp_idx}")

    # Se não houver variáveis de bota-fora para empréstimos, retorna o resultado da primeira etapa
    if not bf_to_emp_vars:
        print("Não foi possível criar variáveis de decisão para bota-fora em empréstimos.")
        print("Isto pode ocorrer se não houver empréstimos utilizados ou se as distâncias excedem os limites.")
        result_step1['usar_emprestimos_bota_fora'] = True  # Mantém a opção ativa
        result_step1['bota_fora_emprestimo'] = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
        result_step1['total_bf_emprestimo'] = 0
        result_step1['momento_bf_emprestimo'] = 0
        return result_step1
    
    # Função objetivo: minimizar distância, priorizando bota-fora mais próximo
    if priorizar_bf_proximo:
        bf_problem += pl.lpSum([
            bf_adjusted_distances[(o_idx, emp_idx)] * bf_to_emp_vars[(o_idx, emp_idx)]
            for (o_idx, emp_idx) in bf_to_emp_vars
        ])
    else:
        # Minimiza o volume enviado para bota-fora convencional
        bf_problem += pl.lpSum(bf_remaining_vars.values())
    
    # Restrição: conservação de volume de bota-fora
    for o_idx, bf_volume in bota_fora_volumes.items():
        if bf_volume > 0:
            bf_problem += (
                pl.lpSum([bf_to_emp_vars.get((o_idx, emp_idx), 0) for emp_idx in emprestimos_idx 
                       if (o_idx, emp_idx) in bf_to_emp_vars]) + 
                bf_remaining_vars[o_idx] == bf_volume,
                f"Conservacao_BF_{o_idx}"
            )
    
    # Restrição: capacidade de cada empréstimo para receber bota-fora
    for emp_idx in emprestimos_idx:
        vol_utilizado = emprestimos_utilizacao.get(emp_idx, 0)
        if vol_utilizado > 0:
            bf_problem += (
                pl.lpSum([bf_to_emp_vars.get((o_idx, emp_idx), 0) for o_idx in bota_fora_volumes.index
                       if (o_idx, emp_idx) in bf_to_emp_vars]) <= vol_utilizado,
                f"Capacidade_Emprestimo_{emp_idx}"
            )
    
    # Resolver o problema
    bf_solver = pl.PULP_CBC_CMD(
        msg=True, 
        timeLimit=int(time_limit*0.3),  # 30% do tempo para segunda etapa
        gapRel=0.01,
        options=['presolve on']
    )
    bf_status = bf_problem.solve(bf_solver)
    
    # Verifica o status da solução
    bf_status_text = pl.LpStatus[bf_status]
    print(f"\nStatus da otimização de bota-fora: {bf_status_text}")
    print(f"Valor da função objetivo: {pl.value(bf_problem.objective)}")
    
    # Cria um DataFrame para o resultado de bota-fora em empréstimos
    bota_fora_emprestimo = pd.DataFrame(0, index=bota_fora_volumes.index, columns=emprestimos_idx)
    
    # Extrai resultados
    for (o_idx, emp_idx), var in bf_to_emp_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            bota_fora_emprestimo.loc[o_idx, emp_idx] = var.value()
            print(f"Alocando {var.value():.2f} m³ de bota-fora da origem {o_idx} para o empréstimo {emp_idx}")
    
    # Atualiza o bota-fora remanescente
    bota_fora_atualizado = pd.Series(0, index=bota_fora_volumes.index)
    for o_idx, var in bf_remaining_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            bota_fora_atualizado[o_idx] = var.value()
    
    # Resumo da distribuição de bota-fora
    total_bf_original = bota_fora_volumes.sum()
    total_bf_emprestimo = bota_fora_emprestimo.sum().sum()
    total_bf_remanescente = bota_fora_atualizado.sum()
    
    print(f"\nResumo de bota-fora:")
    print(f"Total original: {total_bf_original:.2f} m³")
    print(f"Alocado para empréstimos: {total_bf_emprestimo:.2f} m³ ({100*total_bf_emprestimo/total_bf_original:.1f}%)")
    print(f"Remanescente convencional: {total_bf_remanescente:.2f} m³ ({100*total_bf_remanescente/total_bf_original:.1f}%)")
    
    # Calcula momento de bota-fora em empréstimos
    momento_bf_emprestimo = sum(
        bota_fora_emprestimo.loc[o_idx, emp_idx] * bf_distances.get((o_idx, emp_idx), 0)
        for o_idx in bota_fora_emprestimo.index
        for emp_idx in bota_fora_emprestimo.columns
        if bota_fora_emprestimo.loc[o_idx, emp_idx] > 0
    )
    
    # Atualiza o volume por tipo para incluir bota-fora em empréstimos
    if 'volume_por_tipo' in result_step1:
        volume_bf_emprestimo_lateral = sum(
            bota_fora_emprestimo[emp_idx].sum() 
            for emp_idx in emprestimos_laterais_idx
            if emp_idx in bota_fora_emprestimo.columns
        )
        
        volume_bf_emprestimo_concentrado = sum(
            bota_fora_emprestimo[emp_idx].sum() 
            for emp_idx in emprestimos_concentrados_idx
            if emp_idx in bota_fora_emprestimo.columns
        )
        
        if 'emprestimos_laterais' in result_step1['volume_por_tipo']:
            result_step1['volume_por_tipo']['emprestimos_laterais']['bota_fora_recebido'] = volume_bf_emprestimo_lateral
        
        if 'emprestimos_concentrados' in result_step1['volume_por_tipo']:
            result_step1['volume_por_tipo']['emprestimos_concentrados']['bota_fora_recebido'] = volume_bf_emprestimo_concentrado
    
    # Combina os resultados
    final_result = result_step1.copy()
    final_result['bota_fora'] = bota_fora_atualizado
    final_result['bota_fora_emprestimo'] = bota_fora_emprestimo
    final_result['total_bf_emprestimo'] = total_bf_emprestimo
    final_result['usar_emprestimos_bota_fora'] = True
    final_result['priorizar_bf_proximo'] = priorizar_bf_proximo
    final_result['max_dist_bota_fora'] = max_dist_bota_fora
    final_result['momento_bf_emprestimo'] = momento_bf_emprestimo
    final_result['bf_distances'] = bf_distances
    
    return final_result

def generate_distribution_summary(result, origins_df, destinations_df):
    """
    Gera um resumo da distribuição otimizada
    """
    if result is None:
        return "Não foi possível encontrar uma solução factível."
    
    # Cria um resumo das distribuições
    cft_dist = result['cft']
    ca_dist = result['ca']
    bota_fora = result['bota_fora']
    dmt = result['dmt']
    
    # Total por tipo de material
    total_cft = cft_dist.sum().sum()
    total_ca = ca_dist.sum().sum()
    total_bota_fora = bota_fora.sum()
    
    # Verifica se há bota-fora em empréstimos
    total_bf_emprestimo = 0
    if 'bota_fora_emprestimo' in result and result['bota_fora_emprestimo'] is not None:
        total_bf_emprestimo = result['bota_fora_emprestimo'].sum().sum()
    
    summary = [
        f"Status da otimização: {result['status']}",
        f"Momento total de transporte: {result['momento_total']:.2f} m³·m",
        f"Distância média de transporte (DMT): {dmt:.2f} m",
        f"Volume CFT distribuído: {total_cft:.2f} m³",
        f"Volume CA distribuído: {total_ca:.2f} m³",
        f"Volume enviado para bota-fora convencional: {total_bota_fora:.2f} m³"
    ]
    
    # Adiciona informações sobre bota-fora em empréstimos, se aplicável
    if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
        summary.append(f"Volume enviado para bota-fora em empréstimos: {total_bf_emprestimo:.2f} m³")
        summary.append(f"Volume total de bota-fora: {total_bota_fora + total_bf_emprestimo:.2f} m³")
    
    # Verifica se há volume não atendido
    remaining_cft = result['remaining_cft']
    remaining_ca = result['remaining_ca']
    
    total_remaining_cft = remaining_cft.sum()
    total_remaining_ca = remaining_ca.sum()
    
    if total_remaining_cft > 0:
        summary.append(f"Volume CFT não atendido: {total_remaining_cft:.2f} m³")
        # Detalha destinos não atendidos para CFT
        for d_idx in remaining_cft.index:
            try:
                # Acesso seguro ao valor e conversão para escalar
                cft_value = remaining_cft.loc[d_idx]
                if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                    cft_value = cft_value.iloc[0] if not cft_value.empty else 0
                cft_value = float(cft_value)
                
                # Somente listar valores positivos
                if cft_value > 0:
                    summary.append(f"  - Destino {d_idx}: {cft_value:.2f} m³")
            except Exception as e:
                print(f"Erro ao processar remaining_cft para destino {d_idx}: {str(e)}")
                continue
    
    if total_remaining_ca > 0:
        summary.append(f"Volume CA não atendido: {total_remaining_ca:.2f} m³")
        # Detalha destinos não atendidos para CA
        for d_idx in remaining_ca.index:
            try:
                # Acesso seguro ao valor e conversão para escalar
                ca_value = remaining_ca.loc[d_idx]
                if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                    ca_value = ca_value.iloc[0] if not ca_value.empty else 0
                ca_value = float(ca_value)
                
                # Somente listar valores positivos
                if ca_value > 0:
                    summary.append(f"  - Destino {d_idx}: {ca_value:.2f} m³")
            except Exception as e:
                print(f"Erro ao processar remaining_ca para destino {d_idx}: {str(e)}")
                continue
    
    # Informações sobre empréstimos não utilizados
    if 'volume_por_tipo' in result:
        volume_por_tipo = result['volume_por_tipo']
        
        # Cortes
        vpt_cortes = volume_por_tipo['cortes']
        perc_cortes_utilizado = (vpt_cortes['utilizado'] / vpt_cortes['total'] * 100) if vpt_cortes['total'] > 0 else 0
        summary.append(f"Cortes: {vpt_cortes['utilizado']:.2f} m³ utilizados de {vpt_cortes['total']:.2f} m³ ({perc_cortes_utilizado:.1f}%)")
        summary.append(f"Bota-fora de cortes: {vpt_cortes['bota_fora']:.2f} m³ ({vpt_cortes['bota_fora']/vpt_cortes['total']*100:.1f}% do volume de corte)")
        
        # Empréstimos laterais, se houver
        if 'emprestimos_laterais' in volume_por_tipo:
            vpt_emp_lat = volume_por_tipo['emprestimos_laterais']
            if vpt_emp_lat['total'] > 0:
                perc_lat_utilizado = (vpt_emp_lat['utilizado'] / vpt_emp_lat['total'] * 100) if vpt_emp_lat['total'] > 0 else 0
                summary.append(f"Empréstimos Laterais: {vpt_emp_lat['utilizado']:.2f} m³ utilizados de {vpt_emp_lat['total']:.2f} m³ ({perc_lat_utilizado:.1f}%)")
                
                # Adiciona informação sobre bota-fora recebido, se aplicável
                if 'bota_fora_recebido' in vpt_emp_lat:
                    summary.append(f"Bota-fora em Empréstimos Laterais: {vpt_emp_lat['bota_fora_recebido']:.2f} m³")
        
        # Empréstimos concentrados, se houver
        if 'emprestimos_concentrados' in volume_por_tipo:
            vpt_emp_conc = volume_por_tipo['emprestimos_concentrados']
            if vpt_emp_conc['total'] > 0:
                perc_conc_utilizado = (vpt_emp_conc['utilizado'] / vpt_emp_conc['total'] * 100) if vpt_emp_conc['total'] > 0 else 0
                summary.append(f"Empréstimos Concentrados: {vpt_emp_conc['utilizado']:.2f} m³ utilizados de {vpt_emp_conc['total']:.2f} m³ ({perc_conc_utilizado:.1f}%)")
                
                # Adiciona informação sobre bota-fora recebido, se aplicável
                if 'bota_fora_recebido' in vpt_emp_conc:
                    summary.append(f"Bota-fora em Empréstimos Concentrados: {vpt_emp_conc['bota_fora_recebido']:.2f} m³")
    else:
        # Versão mais simples para resultados sem classificação detalhada
        total_emprestimo_nao_utilizado = result.get('total_emprestimo_nao_utilizado', 0)
        if total_emprestimo_nao_utilizado > 0:
            summary.append(f"Volume de empréstimo não utilizado: {total_emprestimo_nao_utilizado:.2f} m³")
    
    return "\n".join(summary)

def create_distribution_report(result, origins_df, destinations_df, filename=None):
    """
    Cria um relatório de distribuição em formato Excel
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
        filename: Nome do arquivo para salvar o relatório
    
    Returns:
        BytesIO object contendo o arquivo Excel
    """
    if result is None:
        return None
    
    # Cria um arquivo Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formato para células
        header_format = workbook.add_format({
            'bold': True, 
            'text_wrap': True, 
            'valign': 'center', 
            'align': 'center',
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })
        
        # Status da otimização
        info_sheet = workbook.add_worksheet('Informações Gerais')
        
        # Título
        info_sheet.write(0, 0, 'Resumo da Distribuição de Terraplenagem', 
                        workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Data e hora
        now = datetime.datetime.now()
        info_sheet.write(1, 0, f'Relatório gerado em: {now.strftime("%d/%m/%Y %H:%M:%S")}')
        
        # Resumo geral
        row = 3
        info_sheet.write(row, 0, 'Status da Otimização', header_format)
        info_sheet.write(row, 1, result['status'], cell_format)
        row += 1
        
        info_sheet.write(row, 0, 'Momento Total de Transporte (m³·m)', header_format)
        info_sheet.write(row, 1, result['momento_total'], number_format)
        row += 1
        
        info_sheet.write(row, 0, 'Distância Média de Transporte (m)', header_format)
        info_sheet.write(row, 1, result['dmt'], number_format)
        row += 1
        
        # Informação sobre bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            info_sheet.write(row, 0, 'Uso de Empréstimos como Bota-Fora', header_format)
            info_sheet.write(row, 1, 'Sim', cell_format)
            row += 1
            
            if 'priorizar_bf_proximo' in result:
                info_sheet.write(row, 0, 'Priorizar Bota-Fora Mais Próximo', header_format)
                info_sheet.write(row, 1, 'Sim' if result['priorizar_bf_proximo'] else 'Não', cell_format)
                row += 1
            
            if 'max_dist_bota_fora' in result and result['max_dist_bota_fora'] is not None:
                info_sheet.write(row, 0, 'Distância Máxima para Bota-Fora em Empréstimos (m)', header_format)
                info_sheet.write(row, 1, result['max_dist_bota_fora'], number_format)
                row += 1
            
            if 'total_bf_emprestimo' in result:
                info_sheet.write(row, 0, 'Volume Total em Bota-Fora em Empréstimos (m³)', header_format)
                info_sheet.write(row, 1, result['total_bf_emprestimo'], number_format)
                row += 1
            
            if 'momento_bf_emprestimo' in result:
                info_sheet.write(row, 0, 'Momento de Transporte para Bota-Fora em Empréstimos (m³·m)', header_format)
                info_sheet.write(row, 1, result['momento_bf_emprestimo'], number_format)
                row += 1
        
        # Resumo de volumes
        row += 1
        info_sheet.write(row, 0, 'Resumo de Volumes', 
                       workbook.add_format({'bold': True, 'font_size': 12}))
        row += 1
        
        # Cabeçalho de volumes
        info_sheet.write(row, 0, 'Tipo', header_format)
        info_sheet.write(row, 1, 'Volume Disponível (m³)', header_format)
        info_sheet.write(row, 2, 'Volume Distribuído (m³)', header_format)
        info_sheet.write(row, 3, 'Volume Não Utilizado (m³)', header_format)
        info_sheet.write(row, 4, 'Utilização (%)', header_format)
        
        # Se estiver habilitado bota-fora em empréstimos, adiciona coluna
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            info_sheet.write(row, 5, 'Bota-Fora Recebido (m³)', header_format)
        
        row += 1
        
        # Volumes de origem
        total_origem = origins_df['Volume disponível (m³)'].sum()
        total_distribuido = result['cft'].sum().sum() + result['ca'].sum().sum()
        
        # Cortes x Empréstimos
        if 'volume_por_tipo' in result:
            vpt = result['volume_por_tipo']
            
            # Cortes
            corte_info = vpt['cortes']
            info_sheet.write(row, 0, 'Cortes', cell_format)
            info_sheet.write(row, 1, corte_info['total'], number_format)
            info_sheet.write(row, 2, corte_info['utilizado'], number_format)
            info_sheet.write(row, 3, corte_info['bota_fora'], number_format)
            if corte_info['total'] > 0:
                info_sheet.write(row, 4, corte_info['utilizado'] / corte_info['total'], 
                               workbook.add_format({'num_format': '0.0%', 'border': 1}))
            else:
                info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
            
            # Se estiver habilitado bota-fora em empréstimos, adiciona valor (não aplicável para cortes)
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
                info_sheet.write(row, 5, 0, number_format)  # Cortes não recebem bota-fora
            
            row += 1
            
            # Empréstimos Laterais, se houver
            if 'emprestimos_laterais' in vpt and vpt['emprestimos_laterais']['total'] > 0:
                emp_lat_info = vpt['emprestimos_laterais']
                info_sheet.write(row, 0, 'Empréstimos Laterais', cell_format)
                info_sheet.write(row, 1, emp_lat_info['total'], number_format)
                info_sheet.write(row, 2, emp_lat_info['utilizado'], number_format)
                info_sheet.write(row, 3, emp_lat_info['nao_utilizado'], number_format)
                if emp_lat_info['total'] > 0:
                    info_sheet.write(row, 4, emp_lat_info['utilizado'] / emp_lat_info['total'], 
                                   workbook.add_format({'num_format': '0.0%', 'border': 1}))
                else:
                    info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
                
                # Se estiver habilitado bota-fora em empréstimos, adiciona valor
                if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
                    bf_recebido = emp_lat_info.get('bota_fora_recebido', 0)
                    info_sheet.write(row, 5, bf_recebido, number_format)
                
                row += 1
            
            # Empréstimos Concentrados, se houver
            if 'emprestimos_concentrados' in vpt and vpt['emprestimos_concentrados']['total'] > 0:
                emp_conc_info = vpt['emprestimos_concentrados']
                info_sheet.write(row, 0, 'Empréstimos Concentrados', cell_format)
                info_sheet.write(row, 1, emp_conc_info['total'], number_format)
                info_sheet.write(row, 2, emp_conc_info['utilizado'], number_format)
                info_sheet.write(row, 3, emp_conc_info['nao_utilizado'], number_format)
                if emp_conc_info['total'] > 0:
                    info_sheet.write(row, 4, emp_conc_info['utilizado'] / emp_conc_info['total'], 
                                   workbook.add_format({'num_format': '0.0%', 'border': 1}))
                else:
                    info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
                
                # Se estiver habilitado bota-fora em empréstimos, adiciona valor
                if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
                    bf_recebido = emp_conc_info.get('bota_fora_recebido', 0)
                    info_sheet.write(row, 5, bf_recebido, number_format)
                
                row += 1
        else:
            # Versão simplificada se não houver classificação detalhada
            info_sheet.write(row, 0, 'Total Origens', cell_format)
            info_sheet.write(row, 1, total_origem, number_format)
            info_sheet.write(row, 2, total_distribuido, number_format)
            info_sheet.write(row, 3, total_origem - total_distribuido, number_format)
            if total_origem > 0:
                info_sheet.write(row, 4, total_distribuido / total_origem, 
                               workbook.add_format({'num_format': '0.0%', 'border': 1}))
            else:
                info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
            
            # Se estiver habilitado bota-fora em empréstimos, adiciona valor total
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
                info_sheet.write(row, 5, result.get('total_bf_emprestimo', 0), number_format)
            
            row += 1
        
        # Total geral
        info_sheet.write(row, 0, 'Total Geral', header_format)
        info_sheet.write(row, 1, total_origem, number_format)
        info_sheet.write(row, 2, total_distribuido, number_format)
        info_sheet.write(row, 3, total_origem - total_distribuido, number_format)
        if total_origem > 0:
            info_sheet.write(row, 4, total_distribuido / total_origem, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Se estiver habilitado bota-fora em empréstimos, adiciona valor total
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            info_sheet.write(row, 5, result.get('total_bf_emprestimo', 0), number_format)
        
        row += 1
        
        # Volumes de destino
        row += 2
        info_sheet.write(row, 0, 'Resumo de Destinos', 
                       workbook.add_format({'bold': True, 'font_size': 12}))
        row += 1
        
        # Cabeçalho de destinos
        info_sheet.write(row, 0, 'Tipo', header_format)
        info_sheet.write(row, 1, 'Volume Necessário (m³)', header_format)
        info_sheet.write(row, 2, 'Volume Atendido (m³)', header_format)
        info_sheet.write(row, 3, 'Volume Faltante (m³)', header_format)
        info_sheet.write(row, 4, 'Atendimento (%)', header_format)
        row += 1
        
        # CFT
        total_cft_necessario = destinations_df['Volume CFT (m³)'].fillna(0).sum()
        total_cft_atendido = result['cft'].sum().sum()
        info_sheet.write(row, 0, 'CFT', cell_format)
        info_sheet.write(row, 1, total_cft_necessario, number_format)
        info_sheet.write(row, 2, total_cft_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_cft_necessario - total_cft_atendido), number_format)
        if total_cft_necessario > 0:
            info_sheet.write(row, 4, total_cft_atendido / total_cft_necessario, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        row += 1
        
        # CA
        total_ca_necessario = destinations_df['Volume CA (m³)'].fillna(0).sum()
        total_ca_atendido = result['ca'].sum().sum()
        info_sheet.write(row, 0, 'CA', cell_format)
        info_sheet.write(row, 1, total_ca_necessario, number_format)
        info_sheet.write(row, 2, total_ca_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_ca_necessario - total_ca_atendido), number_format)
        if total_ca_necessario > 0:
            info_sheet.write(row, 4, total_ca_atendido / total_ca_necessario, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        row += 1
        
        # Total geral
        total_destino = total_cft_necessario + total_ca_necessario
        total_atendido = total_cft_atendido + total_ca_atendido
        info_sheet.write(row, 0, 'Total Geral', header_format)
        info_sheet.write(row, 1, total_destino, number_format)
        info_sheet.write(row, 2, total_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_destino - total_atendido), number_format)
        if total_destino > 0:
            info_sheet.write(row, 4, total_atendido / total_destino, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Ajusta largura das colunas
        info_sheet.set_column('A:A', 22)
        info_sheet.set_column('B:E', 18)
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            info_sheet.set_column('F:F', 18)
        
        # Adiciona planilhas para distribuições detalhadas
        
        # Distribuição CFT
        cft_sheet = workbook.add_worksheet('Distribuição CFT')
        
        # Título
        cft_sheet.write(0, 0, 'Distribuição de Material CFT', 
                       workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distribuição
        row = 2
        
        # Cabeçalho com destinos
        cft_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            cft_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        cft_sheet.write(row, len(destinations_df.index)+1, "Total (m³)", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            cft_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            row_sum = 0
            for j, d_idx in enumerate(destinations_df.index):
                value = result['cft'].loc[o_idx, d_idx]
                cft_sheet.write(row+i, j+1, value if value > 0 else "", number_format)
                row_sum += value
            cft_sheet.write(row+i, len(destinations_df.index)+1, row_sum, number_format)
        
        # Totais por coluna
        total_row = row + len(origins_df.index)
        cft_sheet.write(total_row, 0, "Total (m³)", header_format)
        for j, d_idx in enumerate(destinations_df.index):
            col_sum = result['cft'][d_idx].sum()
            cft_sheet.write(total_row, j+1, col_sum, number_format)
        
        cft_sheet.write(total_row, len(destinations_df.index)+1, 
                      result['cft'].sum().sum(), number_format)
        
        # Ajusta larguras das colunas
        cft_sheet.set_column('A:A', 15)
        cft_sheet.set_column(1, len(destinations_df.index)+1, 12)
        
        # Distribuição CA
        ca_sheet = workbook.add_worksheet('Distribuição CA')
        
        # Título
        ca_sheet.write(0, 0, 'Distribuição de Material CA', 
                      workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distribuição
        row = 2
        
        # Cabeçalho com destinos
        ca_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            ca_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        ca_sheet.write(row, len(destinations_df.index)+1, "Total (m³)", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            ca_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            row_sum = 0
            for j, d_idx in enumerate(destinations_df.index):
                value = result['ca'].loc[o_idx, d_idx]
                ca_sheet.write(row+i, j+1, value if value > 0 else "", number_format)
                row_sum += value
            ca_sheet.write(row+i, len(destinations_df.index)+1, row_sum, number_format)
        
        # Totais por coluna
        total_row = row + len(origins_df.index)
        ca_sheet.write(total_row, 0, "Total (m³)", header_format)
        for j, d_idx in enumerate(destinations_df.index):
            col_sum = result['ca'][d_idx].sum()
            ca_sheet.write(total_row, j+1, col_sum, number_format)
        
        ca_sheet.write(total_row, len(destinations_df.index)+1, 
                     result['ca'].sum().sum(), number_format)
        
        # Ajusta larguras das colunas
        ca_sheet.set_column('A:A', 15)
        ca_sheet.set_column(1, len(destinations_df.index)+1, 12)
        
        # Adiciona nova aba específica para resumo de empréstimos
        empr_sheet = workbook.add_worksheet('Resumo de Empréstimos')

        # Título
        empr_sheet.write(0, 0, 'Resumo de Empréstimos', 
                      workbook.add_format({'bold': True, 'font_size': 14}))

        # Cabeçalho
        row = 2
        empr_sheet.write(row, 0, 'ID', header_format)
        empr_sheet.write(row, 1, 'Tipo', header_format)
        empr_sheet.write(row, 2, 'Volume Disponível (m³)', header_format)
        empr_sheet.write(row, 3, 'Volume Utilizado (m³)', header_format)
        empr_sheet.write(row, 4, 'Volume Não Utilizado (m³)', header_format)
        empr_sheet.write(row, 5, 'Utilização (%)', header_format)
        empr_sheet.write(row, 6, 'Volume Bota-Fora Recebido (m³)', header_format)
        row += 1

        # Identifica os empréstimos
        emprestimos_idx = []
        if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
         emprestimos_laterais_idx = list(result['emprestimos_laterais_nao_utilizados'].keys())
         emprestimos_concentrados_idx = list(result['emprestimos_concentrados_nao_utilizados'].keys())
         emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
        elif 'emprestimos_nao_utilizados' in result:
          emprestimos_idx = list(result['emprestimos_nao_utilizados'].keys())
        else:
            # Tenta identificar pela coluna 'Tipo'
            emprestimos_idx = origins_df[origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)].index.tolist()

        # Total para somar todos os valores
        total_disponivel = 0
        total_utilizado = 0
        total_nao_utilizado = 0
        total_bf_recebido = 0

        # Lista para armazenar dados de cada empréstimo para ordenar depois
        emprestimos_data = []

        # Dados de cada empréstimo
        for i, emp_idx in enumerate(emprestimos_idx):
            # Volume disponível
            vol_disponivel = float(origins_df.loc[emp_idx, 'Volume disponível (m³)'])
    
            # Volume utilizado (CFT + CA)
            vol_cft = result['cft'].loc[emp_idx].sum()
            vol_ca = result['ca'].loc[emp_idx].sum()
            vol_utilizado = vol_cft + vol_ca
    
            # Volume não utilizado
            vol_nao_utilizado = 0
            if 'emprestimos_laterais_nao_utilizados' in result and emp_idx in result['emprestimos_laterais_nao_utilizados']:
                vol_nao_utilizado = float(result['emprestimos_laterais_nao_utilizados'][emp_idx])
                tipo = "Empréstimo Lateral"
            elif 'emprestimos_concentrados_nao_utilizados' in result and emp_idx in result['emprestimos_concentrados_nao_utilizados']:
                vol_nao_utilizado = float(result['emprestimos_concentrados_nao_utilizados'][emp_idx])
                tipo = "Empréstimo Concentrado"
            elif 'emprestimos_nao_utilizados' in result and emp_idx in result['emprestimos_nao_utilizados']:
                vol_nao_utilizado = float(result['emprestimos_nao_utilizados'][emp_idx])
                tipo = "Empréstimo"
            else:
                tipo = "Empréstimo"
    
            # Tenta pegar o tipo diretamente do DataFrame de origens se disponível
            if 'Tipo' in origins_df.columns:
                tipo = origins_df.loc[emp_idx, 'Tipo']
    
            # Utilização (%)
            utilizacao = vol_utilizado / vol_disponivel * 100 if vol_disponivel > 0 else 0
    
            # Volume de bota-fora recebido, se aplicável
            vol_bf_recebido = 0
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                if emp_idx in result['bota_fora_emprestimo'].columns:
                    vol_bf_recebido = result['bota_fora_emprestimo'][emp_idx].sum()
    
            # Adiciona à lista para ordenação posterior
            emprestimos_data.append({
                'idx': emp_idx,
                'tipo': tipo,
                'vol_disponivel': vol_disponivel,
                'vol_utilizado': vol_utilizado,
                'vol_nao_utilizado': vol_nao_utilizado,
                'utilizacao': utilizacao,
                'vol_bf_recebido': vol_bf_recebido
            })
    
            # Soma aos totais
            total_disponivel += vol_disponivel
            total_utilizado += vol_utilizado
            total_nao_utilizado += vol_nao_utilizado
            total_bf_recebido += vol_bf_recebido

        # Ordena empréstimos por ID
        emprestimos_data.sort(key=lambda x: x['idx'])

        # Escreve os dados ordenados
        for i, emp_data in enumerate(emprestimos_data):
            empr_sheet.write(row + i, 0, emp_data['idx'], cell_format)
            empr_sheet.write(row + i, 1, emp_data['tipo'], cell_format)
            empr_sheet.write(row + i, 2, emp_data['vol_disponivel'], number_format)
            empr_sheet.write(row + i, 3, emp_data['vol_utilizado'], number_format)
            empr_sheet.write(row + i, 4, emp_data['vol_nao_utilizado'], number_format)
            empr_sheet.write(row + i, 5, emp_data['utilizacao'] / 100, 
                           workbook.add_format({'num_format': '0.00%', 'border': 1}))
    
            # Volume de bota-fora recebido
            if emp_data['vol_bf_recebido'] > 0:
                empr_sheet.write(row + i, 6, emp_data['vol_bf_recebido'], number_format)
            else:
                empr_sheet.write(row + i, 6, "", cell_format)

        # Linha de total
        total_row = row + len(emprestimos_data)
        empr_sheet.write(total_row, 0, "TOTAL", header_format)
        empr_sheet.write(total_row, 1, "", header_format)
        empr_sheet.write(total_row, 2, total_disponivel, number_format)
        empr_sheet.write(total_row, 3, total_utilizado, number_format)
        empr_sheet.write(total_row, 4, total_nao_utilizado, number_format)

        # Utilização total (%)
        total_utilizacao = total_utilizado / total_disponivel * 100 if total_disponivel > 0 else 0
        empr_sheet.write(total_row, 5, total_utilizacao / 100, 
                       workbook.add_format({'num_format': '0.00%', 'border': 1, 'bold': True}))

                # Total de bota-fora recebido
        empr_sheet.write(total_row, 6, total_bf_recebido, number_format)

        # Ajusta largura das colunas
        empr_sheet.set_column('A:A', 10)  # ID
        empr_sheet.set_column('B:B', 30)  # Tipo
        empr_sheet.set_column('C:G', 18)  # Valores numéricos


        # Adiciona planilha para bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            bf_emp_sheet = workbook.add_worksheet('Bota-Fora em Empréstimos')
            
            # Título
            bf_emp_sheet.write(0, 0, 'Distribuição de Bota-Fora em Empréstimos', 
                             workbook.add_format({'bold': True, 'font_size': 14}))
            
            # Escreve a matriz de distribuição
            row = 2
            
            # Obtém os índices de cortes e empréstimos
            cortes_idx = [idx for idx in origins_df.index if idx in result['bota_fora_emprestimo'].index]
            emprestimos_idx = [idx for idx in origins_df.index if idx in result['bota_fora_emprestimo'].columns]
            
            # Cabeçalho com empréstimos
            bf_emp_sheet.write(row, 0, 'Corte / Empréstimo', header_format)
            for col, emp_idx in enumerate(emprestimos_idx):
                bf_emp_sheet.write(row, col+1, f"Empréstimo {emp_idx}", header_format)
            bf_emp_sheet.write(row, len(emprestimos_idx)+1, "Total (m³)", header_format)
            row += 1
            
            # Linhas com cortes
            for i, o_idx in enumerate(cortes_idx):
                bf_emp_sheet.write(row+i, 0, f"Corte {o_idx}", cell_format)
                row_sum = 0
                for j, emp_idx in enumerate(emprestimos_idx):
                    value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx] if o_idx in result['bota_fora_emprestimo'].index and emp_idx in result['bota_fora_emprestimo'].columns else 0
                    bf_emp_sheet.write(row+i, j+1, value if value > 0 else "", number_format)
                    row_sum += value
                bf_emp_sheet.write(row+i, len(emprestimos_idx)+1, row_sum, number_format)
            
            # Totais por coluna
            total_row = row + len(cortes_idx)
            bf_emp_sheet.write(total_row, 0, "Total (m³)", header_format)
            for j, emp_idx in enumerate(emprestimos_idx):
                col_sum = result['bota_fora_emprestimo'][emp_idx].sum() if emp_idx in result['bota_fora_emprestimo'].columns else 0
                bf_emp_sheet.write(total_row, j+1, col_sum, number_format)
            
            bf_emp_sheet.write(total_row, len(emprestimos_idx)+1, 
                             result['bota_fora_emprestimo'].sum().sum(), number_format)
            
            # Ajusta larguras das colunas
            bf_emp_sheet.set_column('A:A', 15)
            bf_emp_sheet.set_column(1, len(emprestimos_idx)+1, 12)
        
        # Distâncias
        dist_sheet = workbook.add_worksheet('Distâncias')
        
        # Título
        dist_sheet.write(0, 0, 'Matriz de Distâncias (m)', 
                        workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distâncias
        row = 2
        
        # Cabeçalho com destinos
        dist_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            dist_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            dist_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            for j, d_idx in enumerate(destinations_df.index):
                dist_sheet.write(row+i, j+1, result['distances'].loc[o_idx, d_idx], number_format)
        
        # Ajusta larguras das colunas
        dist_sheet.set_column('A:A', 15)
        dist_sheet.set_column(1, len(destinations_df.index), 12)
        
        # Adiciona matriz de distâncias para bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            row += len(origins_df.index) + 2
            
            # Título da seção
            dist_sheet.write(row, 0, 'Matriz de Distâncias para Bota-Fora em Empréstimos (m)', 
                           workbook.add_format({'bold': True, 'font_size': 12}))
            row += 1
            
            # Obtém os índices de cortes e empréstimos
            cortes_idx = [idx for idx in origins_df.index if idx in result['bota_fora_emprestimo'].index]
            emprestimos_idx = [idx for idx in origins_df.index if idx in result['bota_fora_emprestimo'].columns]
            
            # Cabeçalho com empréstimos
            dist_sheet.write(row, 0, 'Corte / Empréstimo', header_format)
            for col, emp_idx in enumerate(emprestimos_idx):
                dist_sheet.write(row, col+1, f"Empréstimo {emp_idx}", header_format)
            row += 1
            
            # Cria uma matriz básica de distâncias entre cortes e empréstimos
            bf_distances = {}
            for o_idx in cortes_idx:
                o_row = origins_df.loc[o_idx]
                for emp_idx in emprestimos_idx:
                    emp_row = origins_df.loc[emp_idx]
                    
                    # Calcula distância básica entre corte e empréstimo
                    dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                    
                    # Adiciona DT fixo do empréstimo, se existir
                    if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                        dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                    
                    bf_distances[(o_idx, emp_idx)] = dist
            
            # Linhas com cortes
            for i, o_idx in enumerate(cortes_idx):
                dist_sheet.write(row+i, 0, f"Corte {o_idx}", cell_format)
                for j, emp_idx in enumerate(emprestimos_idx):
                    dist = bf_distances.get((o_idx, emp_idx), 0)
                    dist_sheet.write(row+i, j+1, dist, number_format)
        
        # Dados de entrada (origens)
        origem_sheet = workbook.add_worksheet('Origens')
        
        # Título
        origem_sheet.write(0, 0, 'Dados das Origens', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalhos das colunas
        for col, colname in enumerate(origins_df.columns):
            origem_sheet.write(row, col+1, colname, header_format)
        origem_sheet.write(row, 0, 'ID', header_format)
        row += 1
        
        # Dados das origens
        for i, (idx, o_row) in enumerate(origins_df.iterrows()):
            origem_sheet.write(row+i, 0, idx, cell_format)
            for j, col in enumerate(origins_df.columns):
                if pd.api.types.is_numeric_dtype(origins_df[col]):
                    origem_sheet.write(row+i, j+1, o_row[col] if pd.notna(o_row[col]) else "", number_format)
                else:
                    origem_sheet.write(row+i, j+1, o_row[col] if pd.notna(o_row[col]) else "", cell_format)
        
        # Adiciona bota-fora convencional
        bota_fora_col = len(origins_df.columns) + 1
        origem_sheet.write(row-1, bota_fora_col, "Bota-Fora (m³)", header_format)
        
        for i, (idx, bf) in enumerate(result['bota_fora'].items()):
            if bf > 0:
                origem_sheet.write(row+i, bota_fora_col, bf, number_format)
            else:
                origem_sheet.write(row+i, bota_fora_col, "", number_format)
        
        # Adiciona bota-fora em empréstimos recebido, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            bf_emp_col = len(origins_df.columns) + 2
            origem_sheet.write(row-1, bf_emp_col, "Bota-Fora Recebido (m³)", header_format)
            
            for i, (idx, o_row) in enumerate(origins_df.iterrows()):
                # Verifica se este índice é um empréstimo que recebeu bota-fora
                if idx in result['bota_fora_emprestimo'].columns:
                    bf_recebido = result['bota_fora_emprestimo'][idx].sum()
                    if bf_recebido > 0:
                        origem_sheet.write(row+i, bf_emp_col, bf_recebido, number_format)
                    else:
                        origem_sheet.write(row+i, bf_emp_col, "", number_format)
                else:
                    origem_sheet.write(row+i, bf_emp_col, "", number_format)
        
        # Ajusta larguras das colunas
        origem_sheet.set_column('A:A', 8)
        origem_sheet.set_column(1, len(origins_df.columns)+2, 15)
        
        # Dados de entrada (destinos)
        destino_sheet = workbook.add_worksheet('Destinos')
        
        # Título
        destino_sheet.write(0, 0, 'Dados dos Destinos', 
                           workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalhos das colunas
        for col, colname in enumerate(destinations_df.columns):
            destino_sheet.write(row, col+1, colname, header_format)
        destino_sheet.write(row, 0, 'ID', header_format)
        row += 1
        
        # Dados dos destinos
        for i, (idx, d_row) in enumerate(destinations_df.iterrows()):
            destino_sheet.write(row+i, 0, idx, cell_format)
            for j, col in enumerate(destinations_df.columns):
                if pd.api.types.is_numeric_dtype(destinations_df[col]):
                    destino_sheet.write(row+i, j+1, d_row[col] if pd.notna(d_row[col]) else "", number_format)
                else:
                    destino_sheet.write(row+i, j+1, d_row[col] if pd.notna(d_row[col]) else "", cell_format)
        
        # Adiciona colunas para verificação de atendimento
        col_offset = len(destinations_df.columns) + 1
        
        # CFT Atendido
        destino_sheet.write(row-1, col_offset, "CFT Atendido (m³)", header_format)
        # CFT Restante
        destino_sheet.write(row-1, col_offset+1, "CFT Faltante (m³)", header_format)
        # CA Atendido
        destino_sheet.write(row-1, col_offset+2, "CA Atendido (m³)", header_format)
        # CA Restante
        destino_sheet.write(row-1, col_offset+3, "CA Faltante (m³)", header_format)
        
        # Preenche valores
        for i, d_idx in enumerate(destinations_df.index):
            # CFT Atendido
            cft_atendido = result['cft'][d_idx].sum()
            destino_sheet.write(row+i, col_offset, cft_atendido, number_format)
            
            # CFT Restante
            cft_faltante = result['remaining_cft'][d_idx]
            destino_sheet.write(row+i, col_offset+1, cft_faltante if cft_faltante > 0 else "", number_format)
            
            # CA Atendido
            ca_atendido = result['ca'][d_idx].sum()
            destino_sheet.write(row+i, col_offset+2, ca_atendido, number_format)
            
            # CA Restante
            ca_faltante = result['remaining_ca'][d_idx]
            destino_sheet.write(row+i, col_offset+3, ca_faltante if ca_faltante > 0 else "", number_format)
        
        # Ajusta larguras das colunas
        destino_sheet.set_column('A:A', 8)
        destino_sheet.set_column(1, len(destinations_df.columns)+4, 15)
        
        # Resumo detalhado de origens
        detail_sheet = workbook.add_worksheet('Detalhamento por Origem')
        
        # Título
        detail_sheet.write(0, 0, 'Detalhamento de Uso por Origem', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalho
        detail_sheet.write(row, 0, 'ID', header_format)
        detail_sheet.write(row, 1, 'Tipo', header_format)
        detail_sheet.write(row, 2, 'Volume disponível (m³)', header_format)
        detail_sheet.write(row, 3, 'Volume para CFT (m³)', header_format)
        detail_sheet.write(row, 4, 'Volume para CA (m³)', header_format)
        detail_sheet.write(row, 5, 'Volume para Bota-fora (m³)', header_format)
        detail_sheet.write(row, 6, 'Volume não utilizado (m³)', header_format)
        detail_sheet.write(row, 7, 'Utilização (%)', header_format)
        
        # Adiciona coluna para bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            detail_sheet.write(row, 8, 'Bota-Fora Recebido (m³)', header_format)
        
        row += 1
        
        # Dados
        for i, o_idx in enumerate(origins_df.index):
            vol_disp = origins_df.loc[o_idx, 'Volume disponível (m³)']
            vol_cft = result['cft'].loc[o_idx].sum()
            vol_ca = result['ca'].loc[o_idx].sum()
            vol_bf = result['bota_fora'][o_idx] if o_idx in result['bota_fora'] else 0
            
            # Calcula o volume não utilizado para empréstimos
            vol_nao_utilizado = 0
            if o_idx in result.get('emprestimos_laterais_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_laterais_nao_utilizados'][o_idx]
            elif o_idx in result.get('emprestimos_concentrados_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_concentrados_nao_utilizados'][o_idx]
            elif o_idx in result.get('emprestimos_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_nao_utilizados'][o_idx]
            
            # Calcula a utilização
            vol_utilizado = vol_cft + vol_ca
            utilizacao = vol_utilizado / vol_disp if vol_disp > 0 else 0
            
            # Escreve os dados
            detail_sheet.write(row+i, 0, o_idx, cell_format)
            detail_sheet.write(row+i, 1, origins_df.loc[o_idx, 'Tipo'], cell_format)
            detail_sheet.write(row+i, 2, vol_disp, number_format)
            detail_sheet.write(row+i, 3, vol_cft if vol_cft > 0 else "", number_format)
            detail_sheet.write(row+i, 4, vol_ca if vol_ca > 0 else "", number_format)
            detail_sheet.write(row+i, 5, vol_bf if vol_bf > 0 else "", number_format)
            detail_sheet.write(row+i, 6, vol_nao_utilizado if vol_nao_utilizado > 0 else "", number_format)
            detail_sheet.write(row+i, 7, utilizacao, workbook.add_format({'num_format': '0.0%', 'border': 1}))
            
            # Adiciona volume de bota-fora recebido, se aplicável
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                vol_bf_recebido = 0
                if o_idx in result['bota_fora_emprestimo'].columns:
                    vol_bf_recebido = result['bota_fora_emprestimo'][o_idx].sum()
                
                detail_sheet.write(row+i, 8, vol_bf_recebido if vol_bf_recebido > 0 else "", number_format)
        
        # Ajusta larguras das colunas
        detail_sheet.set_column('A:A', 8)
        detail_sheet.set_column('B:B', 25)
        detail_sheet.set_column('C:H', 18)
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            detail_sheet.set_column('I:I', 18)
        
        # Resumo detalhado de destinos
        dest_detail_sheet = workbook.add_worksheet('Detalhamento por Destino')
        
        # Título
        dest_detail_sheet.write(0, 0, 'Detalhamento de Atendimento por Destino', 
                                workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalho
        dest_detail_sheet.write(row, 0, 'ID', header_format)
        dest_detail_sheet.write(row, 1, 'Centro de Massa (m)', header_format)
        dest_detail_sheet.write(row, 2, 'Volume CFT necessário (m³)', header_format)
        dest_detail_sheet.write(row, 3, 'Volume CFT atendido (m³)', header_format)
        dest_detail_sheet.write(row, 4, 'Volume CFT faltante (m³)', header_format)
        dest_detail_sheet.write(row, 5, 'Volume CA necessário (m³)', header_format)
        dest_detail_sheet.write(row, 6, 'Volume CA atendido (m³)', header_format)
        dest_detail_sheet.write(row, 7, 'Volume CA faltante (m³)', header_format)
        dest_detail_sheet.write(row, 8, 'Atendimento total (%)', header_format)
        row += 1
        
        # Dados
        for i, d_idx in enumerate(destinations_df.index):
            vol_cft_nec = destinations_df.loc[d_idx, 'Volume CFT (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CFT (m³)']) else 0
            vol_ca_nec = destinations_df.loc[d_idx, 'Volume CA (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CA (m³)']) else 0
            
            vol_cft_ate = result['cft'][d_idx].sum()
            vol_ca_ate = result['ca'][d_idx].sum()
            
            vol_cft_falt = max(0, vol_cft_nec - vol_cft_ate)
            vol_ca_falt = max(0, vol_ca_nec - vol_ca_ate)
            
            vol_total_nec = vol_cft_nec + vol_ca_nec
            vol_total_ate = vol_cft_ate + vol_ca_ate
            
            atendimento = vol_total_ate / vol_total_nec if vol_total_nec > 0 else 1
            
            # Escreve os dados
            dest_detail_sheet.write(row+i, 0, d_idx, cell_format)
            dest_detail_sheet.write(row+i, 1, destinations_df.loc[d_idx, 'Centro de Massa (m)'], number_format)
            dest_detail_sheet.write(row+i, 2, vol_cft_nec if vol_cft_nec > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 3, vol_cft_ate if vol_cft_ate > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 4, vol_cft_falt if vol_cft_falt > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 5, vol_ca_nec if vol_ca_nec > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 6, vol_ca_ate if vol_ca_ate > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 7, vol_ca_falt if vol_ca_falt > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 8, atendimento, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Ajusta larguras das colunas
        dest_detail_sheet.set_column('A:A', 8)
        dest_detail_sheet.set_column('B:H', 18)
        dest_detail_sheet.set_column('I:I', 15)
        
        # Dados da otimização
        params_sheet = workbook.add_worksheet('Parâmetros')
        
        # Título
        params_sheet.write(0, 0, 'Parâmetros da Otimização', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Parâmetros utilizados
        params_sheet.write(row, 0, 'Parâmetro', header_format)
        params_sheet.write(row, 1, 'Valor', header_format)
        row += 1
        
        params_sheet.write(row, 0, 'Favorecimento de materiais de corte', cell_format)
        params_sheet.write(row, 1, 'Sim' if result.get('favor_cortes', False) else 'Não', cell_format)
        row += 1
        
        params_sheet.write(row, 0, 'Distância máxima para cortes (m)', cell_format)
        if result.get('max_dist_cortes') is not None:
            params_sheet.write(row, 1, result['max_dist_cortes'], number_format)
        else:
            params_sheet.write(row, 1, "Sem limite", cell_format)
        row += 1
        
        if 'max_dist_emprestimos_laterais' in result:
            params_sheet.write(row, 0, 'Distância máxima para empréstimos laterais (m)', cell_format)
            if result.get('max_dist_emprestimos_laterais') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos_laterais'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
            
            params_sheet.write(row, 0, 'Distância máxima para empréstimos concentrados (m)', cell_format)
            if result.get('max_dist_emprestimos_concentrados') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos_concentrados'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
        elif 'max_dist_emprestimos' in result:
            params_sheet.write(row, 0, 'Distância máxima para empréstimos (m)', cell_format)
            if result.get('max_dist_emprestimos') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
        
        # Parâmetros de bota-fora em empréstimos
        if 'usar_emprestimos_bota_fora' in result:
            params_sheet.write(row, 0, 'Usar empréstimos como bota-fora', cell_format)
            params_sheet.write(row, 1, 'Sim' if result['usar_emprestimos_bota_fora'] else 'Não', cell_format)
            row += 1
            
            if result.get('usar_emprestimos_bota_fora'):
                params_sheet.write(row, 0, 'Priorizar bota-fora mais próximo', cell_format)
                params_sheet.write(row, 1, 'Sim' if result.get('priorizar_bf_proximo', True) else 'Não', cell_format)
                row += 1
                
                params_sheet.write(row, 0, 'Distância máxima para bota-fora em empréstimos (m)', cell_format)
                if result.get('max_dist_bota_fora') is not None:
                    params_sheet.write(row, 1, result['max_dist_bota_fora'], number_format)
                else:
                    params_sheet.write(row, 1, "Sem limite", cell_format)
                row += 1
        
        # Adiciona informações sobre alocações fixas, se houver
        fixed_allocs = result.get('fixed_allocations', [])
        if fixed_allocs:
            row += 2
            params_sheet.write(row, 0, 'Alocações Fixas Pré-definidas', 
                              workbook.add_format({'bold': True, 'font_size': 12}))
            row += 1
            
            params_sheet.write(row, 0, 'Origem', header_format)
            params_sheet.write(row, 1, 'Destino', header_format)
            params_sheet.write(row, 2, 'Volume (m³)', header_format)
            params_sheet.write(row, 3, 'Tipo', header_format)
            row += 1
            
            for i, alloc in enumerate(fixed_allocs):
                params_sheet.write(row+i, 0, alloc['origem'], cell_format)
                params_sheet.write(row+i, 1, alloc['destino'], cell_format)
                params_sheet.write(row+i, 2, float(alloc['volume']), number_format)
                params_sheet.write(row+i, 3, alloc['tipo'], cell_format)
        
        # Ajusta larguras das colunas
        params_sheet.set_column('A:A', 40)
        params_sheet.set_column('B:D', 15)
        
        # Quadro de Distribuição de Terraplenagem (consolidado)
        dist_quadro_sheet = workbook.add_worksheet('Quadro de Distribuição')

        # Título
        dist_quadro_sheet.write(0, 0, 'Quadro de Distribuição de Terraplenagem', 
                              workbook.add_format({'bold': True, 'font_size': 14}))

        # Cabeçalhos
        row = 2
        dist_quadro_sheet.write(row, 0, 'Origem', header_format)
        dist_quadro_sheet.write(row, 1, 'Destino', header_format)
        dist_quadro_sheet.write(row, 2, 'Utilização', header_format)
        dist_quadro_sheet.write(row, 3, 'Volume (m³)', header_format)
        dist_quadro_sheet.write(row, 4, 'DT (km)', header_format)
        dist_quadro_sheet.write(row, 5, 'Momento (m³.km)', header_format)
        row += 1

        # Dados de CFT
        all_movements = []
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                vol_cft = result['cft'].loc[o_idx, d_idx]
                if vol_cft > 0:
                    dt_metros = result['distances'].loc[o_idx, d_idx]
                    dt_km = dt_metros / 1000  # Converter para km
                    momento = vol_cft * dt_km  # Em m³.km
                    
                    all_movements.append({
                        'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                        'destino': d_idx,  # Já é o ID original pois é o índice do DataFrame
                        'utilizacao': 'CFT',
                        'volume': vol_cft,
                        'dt_km': dt_km,
                        'momento': momento,
                        'sort_key': 1  # Para ordenação: CFT primeiro
                    })

        # Dados de CA
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                vol_ca = result['ca'].loc[o_idx, d_idx]
                if vol_ca > 0:
                    dt_metros = result['distances'].loc[o_idx, d_idx]
                    dt_km = dt_metros / 1000  # Converter para km
                    momento = vol_ca * dt_km  # Em m³.km
                    
                    all_movements.append({
                        'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                        'destino': d_idx,  # Já é o ID original pois é o índice do DataFrame
                        'utilizacao': 'CA',
                        'volume': vol_ca,
                        'dt_km': dt_km,
                        'momento': momento,
                        'sort_key': 2  # Para ordenação: CA depois
                    })

        # Dados de Bota-fora convencional
        for o_idx, vol_bf in result['bota_fora'].items():
            if vol_bf > 0:
                all_movements.append({
                    'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                    'destino': "Bota-fora",
                    'utilizacao': 'Bota-fora',
                    'volume': vol_bf,
                    'dt_km': 0,  # Distância não aplicável para bota-fora
                    'momento': 0,  # Momento não aplicável para bota-fora
                    'sort_key': 3  # Para ordenação: Bota-fora por último
                })
        
        # Dados de Bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            bf_distances_data = {}
            
            # Calcula distâncias entre cortes e empréstimos
            for o_idx in result['bota_fora_emprestimo'].index:
                o_row = origins_df.loc[o_idx]
                for emp_idx in result['bota_fora_emprestimo'].columns:
                    if result['bota_fora_emprestimo'].loc[o_idx, emp_idx] > 0:
                        emp_row = origins_df.loc[emp_idx]
                        
                        # Calcula distância básica entre corte e empréstimo
                        dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                        
                        # Adiciona DT fixo do empréstimo, se existir
                        if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                            dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                        
                        bf_distances_data[(o_idx, emp_idx)] = dist
            
            # Adiciona movimentos de bota-fora em empréstimos
            for o_idx in result['bota_fora_emprestimo'].index:
                for emp_idx in result['bota_fora_emprestimo'].columns:
                    volume = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                    if volume > 0:
                        dt_metros = bf_distances_data.get((o_idx, emp_idx), 0)
                        dt_km = dt_metros / 1000  # Converter para km
                        momento = volume * dt_km  # Em m³.km
                        
                        all_movements.append({
                            'origem': o_idx,
                            'destino': f"E{emp_idx}",  # Prefixo E para indicar empréstimo
                            'utilizacao': 'Bota-fora em Empréstimo',
                            'volume': volume,
                            'dt_km': dt_km,
                            'momento': momento,
                            'sort_key': 4  # Para ordenação: Bota-fora em empréstimo por último
                        })

        # Ordenar movimentos (primeiro por origem, depois por tipo de utilização)
        all_movements.sort(key=lambda x: (x['origem'], x['sort_key']))

        # Escrever todos os movimentos
        for i, mov in enumerate(all_movements):
            dist_quadro_sheet.write(row + i, 0, mov['origem'], cell_format)
            dist_quadro_sheet.write(row + i, 1, mov['destino'], cell_format)
            dist_quadro_sheet.write(row + i, 2, mov['utilizacao'], cell_format)
            dist_quadro_sheet.write(row + i, 3, mov['volume'], number_format)
            dist_quadro_sheet.write(row + i, 4, mov['dt_km'], workbook.add_format({'num_format': '0.000', 'border': 1}))
            dist_quadro_sheet.write(row + i, 5, mov['momento'], workbook.add_format({'num_format': '0.000', 'border': 1}))

        # Linha com totais
        total_row = row + len(all_movements)
        dist_quadro_sheet.write(total_row, 0, "TOTAL", header_format)
        dist_quadro_sheet.write(total_row, 1, "", header_format)
        dist_quadro_sheet.write(total_row, 2, "", header_format)
        dist_quadro_sheet.write(total_row, 3, sum(mov['volume'] for mov in all_movements), number_format)
        dist_quadro_sheet.write(total_row, 4, "", header_format)
        dist_quadro_sheet.write(total_row, 5, sum(mov['momento'] for mov in all_movements), 
                              workbook.add_format({'num_format': '0.000', 'border': 1, 'bold': True}))

        # Ajusta larguras das colunas
        dist_quadro_sheet.set_column('A:B', 15)
        dist_quadro_sheet.set_column('C:C', 22)  # Coluna mais larga para "Bota-fora em Empréstimo"
        dist_quadro_sheet.set_column('D:F', 16)
        
    # Se foi fornecido um nome de arquivo, salva nele
    if filename:
        with open(filename, 'wb') as f:
            f.write(output.getvalue())
    
    # Retorna o BytesIO para usar no Streamlit
    output.seek(0)
    return output

def export_optimization_results(result, origins_df, destinations_df):
    """
    Exporta os resultados da otimização para JSON
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        String JSON com os resultados
    """
    if result is None:
        return json.dumps({'error': 'Não foi possível encontrar uma solução factível.'})
    
    # Prepara os resultados para exportação
    export_data = {
        'status': result['status'],
        'momento_total': round(float(result['momento_total']), 2),
        'dmt': round(float(result['dmt']), 2),
        'favor_cortes': result['favor_cortes'],
        'max_dist_cortes': result['max_dist_cortes'],
        'parametros': {
            'favor_cortes': result['favor_cortes'],
            'max_dist_cortes': result['max_dist_cortes']
        }
    }
    
    # Adiciona parâmetros de empréstimos, se disponíveis
    if 'max_dist_emprestimos_laterais' in result:
        export_data['parametros']['max_dist_emprestimos_laterais'] = result['max_dist_emprestimos_laterais']
        export_data['parametros']['max_dist_emprestimos_concentrados'] = result['max_dist_emprestimos_concentrados']
    elif 'max_dist_emprestimos' in result:
        export_data['parametros']['max_dist_emprestimos'] = result['max_dist_emprestimos']
    
    # Adiciona parâmetros de bota-fora em empréstimos, se aplicável
    if 'usar_emprestimos_bota_fora' in result:
        export_data['parametros']['usar_emprestimos_bota_fora'] = result['usar_emprestimos_bota_fora']
        if result.get('usar_emprestimos_bota_fora'):
            export_data['parametros']['priorizar_bf_proximo'] = result.get('priorizar_bf_proximo', True)
            export_data['parametros']['max_dist_bota_fora'] = result.get('max_dist_bota_fora')
            export_data['total_bf_emprestimo'] = round(float(result.get('total_bf_emprestimo', 0)), 2)
            export_data['momento_bf_emprestimo'] = round(float(result.get('momento_bf_emprestimo', 0)), 2)
    
    # Distribuição CFT
    cft_dist = []
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            value = result['cft'].loc[o_idx, d_idx]
            if value > 0:
                cft_dist.append({
                    'origem': str(o_idx),
                    'destino': str(d_idx),
                    'volume': round(float(value), 2),
                    'distancia': round(float(result['distances'].loc[o_idx, d_idx]), 2),
                    'momento': round(float(value * result['distances'].loc[o_idx, d_idx]), 2)
                })
    
    # Distribuição CA
    ca_dist = []
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            value = result['ca'].loc[o_idx, d_idx]
            if value > 0:
                ca_dist.append({
                    'origem': str(o_idx),
                    'destino': str(d_idx),
                    'volume': round(float(value), 2),
                    'distancia': round(float(result['distances'].loc[o_idx, d_idx]), 2),
                    'momento': round(float(value * result['distances'].loc[o_idx, d_idx]), 2)
                })
    
    # Bota-fora convencional
    bota_fora = []
    for o_idx, value in result['bota_fora'].items():
        if value > 0:
            bota_fora.append({
                'origem': str(o_idx),
                'volume': round(float(value), 2)
            })
    
    # Bota-fora em empréstimos, se aplicável
    bota_fora_emprestimo = []
    if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
        bf_distances_data = {}
        
        # Calcula distâncias entre cortes e empréstimos
        for o_idx in result['bota_fora_emprestimo'].index:
            o_row = origins_df.loc[o_idx]
            for emp_idx in result['bota_fora_emprestimo'].columns:
                if result['bota_fora_emprestimo'].loc[o_idx, emp_idx] > 0:
                    emp_row = origins_df.loc[emp_idx]
                    
                    # Calcula distância básica entre corte e empréstimo
                    dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                    
                    # Adiciona DT fixo do empréstimo, se existir
                    if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                        dist += origins_df.loc[emp_idx, 'DT Fixo (m)']
                    
                    bf_distances_data[(o_idx, emp_idx)] = dist
        
        # Adiciona dados de bota-fora em empréstimos
        for o_idx in result['bota_fora_emprestimo'].index:
            for emp_idx in result['bota_fora_emprestimo'].columns:
                value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                if value > 0:
                    dist = bf_distances_data.get((o_idx, emp_idx), 0)
                    
                    bota_fora_emprestimo.append({
                        'origem': str(o_idx),
                        'emprestimo': str(emp_idx),
                        'volume': round(float(value), 2),
                        'distancia': round(float(dist), 2),
                        'momento': round(float(value * dist), 2)
                    })
    
    # Empréstimos não utilizados
    emprestimos_nao_utilizados = []
    
    # Verifica se temos a classificação avançada ou a simples
    if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
        for o_idx, value in result['emprestimos_laterais_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Lateral'
                })
        for o_idx, value in result['emprestimos_concentrados_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Concentrado'
                })
    elif 'emprestimos_nao_utilizados' in result:
        for o_idx, value in result['emprestimos_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Empréstimo'
                })
    
    # Volumes restantes
    volumes_restantes = []
    for d_idx in destinations_df.index:
        cft_rest = result['remaining_cft'].get(d_idx, 0)
        ca_rest = result['remaining_ca'].get(d_idx, 0)
        
        if cft_rest > 0 or ca_rest > 0:
            volumes_restantes.append({
                'destino': str(d_idx),
                'volume_cft': round(float(cft_rest), 2) if cft_rest > 0 else 0,
                'volume_ca': round(float(ca_rest), 2) if ca_rest > 0 else 0
            })
    
    # Resumo por origem
    resumo_origens = []
    for o_idx in origins_df.index:
        vol_disp = origins_df.loc[o_idx, 'Volume disponível (m³)']
        vol_cft = result['cft'].loc[o_idx].sum()
        vol_ca = result['ca'].loc[o_idx].sum()
        vol_bf = result['bota_fora'].get(o_idx, 0)
        
        # Volume não utilizado
        vol_nao_utilizado = 0
        if 'emprestimos_laterais_nao_utilizados' in result and o_idx in result['emprestimos_laterais_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_laterais_nao_utilizados'][o_idx]
        elif 'emprestimos_concentrados_nao_utilizados' in result and o_idx in result['emprestimos_concentrados_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_concentrados_nao_utilizados'][o_idx]
        elif 'emprestimos_nao_utilizados' in result and o_idx in result['emprestimos_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_nao_utilizados'][o_idx]
        
        # Volume de bota-fora recebido, se aplicável
        vol_bf_recebido = 0
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            if o_idx in result['bota_fora_emprestimo'].columns:
                vol_bf_recebido = result['bota_fora_emprestimo'][o_idx].sum()
        
        # Tipo de origem
        tipo_origem = origins_df.loc[o_idx, 'Tipo']
        
        origem_data = {
            'id': str(o_idx),
            'tipo': tipo_origem,
            'volume_disponivel': round(float(vol_disp), 2),
            'volume_cft': round(float(vol_cft), 2),
            'volume_ca': round(float(vol_ca), 2),
            'volume_bota_fora': round(float(vol_bf), 2),
            'volume_nao_utilizado': round(float(vol_nao_utilizado), 2),
            'utilizacao': round(float((vol_cft + vol_ca) / vol_disp), 4) if vol_disp > 0 else 0
        }
        
        # Adiciona volume de bota-fora recebido, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora']:
            origem_data['volume_bf_recebido'] = round(float(vol_bf_recebido), 2)
        
        resumo_origens.append(origem_data)
    
    # Resumo por destino
    resumo_destinos = []
    for d_idx in destinations_df.index:
        vol_cft_nec = destinations_df.loc[d_idx, 'Volume CFT (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CFT (m³)']) else 0
        vol_ca_nec = destinations_df.loc[d_idx, 'Volume CA (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CA (m³)']) else 0
        
        vol_cft_ate = result['cft'][d_idx].sum()
        vol_ca_ate = result['ca'][d_idx].sum()
        
        vol_total_nec = vol_cft_nec + vol_ca_nec
        vol_total_ate = vol_cft_ate + vol_ca_ate
        
        atendimento = vol_total_ate / vol_total_nec if vol_total_nec > 0 else 1
        
        resumo_destinos.append({
            'id': str(d_idx),
            'volume_cft_necessario': round(float(vol_cft_nec), 2),
            'volume_cft_atendido': round(float(vol_cft_ate), 2),
            'volume_cft_faltante': round(float(max(0, vol_cft_nec - vol_cft_ate)), 2),
            'volume_ca_necessario': round(float(vol_ca_nec), 2),
            'volume_ca_atendido': round(float(vol_ca_ate), 2),
            'volume_ca_faltante': round(float(max(0, vol_ca_nec - vol_ca_ate)), 2),
            'atendimento': round(float(atendimento), 4)
        })
    
    # Totais
    export_data['totais'] = {
        'volume_cft': round(float(result['cft'].sum().sum()), 2),
        'volume_ca': round(float(result['ca'].sum().sum()), 2),
        'volume_bota_fora': round(float(result['bota_fora'].sum()), 2),
        'volume_nao_utilizado': round(float(result.get('total_emprestimo_nao_utilizado', 0)), 2)
    }
    
    # Adicionar as alocações fixas
    if 'fixed_allocations' in result and result['fixed_allocations']:
        export_data['alocacoes_fixas'] = result['fixed_allocations']
    
    # Adiciona todos os dados ao objeto final
    export_data.update({
        'distribuicao_cft': cft_dist,
        'distribuicao_ca': ca_dist,
        'bota_fora': bota_fora,
        'emprestimos_nao_utilizados': emprestimos_nao_utilizados,
        'volumes_restantes': volumes_restantes,
        'resumo_origens': resumo_origens,
        'resumo_destinos': resumo_destinos
    })
    
    # Adiciona dados de bota-fora em empréstimos, se aplicável
    if bota_fora_emprestimo:
        export_data['bota_fora_emprestimo'] = bota_fora_emprestimo
    
    return json.dumps(export_data, indent=2)
def create_distance_interval_report(result, origins_df, destinations_df, cost_per_interval=None):
    """
    Cria um relatório que divide o volume de material distribuído em intervalos específicos de distância.
    Inclui cálculo de custos para cada intervalo.
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
        cost_per_interval: Dicionário com custo por m³ para cada intervalo (descrição do serviço como chave)
    
    Returns:
        DataFrame com o relatório e BytesIO object com o arquivo Excel
    """
    if result is None:
        return None, None
    
    # Definir os intervalos de distância em metros
    intervals = [
        (0, 50, "ESCAVAÇÃO, CARGA E TRANSPORTE DE MATERIAL DE 1ª CATEGORIA ATÉ 50M"),
        (51, 200, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 51 A 200M)"),
        (201, 400, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 201 A 400M)"),
        (401, 600, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 401 A 600M)"),
        (601, 800, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 601 A 800M)"),
        (801, 1000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 801 A 1.000M)"),
        (1001, 1200, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.001 A 1.200M)"),
        (1201, 1400, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.201 A 1.400M)"),
        (1401, 1600, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.401 A 1.600M)"),
        (1601, 1800, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.601 A 1.800M)"),
        (1801, 2000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.801 A 2.000M)"),
        (2001, 3000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 2.001 A 3.000M)"),
        (3001, 5000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 3.001 A 5.000M)"),
        (5001, 10000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 5.001 A 10.000M)"),
        (10001, float('inf'), "ESCAV. E CARGA 1ª CATEG - DMT>10,0KM")
    ]
    
    # Categoria específica para material sem transporte
    sem_transporte = "ESCAV. E CARGA 1ª CATEG. - SEM TRANSPORTE"
    
    # Inicializar o dicionário para armazenar os volumes por intervalo
    volumes_por_intervalo = {desc: 0.0 for _, _, desc in intervals}
    volumes_por_intervalo[sem_transporte] = 0.0
    
    # Analisar distribuição CFT
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            vol_cft = result['cft'].loc[o_idx, d_idx]
            if vol_cft > 0:
                dist = result['distances'].loc[o_idx, d_idx]
                # Encontrar o intervalo correspondente
                found = False
                for min_dist, max_dist, desc in intervals:
                    if min_dist <= dist <= max_dist:
                        volumes_por_intervalo[desc] += vol_cft
                        found = True
                        break
                
                if not found:
                    # Se não encontrou intervalo, verificar se é sem transporte (dist = 0)
                    if dist == 0:
                        volumes_por_intervalo[sem_transporte] += vol_cft
    
    # Analisar distribuição CA
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            vol_ca = result['ca'].loc[o_idx, d_idx]
            if vol_ca > 0:
                dist = result['distances'].loc[o_idx, d_idx]
                # Encontrar o intervalo correspondente
                found = False
                for min_dist, max_dist, desc in intervals:
                    if min_dist <= dist <= max_dist:
                        volumes_por_intervalo[desc] += vol_ca
                        found = True
                        break
                
                if not found:
                    # Se não encontrou intervalo, verificar se é sem transporte (dist = 0)
                    if dist == 0:
                        volumes_por_intervalo[sem_transporte] += vol_ca
    
    # Analisar bota-fora em empréstimo, se aplicável
    if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
        # Distâncias entre origens e empréstimos
        bf_distances = result.get('bf_distances', {})
        
        for o_idx in result['bota_fora_emprestimo'].index:
            for emp_idx in result['bota_fora_emprestimo'].columns:
                vol_bf = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                if vol_bf > 0:
                    # Obter distância
                    dist = bf_distances.get((o_idx, emp_idx), 0)
                    
                    # Encontrar o intervalo correspondente
                    found = False
                    for min_dist, max_dist, desc in intervals:
                        if min_dist <= dist <= max_dist:
                            volumes_por_intervalo[desc] += vol_bf
                            found = True
                            break
                    
                    if not found:
                        # Se não encontrou intervalo, verificar se é sem transporte (dist = 0)
                        if dist == 0:
                            volumes_por_intervalo[sem_transporte] += vol_bf
    
    # Criar DataFrame com os resultados
    report_data = []
    
    # Se não temos informações de custo, usamos valores padrão ou zeros
    if cost_per_interval is None:
        cost_per_interval = {}
    
    for desc, volume in volumes_por_intervalo.items():
        if volume > 0:  # Só incluir intervalos que têm volume
            # Obter custo unitário do dicionário ou usar valor padrão
            custo_unitario = cost_per_interval.get(desc, 0.0)
            custo_total = volume * custo_unitario
            
            report_data.append({
                'Descrição do Serviço': desc,
                'Volume (m³)': round(volume, 2),
                'Custo Unitário (R$/m³)': round(custo_unitario, 2),
                'Custo Total (R$)': round(custo_total, 2)
            })
    
    # Adicionar linha total
    total_volume = sum(item['Volume (m³)'] for item in report_data)
    total_custo = sum(item['Custo Total (R$)'] for item in report_data)
    
    report_data.append({
        'Descrição do Serviço': 'TOTAL',
        'Volume (m³)': round(total_volume, 2),
        'Custo Unitário (R$/m³)': round(total_custo / total_volume, 2) if total_volume > 0 else 0,
        'Custo Total (R$)': round(total_custo, 2)
    })
    
    report_df = pd.DataFrame(report_data)
    
    # Criar arquivo Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formato para células
        header_format = workbook.add_format({
            'bold': True, 
            'text_wrap': True, 
            'valign': 'center',
            'align': 'center', 
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })
        
        money_format = workbook.add_format({
            'num_format': 'R$ #,##0.00',
            'border': 1
        })
        
        bold_number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1,
            'bold': True
        })
        
        bold_money_format = workbook.add_format({
            'num_format': 'R$ #,##0.00',
            'border': 1,
            'bold': True
        })
        
        # Criar planilha
        worksheet = workbook.add_worksheet('Relatório por Intervalos')
        
        # Título
        worksheet.write(0, 0, 'Relatório de Volume e Custo por Intervalos de Distância', 
                       workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Data e hora
        now = datetime.datetime.now()
        worksheet.write(1, 0, f'Relatório gerado em: {now.strftime("%d/%m/%Y %H:%M:%S")}')
        
        # Cabeçalhos
        worksheet.write(3, 0, 'Descrição do Serviço', header_format)
        worksheet.write(3, 1, 'Volume (m³)', header_format)
        worksheet.write(3, 2, 'Custo Unitário (R$/m³)', header_format)
        worksheet.write(3, 3, 'Custo Total (R$)', header_format)
        
        # Dados
        for i, row in enumerate(report_data):
            is_total = row['Descrição do Serviço'] == 'TOTAL'
            format_to_use = cell_format if not is_total else workbook.add_format({'bold': True, 'border': 1})
            number_fmt = number_format if not is_total else bold_number_format
            money_fmt = money_format if not is_total else bold_money_format
            
            worksheet.write(4 + i, 0, row['Descrição do Serviço'], format_to_use)
            worksheet.write(4 + i, 1, row['Volume (m³)'], number_fmt)
            worksheet.write(4 + i, 2, row['Custo Unitário (R$/m³)'], money_fmt)
            worksheet.write(4 + i, 3, row['Custo Total (R$)'], money_fmt)
        
        # Ajustar larguras
        worksheet.set_column('A:A', 85)
        worksheet.set_column('B:B', 15)
        worksheet.set_column('C:C', 20)
        worksheet.set_column('D:D', 20)
    
    output.seek(0)
    return report_df, output

def display_optimization_charts(result, origins_df, destinations_df):
    """
    Exibe gráficos sobre a distribuição otimizada usando Streamlit
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    """
    if result is None:
        st.warning("Não é possível gerar gráficos para uma solução não factível.")
        return
    
    # Identifica os tipos de empréstimos a partir dos resultados
    if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
        emprestimos_laterais_idx = list(result['emprestimos_laterais_nao_utilizados'].keys())
        emprestimos_concentrados_idx = list(result['emprestimos_concentrados_nao_utilizados'].keys())
        emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
    elif 'emprestimos_nao_utilizados' in result:
        emprestimos_idx = list(result['emprestimos_nao_utilizados'].keys())
        emprestimos_laterais_idx = []
        emprestimos_concentrados_idx = []
    else:
        # Tenta identificar a partir do tipo na origem
        try:
            emprestimos_idx = origins_df[
                origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)
            ].index.tolist()
            emprestimos_laterais_idx = origins_df[
                origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True)
            ].index.tolist()
            emprestimos_concentrados_idx = [idx for idx in emprestimos_idx 
                                          if idx not in emprestimos_laterais_idx]
        except Exception as e:
            print(f"Erro ao identificar tipos de empréstimos: {str(e)}")
            emprestimos_idx = []
            emprestimos_laterais_idx = []
            emprestimos_concentrados_idx = []
    
    cortes_idx = [idx for idx in origins_df.index if idx not in emprestimos_idx]
    
    # Organiza dados para gráficos
    
    # 1. Gráfico de distribuição de volumes por tipo de origem
    st.subheader("Distribuição de Volumes por Tipo de Origem")
    
    try:
        # Prepara dados para o gráfico com tratamento seguro
        volume_cortes_cft = 0
        volume_cortes_ca = 0
        volume_cortes_bf = 0
        
        # Calculando volumes para cortes com verificações seguras
        for idx in cortes_idx:
            try:
                # Soma volume CFT para este corte
                row_sum_cft = 0
                for d_idx in destinations_df.index:
                    try:
                        value = result['cft'].loc[idx, d_idx]
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            value = float(value.iloc[0]) if not value.empty else 0
                        else:
                            value = float(value)
                        row_sum_cft += value
                    except Exception as e:
                        print(f"Erro ao calcular volume CFT para corte {idx}, destino {d_idx}: {str(e)}")
                volume_cortes_cft += row_sum_cft
                
                # Soma volume CA para este corte
                row_sum_ca = 0
                for d_idx in destinations_df.index:
                    try:
                        value = result['ca'].loc[idx, d_idx]
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            value = float(value.iloc[0]) if not value.empty else 0
                        else:
                            value = float(value)
                        row_sum_ca += value
                    except Exception as e:
                        print(f"Erro ao calcular volume CA para corte {idx}, destino {d_idx}: {str(e)}")
                volume_cortes_ca += row_sum_ca
                
                # Soma volume bota-fora para este corte
                if idx in result['bota_fora'].index:
                    value = result['bota_fora'].loc[idx]
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        value = float(value.iloc[0]) if not value.empty else 0
                    else:
                        value = float(value)
                    volume_cortes_bf += value
            except Exception as e:
                print(f"Erro ao processar volumes para corte {idx}: {str(e)}")
        
        # Volumes de bota-fora em empréstimos, se aplicável
        volume_bf_emprestimo_lateral = 0
        volume_bf_emprestimo_concentrado = 0
        
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            # Para empréstimos laterais
            for emp_idx in emprestimos_laterais_idx:
                if emp_idx in result['bota_fora_emprestimo'].columns:
                    try:
                        col_sum = 0
                        for o_idx in result['bota_fora_emprestimo'].index:
                            try:
                                value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                                if isinstance(value, (pd.Series, pd.DataFrame)):
                                    value = float(value.iloc[0]) if not value.empty else 0
                                else:
                                    value = float(value)
                                col_sum += value
                            except Exception as e:
                                print(f"Erro ao calcular BF empréstimo lateral para origem {o_idx}, empréstimo {emp_idx}: {str(e)}")
                        volume_bf_emprestimo_lateral += col_sum
                    except Exception as e:
                        print(f"Erro ao processar BF para empréstimo lateral {emp_idx}: {str(e)}")
            
            # Para empréstimos concentrados
            for emp_idx in emprestimos_concentrados_idx:
                if emp_idx in result['bota_fora_emprestimo'].columns:
                    try:
                        col_sum = 0
                        for o_idx in result['bota_fora_emprestimo'].index:
                            try:
                                value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                                if isinstance(value, (pd.Series, pd.DataFrame)):
                                    value = float(value.iloc[0]) if not value.empty else 0
                                else:
                                    value = float(value)
                                col_sum += value
                            except Exception as e:
                                print(f"Erro ao calcular BF empréstimo concentrado para origem {o_idx}, empréstimo {emp_idx}: {str(e)}")
                        volume_bf_emprestimo_concentrado += col_sum
                    except Exception as e:
                        print(f"Erro ao processar BF para empréstimo concentrado {emp_idx}: {str(e)}")
        
        if emprestimos_laterais_idx and emprestimos_concentrados_idx:
            # Versão com empréstimos separados
            volume_emp_lat_cft = 0
            volume_emp_lat_ca = 0
            volume_emp_lat_nu = result.get('total_emprestimo_lateral_nao_utilizado', 0)
            
            # Calculando volumes para empréstimos laterais
            for idx in emprestimos_laterais_idx:
                try:
                    # Soma volume CFT para este empréstimo lateral
                    row_sum_cft = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['cft'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_cft += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CFT para empréstimo lateral {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_lat_cft += row_sum_cft
                    
                    # Soma volume CA para este empréstimo lateral
                    row_sum_ca = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['ca'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_ca += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CA para empréstimo lateral {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_lat_ca += row_sum_ca
                except Exception as e:
                    print(f"Erro ao processar volumes para empréstimo lateral {idx}: {str(e)}")
            
            volume_emp_conc_cft = 0
            volume_emp_conc_ca = 0
            volume_emp_conc_nu = result.get('total_emprestimo_concentrado_nao_utilizado', 0)
            
            # Calculando volumes para empréstimos concentrados
            for idx in emprestimos_concentrados_idx:
                try:
                    # Soma volume CFT para este empréstimo concentrado
                    row_sum_cft = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['cft'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_cft += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CFT para empréstimo concentrado {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_conc_cft += row_sum_cft
                    
                    # Soma volume CA para este empréstimo concentrado
                    row_sum_ca = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['ca'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_ca += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CA para empréstimo concentrado {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_conc_ca += row_sum_ca
                except Exception as e:
                    print(f"Erro ao processar volumes para empréstimo concentrado {idx}: {str(e)}")
            
            dados_grafico = {
                'Tipo': ['Cortes', 'Cortes', 'Cortes', 
                       'Emp. Laterais', 'Emp. Laterais', 'Emp. Laterais',
                       'Emp. Concentrados', 'Emp. Concentrados', 'Emp. Concentrados'],
                'Categoria': ['CFT', 'CA', 'Bota-fora',
                            'CFT', 'CA', 'Não Utilizado',
                            'CFT', 'CA', 'Não Utilizado'],
                'Volume (m³)': [volume_cortes_cft, volume_cortes_ca, volume_cortes_bf,
                              volume_emp_lat_cft, volume_emp_lat_ca, volume_emp_lat_nu,
                              volume_emp_conc_cft, volume_emp_conc_ca, volume_emp_conc_nu]
            }
            
            # Adiciona dados de bota-fora em empréstimos, se aplicável
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                if volume_bf_emprestimo_lateral > 0:
                    dados_grafico['Tipo'].append('Emp. Laterais')
                    dados_grafico['Categoria'].append('BF Recebido')
                    dados_grafico['Volume (m³)'].append(volume_bf_emprestimo_lateral)
                
                if volume_bf_emprestimo_concentrado > 0:
                    dados_grafico['Tipo'].append('Emp. Concentrados')
                    dados_grafico['Categoria'].append('BF Recebido')
                    dados_grafico['Volume (m³)'].append(volume_bf_emprestimo_concentrado)
        else:
            # Versão simplificada
            volume_emp_cft = 0
            volume_emp_ca = 0
            volume_emp_nu = result.get('total_emprestimo_nao_utilizado', 0)
            volume_bf_emprestimo_total = volume_bf_emprestimo_lateral + volume_bf_emprestimo_concentrado
            
            # Calculando volumes para empréstimos (versão simplificada)
            for idx in emprestimos_idx:
                try:
                    # Soma volume CFT para este empréstimo
                    row_sum_cft = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['cft'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_cft += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CFT para empréstimo {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_cft += row_sum_cft
                    
                    # Soma volume CA para este empréstimo
                    row_sum_ca = 0
                    for d_idx in destinations_df.index:
                        try:
                            value = result['ca'].loc[idx, d_idx]
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = float(value.iloc[0]) if not value.empty else 0
                            else:
                                value = float(value)
                            row_sum_ca += value
                        except Exception as e:
                            print(f"Erro ao calcular volume CA para empréstimo {idx}, destino {d_idx}: {str(e)}")
                    volume_emp_ca += row_sum_ca
                except Exception as e:
                    print(f"Erro ao processar volumes para empréstimo {idx}: {str(e)}")
            
            dados_grafico = {
                'Tipo': ['Cortes', 'Cortes', 'Cortes', 
                       'Empréstimos', 'Empréstimos', 'Empréstimos'],
                'Categoria': ['CFT', 'CA', 'Bota-fora',
                            'CFT', 'CA', 'Não Utilizado'],
                'Volume (m³)': [volume_cortes_cft, volume_cortes_ca, volume_cortes_bf,
                              volume_emp_cft, volume_emp_ca, volume_emp_nu]
            }
            
            # Adiciona dados de bota-fora em empréstimos, se aplicável
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                if volume_bf_emprestimo_total > 0:
                    dados_grafico['Tipo'].append('Empréstimos')
                    dados_grafico['Categoria'].append('BF Recebido')
                    dados_grafico['Volume (m³)'].append(volume_bf_emprestimo_total)
        
        chart_df = pd.DataFrame(dados_grafico)
        
        # Filtra categorias com volume zero para melhor visualização
        chart_df = chart_df[chart_df['Volume (m³)'] > 0]
        
        if not chart_df.empty:
            chart = st.bar_chart(
                chart_df,
                x="Tipo",
                y="Volume (m³)",
                color="Categoria"
            )
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de distribuição por tipo de origem: {str(e)}")
    
    # 2. Gráfico de distribuição de materiais por destino
    st.subheader("Distribuição de Materiais por Destino")
    
    try:
        dados_destinos = []
        for d_idx in destinations_df.index:
            try:
                # Calcular volume CFT atendido
                vol_cft_atendido = 0
                for o_idx in origins_df.index:
                    try:
                        value = result['cft'].loc[o_idx, d_idx]
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            value = float(value.iloc[0]) if not value.empty else 0
                        else:
                            value = float(value)
                        vol_cft_atendido += value
                    except Exception as e:
                        print(f"Erro ao calcular volume CFT atendido para destino {d_idx}, origem {o_idx}: {str(e)}")
                
                # Calcular volume CA atendido
                vol_ca_atendido = 0
                for o_idx in origins_df.index:
                    try:
                        value = result['ca'].loc[o_idx, d_idx]
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            value = float(value.iloc[0]) if not value.empty else 0
                        else:
                            value = float(value)
                        vol_ca_atendido += value
                    except Exception as e:
                        print(f"Erro ao calcular volume CA atendido para destino {d_idx}, origem {o_idx}: {str(e)}")
                
                # Acessar remaining_cft e remaining_ca de forma segura
                if d_idx in result['remaining_cft'].index:
                    cft_faltante = result['remaining_cft'].loc[d_idx]
                    if isinstance(cft_faltante, (pd.Series, pd.DataFrame)):
                        cft_faltante = float(cft_faltante.iloc[0]) if not cft_faltante.empty else 0
                    else:
                        cft_faltante = float(cft_faltante)
                else:
                    cft_faltante = 0
                
                if d_idx in result['remaining_ca'].index:
                    ca_faltante = result['remaining_ca'].loc[d_idx]
                    if isinstance(ca_faltante, (pd.Series, pd.DataFrame)):
                        ca_faltante = float(ca_faltante.iloc[0]) if not ca_faltante.empty else 0
                    else:
                        ca_faltante = float(ca_faltante)
                else:
                    ca_faltante = 0
                
                # Garantir valores positivos
                vol_cft_faltante = max(0, cft_faltante)
                vol_ca_faltante = max(0, ca_faltante)
                
                # Adicionar dados se houver volumes
                if vol_cft_atendido > 0:
                    dados_destinos.append({
                        'Destino': f"Dest. {d_idx}",
                        'Categoria': 'CFT Atendido',
                        'Volume (m³)': vol_cft_atendido
                    })
                if vol_ca_atendido > 0:
                    dados_destinos.append({
                        'Destino': f"Dest. {d_idx}",
                        'Categoria': 'CA Atendido',
                        'Volume (m³)': vol_ca_atendido
                    })
                if vol_cft_faltante > 0:
                    dados_destinos.append({
                        'Destino': f"Dest. {d_idx}",
                        'Categoria': 'CFT Faltante',
                        'Volume (m³)': vol_cft_faltante
                    })
                if vol_ca_faltante > 0:
                    dados_destinos.append({
                        'Destino': f"Dest. {d_idx}",
                        'Categoria': 'CA Faltante',
                        'Volume (m³)': vol_ca_faltante
                    })
            except Exception as e:
                print(f"Erro ao processar dados para o destino {d_idx}: {str(e)}")
                continue
        
        dest_chart_df = pd.DataFrame(dados_destinos)
        
        if not dest_chart_df.empty:
            chart = st.bar_chart(
                dest_chart_df,
                x="Destino",
                y="Volume (m³)",
                color="Categoria"
            )
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de distribuição por destino: {str(e)}")
    
    # 3. Gráfico de utilização de origens
    st.subheader("Utilização das Origens")
    
    try:
        dados_utilizacao = []
        for o_idx in origins_df.index:
            try:
                vol_disp = float(origins_df.loc[o_idx, 'Volume disponível (m³)'])
                
                # Calcular volume usado (CFT + CA)
                vol_cft = 0
                vol_ca = 0
                for d_idx in destinations_df.index:
                    try:
                        cft_value = result['cft'].loc[o_idx, d_idx]
                        if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                            cft_value = float(cft_value.iloc[0]) if not cft_value.empty else 0
                        else:
                            cft_value = float(cft_value)
                        vol_cft += cft_value
                        
                        ca_value = result['ca'].loc[o_idx, d_idx]
                        if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                            ca_value = float(ca_value.iloc[0]) if not ca_value.empty else 0
                        else:
                            ca_value = float(ca_value)
                        vol_ca += ca_value
                    except Exception as e:
                        print(f"Erro ao calcular volume usado para origem {o_idx}, destino {d_idx}: {str(e)}")
                
                vol_usado = vol_cft + vol_ca
                
                # Calcular volume bota-fora
                vol_bf = 0
                if o_idx in result['bota_fora'].index:
                    bf_value = result['bota_fora'].loc[o_idx]
                    if isinstance(bf_value, (pd.Series, pd.DataFrame)):
                        bf_value = float(bf_value.iloc[0]) if not bf_value.empty else 0
                    else:
                        bf_value = float(bf_value)
                    vol_bf = bf_value
                
                vol_nu = 0
                vol_bf_recebido = 0
                
                # Para empréstimos, calcular volume não utilizado
                if o_idx in emprestimos_idx:
                    # Para empréstimos, o não utilizado é o volume que não foi distribuído
                    if 'emprestimos_laterais_nao_utilizados' in result and o_idx in result['emprestimos_laterais_nao_utilizados']:
                        nu_value = result['emprestimos_laterais_nao_utilizados'][o_idx]
                        if isinstance(nu_value, (pd.Series, pd.DataFrame)):
                            nu_value = float(nu_value.iloc[0]) if not nu_value.empty else 0
                        else:
                            nu_value = float(nu_value)
                        vol_nu = nu_value
                    elif 'emprestimos_concentrados_nao_utilizados' in result and o_idx in result['emprestimos_concentrados_nao_utilizados']:
                        nu_value = result['emprestimos_concentrados_nao_utilizados'][o_idx]
                        if isinstance(nu_value, (pd.Series, pd.DataFrame)):
                            nu_value = float(nu_value.iloc[0]) if not nu_value.empty else 0
                        else:
                            nu_value = float(nu_value)
                        vol_nu = nu_value
                    elif 'emprestimos_nao_utilizados' in result and o_idx in result['emprestimos_nao_utilizados']:
                        nu_value = result['emprestimos_nao_utilizados'][o_idx]
                        if isinstance(nu_value, (pd.Series, pd.DataFrame)):
                            nu_value = float(nu_value.iloc[0]) if not nu_value.empty else 0
                        else:
                            nu_value = float(nu_value)
                        vol_nu = nu_value
                    
                    # Verifica se recebeu bota-fora, se aplicável
                    if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                        if o_idx in result['bota_fora_emprestimo'].columns:
                            # Soma todo bota-fora recebido por este empréstimo
                            for src_idx in result['bota_fora_emprestimo'].index:
                                try:
                                    bf_rec_value = result['bota_fora_emprestimo'].loc[src_idx, o_idx]
                                    if isinstance(bf_rec_value, (pd.Series, pd.DataFrame)):
                                        bf_rec_value = float(bf_rec_value.iloc[0]) if not bf_rec_value.empty else 0
                                    else:
                                        bf_rec_value = float(bf_rec_value)
                                    vol_bf_recebido += bf_rec_value
                                except Exception as e:
                                    print(f"Erro ao calcular BF recebido para empréstimo {o_idx}, origem {src_idx}: {str(e)}")
                
                # Calcula a utilização percentual
                utilizacao = vol_usado / vol_disp if vol_disp > 0 else 0
                
                # Determina o tipo para legenda do gráfico
                tipo_origem = 'Corte'
                if o_idx in emprestimos_laterais_idx:
                    tipo_origem = 'Emp. Lateral'
                elif o_idx in emprestimos_concentrados_idx:
                    tipo_origem = 'Emp. Concentrado'
                elif o_idx in emprestimos_idx:
                    tipo_origem = 'Empréstimo'
                
                origem_data = {
                    'Origem': f"Origem {o_idx}",
                    'Utilização (%)': utilizacao * 100,
                    'Tipo': tipo_origem,
                    'Volume Disponível (m³)': vol_disp,
                    'Volume Usado (m³)': vol_usado,
                    'Volume Bota-fora/Não Util. (m³)': vol_bf if o_idx in cortes_idx else vol_nu
                }
                
                # Adiciona volume de bota-fora recebido, se aplicável
                if vol_bf_recebido > 0:
                    origem_data['BF Recebido (m³)'] = vol_bf_recebido
                
                dados_utilizacao.append(origem_data)
            except Exception as e:
                print(f"Erro ao processar utilização para origem {o_idx}: {str(e)}")
                continue
        
        util_chart_df = pd.DataFrame(dados_utilizacao)
        
        # Ordena por utilização
        util_chart_df = util_chart_df.sort_values('Utilização (%)', ascending=False)
        
        if not util_chart_df.empty:
            chart = st.bar_chart(
                util_chart_df,
                x="Origem",
                y="Utilização (%)",
                color="Tipo"
            )
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de utilização das origens: {str(e)}")
    
    # 4. Gráfico de histograma de distância média
    st.subheader("Histograma de Distâncias de Transporte")
    
    try:
        dados_distancia = []
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                try:
                    # Verifica CFT
                    cft_value = result['cft'].loc[o_idx, d_idx]
                    if isinstance(cft_value, (pd.Series, pd.DataFrame)):
                        cft_value = float(cft_value.iloc[0]) if not cft_value.empty else 0
                    else:
                        cft_value = float(cft_value)
                    
                    if cft_value > 0:
                        dist_value = result['distances'].loc[o_idx, d_idx]
                        if isinstance(dist_value, (pd.Series, pd.DataFrame)):
                            dist_value = float(dist_value.iloc[0]) if not dist_value.empty else 0
                        else:
                            dist_value = float(dist_value)
                        
                        # Determina o tipo para legenda
                        tipo_origem = 'Corte'
                        if o_idx in emprestimos_laterais_idx:
                            tipo_origem = 'Emp. Lateral'
                        elif o_idx in emprestimos_concentrados_idx:
                            tipo_origem = 'Emp. Concentrado'
                        elif o_idx in emprestimos_idx:
                            tipo_origem = 'Empréstimo'
                        
                        dados_distancia.append({
                            'Distância (m)': dist_value,
                            'Volume (m³)': cft_value,
                            'Material': 'CFT',
                            'Tipo Origem': tipo_origem
                        })
                except Exception as e:
                    print(f"Erro ao processar distância CFT para origem {o_idx}, destino {d_idx}: {str(e)}")
                
                try:
                                        # Verifica CA
                    ca_value = result['ca'].loc[o_idx, d_idx]
                    if isinstance(ca_value, (pd.Series, pd.DataFrame)):
                        ca_value = float(ca_value.iloc[0]) if not ca_value.empty else 0
                    else:
                        ca_value = float(ca_value)
                    
                    if ca_value > 0:
                        dist_value = result['distances'].loc[o_idx, d_idx]
                        if isinstance(dist_value, (pd.Series, pd.DataFrame)):
                            dist_value = float(dist_value.iloc[0]) if not dist_value.empty else 0
                        else:
                            dist_value = float(dist_value)
                        
                        # Determina o tipo para legenda
                        tipo_origem = 'Corte'
                        if o_idx in emprestimos_laterais_idx:
                            tipo_origem = 'Emp. Lateral'
                        elif o_idx in emprestimos_concentrados_idx:
                            tipo_origem = 'Emp. Concentrado'
                        elif o_idx in emprestimos_idx:
                            tipo_origem = 'Empréstimo'
                        
                        dados_distancia.append({
                            'Distância (m)': dist_value,
                            'Volume (m³)': ca_value,
                            'Material': 'CA',
                            'Tipo Origem': tipo_origem
                        })
                except Exception as e:
                    print(f"Erro ao processar distância CA para origem {o_idx}, destino {d_idx}: {str(e)}")
        
        # Adiciona dados de bota-fora em empréstimos, se aplicável
        if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
            # Extrair dados para gráfico
            try:
                # Primeiro cria um DataFrame de distâncias BF
                bf_distances_data = {}
                for o_idx in cortes_idx:
                    o_row = origins_df.loc[o_idx]
                    for emp_idx in emprestimos_idx:
                        if emp_idx in result['bota_fora_emprestimo'].columns:
                            bf_value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                            if isinstance(bf_value, (pd.Series, pd.DataFrame)):
                                bf_value = float(bf_value.iloc[0]) if not bf_value.empty else 0
                            else:
                                bf_value = float(bf_value)
                            
                            if bf_value > 0:
                                # Calcula distância básica
                                emp_row = origins_df.loc[emp_idx]
                                dist = abs(float(o_row['Centro de Massa (m)']) - float(emp_row['Centro de Massa (m)']))
                                
                                # Adiciona DT fixo, se existir
                                if 'DT Fixo (m)' in origins_df.columns and pd.notna(origins_df.loc[emp_idx, 'DT Fixo (m)']):
                                    dist += float(origins_df.loc[emp_idx, 'DT Fixo (m)'])
                                
                                bf_distances_data[(o_idx, emp_idx)] = dist
                
                # Adiciona os dados ao gráfico
                for (o_idx, emp_idx), dist in bf_distances_data.items():
                    try:
                        bf_value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                        if isinstance(bf_value, (pd.Series, pd.DataFrame)):
                            bf_value = float(bf_value.iloc[0]) if not bf_value.empty else 0
                        else:
                            bf_value = float(bf_value)
                        
                        if bf_value > 0:
                            dados_distancia.append({
                                'Distância (m)': dist,
                                'Volume (m³)': bf_value,
                                'Material': 'BF Empréstimo',
                                'Tipo Origem': 'Corte'
                            })
                    except Exception as e:
                        print(f"Erro ao processar distância BF para origem {o_idx}, empréstimo {emp_idx}: {str(e)}")
            except Exception as e:
                print(f"Erro ao calcular distâncias de bota-fora em empréstimos: {str(e)}")
        
        dist_chart_df = pd.DataFrame(dados_distancia)
        
        if not dist_chart_df.empty:
            # Histograma de distância
            hist_values = dist_chart_df['Distância (m)'].tolist()
            
            if hist_values:
                st.write(f"Distância Média de Transporte: {result['dmt']:.2f} m")
                st.write(f"Distância Mínima: {min(hist_values):.2f} m")
                st.write(f"Distância Máxima: {max(hist_values):.2f} m")
                
                # Criar histograma com base no volume
                # Agrupa por faixas de distância
                dist_chart_df['Faixa de Distância'] = pd.cut(
                    dist_chart_df['Distância (m)'], 
                    bins=10,
                    include_lowest=True
                )
                
                # Calcula o volume por faixa
                try:
                    volume_por_faixa = dist_chart_df.groupby(['Faixa de Distância', 'Material'])['Volume (m³)'].sum().reset_index()
                    
                    # Converte a faixa para string para melhor visualização
                    volume_por_faixa['Faixa de Distância'] = volume_por_faixa['Faixa de Distância'].astype(str)
                    
                    chart = st.bar_chart(
                        volume_por_faixa,
                        x="Faixa de Distância",
                        y="Volume (m³)",
                        color="Material"
                    )
                except Exception as e:
                    st.error(f"Erro ao criar histograma por volume: {str(e)}")
    except Exception as e:
        st.error(f"Erro ao gerar histograma de distâncias: {str(e)}")
    
    # 5. Gráfico específico para bota-fora em empréstimos, se aplicável
    if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
        st.subheader("Distribuição de Bota-Fora em Empréstimos")
        
        try:
            dados_bf_emprestimo = []
            for o_idx in result['bota_fora_emprestimo'].index:
                for emp_idx in result['bota_fora_emprestimo'].columns:
                    try:
                        bf_value = result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                        if isinstance(bf_value, (pd.Series, pd.DataFrame)):
                            bf_value = float(bf_value.iloc[0]) if not bf_value.empty else 0
                        else:
                            bf_value = float(bf_value)
                        
                        if bf_value > 0:
                            tipo_emprestimo = 'Emp. Lateral' if emp_idx in emprestimos_laterais_idx else 'Emp. Concentrado'
                            dados_bf_emprestimo.append({
                                'Corte': f"Corte {o_idx}",
                                'Empréstimo': f"Emp. {emp_idx}",
                                'Tipo Empréstimo': tipo_emprestimo,
                                'Volume (m³)': bf_value
                            })
                    except Exception as e:
                        print(f"Erro ao processar BF em empréstimo para origem {o_idx}, empréstimo {emp_idx}: {str(e)}")
            
            bf_emp_df = pd.DataFrame(dados_bf_emprestimo)
            
            if not bf_emp_df.empty:
                # Grupo por empréstimo
                chart1 = st.bar_chart(
                    bf_emp_df,
                    x="Empréstimo",
                    y="Volume (m³)",
                    color="Corte"
                )
                
                # Grupo por corte
                chart2 = st.bar_chart(
                    bf_emp_df,
                    x="Corte",
                    y="Volume (m³)",
                    color="Tipo Empréstimo"
                )
        except Exception as e:
            st.error(f"Erro ao gerar gráficos de bota-fora em empréstimos: {str(e)}")


def validate_input_data(origins_df, destinations_df):
    """
    Valida os dados de entrada e corrige problemas comuns
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        Tuple com (origins_df, destinations_df) validados e corrigidos,
        lista de mensagens de aviso
    """
    warnings = []
    
    # Cria cópias para não modificar os originais
    origins = origins_df.copy()
    destinations = destinations_df.copy()
    
    # Verifica e corrige valores nulos
    for col in ["Volume disponível (m³)", "ISC", "Centro de Massa (m)"]:
        if col in origins.columns and origins[col].isna().any():
            count_nulls = origins[col].isna().sum()
            warnings.append(f"Encontrados {count_nulls} valores nulos na coluna '{col}' das origens. Serão substituídos.")
            
            if col == "Volume disponível (m³)":
                origins[col] = origins[col].fillna(0)
            elif col == "ISC":
                origins[col] = origins[col].fillna(0)
            elif col == "Centro de Massa (m)":
                # Usa a média ou 0 se não houver valores
                fill_value = origins[col].mean() if not origins[col].isna().all() else 0
                origins[col] = origins[col].fillna(fill_value)
    
    # Verifica e corrige valores nulos nos destinos
    for col in ["Volume CFT (m³)", "Volume CA (m³)", "Centro de Massa (m)"]:
        if col in destinations.columns and destinations[col].isna().any():
            count_nulls = destinations[col].isna().sum()
            warnings.append(f"Encontrados {count_nulls} valores nulos na coluna '{col}' dos destinos. Serão substituídos.")
            
            if col in ["Volume CFT (m³)", "Volume CA (m³)"]:
                destinations[col] = destinations[col].fillna(0)
            elif col == "Centro de Massa (m)":
                # Usa a média ou 0 se não houver valores
                fill_value = destinations[col].mean() if not destinations[col].isna().all() else 0
                destinations[col] = destinations[col].fillna(fill_value)
    
    # Verifica se a coluna de ISC mínimo existe, se não, cria
    if "ISC mínimo exigido" not in destinations.columns:
        destinations["ISC mínimo exigido"] = 0
        warnings.append("Coluna 'ISC mínimo exigido' não encontrada nos destinos. Criada com valor 0.")
    
    # Verifica se a coluna de tipo existe nas origens, se não, tenta inferir
    if "Tipo" not in origins.columns:
        # Tenta inferir o tipo pela descrição ou outras colunas
        origins["Tipo"] = "Corte"  # Valor padrão
        warnings.append("Coluna 'Tipo' não encontrada nas origens. Todos os materiais serão considerados como 'Corte'.")
    
    # Verifica valores negativos em volumes
    if (origins["Volume disponível (m³)"] < 0).any():
        count_neg = (origins["Volume disponível (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume de origens. Serão substituídos por 0.")
        origins.loc[origins["Volume disponível (m³)"] < 0, "Volume disponível (m³)"] = 0
    
    if "Volume CFT (m³)" in destinations.columns and (destinations["Volume CFT (m³)"] < 0).any():
        count_neg = (destinations["Volume CFT (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume CFT de destinos. Serão substituídos por 0.")
        destinations.loc[destinations["Volume CFT (m³)"] < 0, "Volume CFT (m³)"] = 0
    
    if "Volume CA (m³)" in destinations.columns and (destinations["Volume CA (m³)"] < 0).any():
        count_neg = (destinations["Volume CA (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume CA de destinos. Serão substituídos por 0.")
        destinations.loc[destinations["Volume CA (m³)"] < 0, "Volume CA (m³)"] = 0
    
    return origins, destinations, warnings

def prepare_data_for_optimization(origins_df, destinations_df):
    """
    Prepara os dados para otimização, garantindo que estejam no formato correto
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        Tuple com (origins_df, destinations_df) prontos para otimização
    """
    # Valida e corrige os dados
    origins, destinations, warnings = validate_input_data(origins_df, destinations_df)
    
    # REMOVIDO: Não converter índices para string, pois já são os IDs corretos
    # origins.index = origins.index.astype(str)
    # destinations.index = destinations.index.astype(str)
    
    # Exibe avisos no Streamlit, se houver
    for warning in warnings:
        st.warning(warning)
    
    return origins, destinations

def create_interface():
    """
    Cria a interface do Streamlit para o sistema de otimização
    """
    st.set_page_config(page_title="Otimização de Distribuição de Terraplenagem", layout="wide")
    
    st.title("Sistema de Distribuição de Terraplenagem")
    
    with st.sidebar:
        st.header("Opções")
        
        tab = st.radio("Selecione a operação:", 
                     ["Carregar Dados", "Visualizar Dados", 
                      "Executar Otimização", "Visualizar Resultados", "Exportar"])
    
    # Inicializa estados da sessão, se não existirem
    if 'origins_df' not in st.session_state:
        st.session_state.origins_df = None
    if 'destinations_df' not in st.session_state:
        st.session_state.destinations_df = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'fixed_allocations' not in st.session_state:
        st.session_state.fixed_allocations = []
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = None
    if 'show_add_form' not in st.session_state:
        st.session_state.show_add_form = False
    if 'current_volume_disponivel' not in st.session_state:
        st.session_state.current_volume_disponivel = {}
    if 'current_volume_alocado' not in st.session_state:
        st.session_state.current_volume_alocado = {}
    if 'selected_origem' not in st.session_state:
        st.session_state.selected_origem = None
    
    # ABA: CARREGAR DADOS
    if tab == "Carregar Dados":
        st.header("Carregar Dados de Origem e Destino")
        
        st.subheader("1. Dados das Origens")
        st.write("""
        Carregue um arquivo Excel ou CSV com os dados das origens (cortes e empréstimos).
        
        O arquivo deve ter as seguintes colunas:
        - Tipo: Descrição do tipo de origem (Corte, Empréstimo, Empréstimo Lateral, etc.)
        - Centro de Massa (m): Posição (em metros) do centro de massa
        - Volume disponível (m³): Volume total disponível
        - ISC: Índice de Suporte Califórnia do material
        - DT Fixo (m) (opcional): Distância adicional fixa para empréstimos
        """)
        
        origins_file = st.file_uploader("Escolha o arquivo com os dados das origens", 
                                      type=["xlsx", "xls", "csv"], key="origins_uploader")
        
        if origins_file is not None:
            try:
                if origins_file.name.endswith('.csv'):
                    origins_df = pd.read_csv(origins_file, index_col="ID")  # Usar ID como índice
                else:
                    origins_df = pd.read_excel(origins_file, index_col="ID")  # Usar ID como índice
                
                # Verifica colunas mínimas necessárias
                required_cols = ["Tipo", "Centro de Massa (m)", "Volume disponível (m³)", "ISC"]
                missing_cols = [col for col in required_cols if col not in origins_df.columns]
                
                if missing_cols:
                    st.error(f"Colunas obrigatórias não encontradas no arquivo de origens: {', '.join(missing_cols)}")
                else:
                    st.session_state.origins_df = origins_df
                    st.success(f"Dados carregados com sucesso: {len(origins_df)} origens encontradas")
                    st.write(origins_df)
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {str(e)}")
        
        st.subheader("2. Dados dos Destinos")
        st.write("""
        Carregue um arquivo Excel ou CSV com os dados dos destinos.
        
        O arquivo deve ter as seguintes colunas:
        - Centro de Massa (m): Posição (em metros) do centro de massa
        - Volume CFT (m³): Volume necessário para corpo de fundação terroso
        - Volume CA (m³): Volume necessário para corpo do aterro
        - ISC mínimo exigido: ISC mínimo exigido para o material de CFT
        """)
        
        destinations_file = st.file_uploader("Escolha o arquivo com os dados dos destinos", 
                                          type=["xlsx", "xls", "csv"], key="destinations_uploader")
        
        if destinations_file is not None:
            try:
                if destinations_file.name.endswith('.csv'):
                    destinations_df = pd.read_csv(destinations_file, index_col="ID")  # Usar ID como índice
                else:
                    destinations_df = pd.read_excel(destinations_file, index_col="ID")  # Usar ID como índice
                
                # Verifica colunas mínimas necessárias
                required_cols = ["Centro de Massa (m)", "Volume CFT (m³)", "Volume CA (m³)"]
                missing_cols = [col for col in required_cols if col not in destinations_df.columns]
                
                if missing_cols:
                    st.error(f"Colunas obrigatórias não encontradas no arquivo de destinos: {', '.join(missing_cols)}")
                else:
                    st.session_state.destinations_df = destinations_df
                    st.success(f"Dados carregados com sucesso: {len(destinations_df)} destinos encontrados")
                    st.write(destinations_df)
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {str(e)}")
    
    # ABA: VISUALIZAR DADOS
    elif tab == "Visualizar Dados":
        st.header("Visualizar Dados Carregados")
        
        if st.session_state.origins_df is not None and st.session_state.destinations_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dados das Origens")
                st.dataframe(st.session_state.origins_df)
                
                # Resumo das origens
                st.subheader("Resumo das Origens")
                origens_por_tipo = st.session_state.origins_df.groupby('Tipo').agg({
                    'Volume disponível (m³)': 'sum',
                    'ISC': 'mean'
                })
                origens_por_tipo = origens_por_tipo.reset_index()
                origens_por_tipo.columns = ['Tipo', 'Volume Total (m³)', 'ISC Médio']
                st.dataframe(origens_por_tipo)
                
                # Gráfico de volumes por tipo
                st.subheader("Volume por Tipo de Origem")
                chart_data = pd.DataFrame({
                    'Tipo': st.session_state.origins_df['Tipo'],
                    'Volume (m³)': st.session_state.origins_df['Volume disponível (m³)']
                })
                st.bar_chart(chart_data, x='Tipo', y='Volume (m³)')
            
            with col2:
                st.subheader("Dados dos Destinos")
                st.dataframe(st.session_state.destinations_df)
                
                # Resumo dos destinos
                st.subheader("Resumo dos Destinos")
                total_cft = st.session_state.destinations_df['Volume CFT (m³)'].fillna(0).sum()
                total_ca = st.session_state.destinations_df['Volume CA (m³)'].fillna(0).sum()
                
                resumo_destinos = pd.DataFrame({
                    'Tipo': ['CFT', 'CA', 'Total'],
                    'Volume Total (m³)': [total_cft, total_ca, total_cft + total_ca]
                })
                st.dataframe(resumo_destinos)
                
                # Gráfico de volumes por destino
                st.subheader("Volume por Destino")
                chart_data = pd.DataFrame()
                
                for i, row in st.session_state.destinations_df.iterrows():
                    if pd.notna(row['Volume CFT (m³)']) and row['Volume CFT (m³)'] > 0:
                        chart_data = pd.concat([chart_data, pd.DataFrame({
                            'Destino': [f"Destino {i}"],
                            'Tipo': ['CFT'],
                            'Volume (m³)': [row['Volume CFT (m³)']]
                        })])
                    
                    if pd.notna(row['Volume CA (m³)']) and row['Volume CA (m³)'] > 0:
                        chart_data = pd.concat([chart_data, pd.DataFrame({
                            'Destino': [f"Destino {i}"],
                            'Tipo': ['CA'],
                            'Volume (m³)': [row['Volume CA (m³)']]
                        })])
                
                st.bar_chart(chart_data, x='Destino', y='Volume (m³)', color='Tipo')
            
            # Verificação de factibilidade
            st.subheader("Verificação de Factibilidade")
            is_feasible, message = check_feasibility(st.session_state.origins_df, st.session_state.destinations_df)
            
            if is_feasible:
                st.success(message)
                
                total_origem = st.session_state.origins_df['Volume disponível (m³)'].sum()
                total_destino = (st.session_state.destinations_df['Volume CFT (m³)'].fillna(0) + 
                                st.session_state.destinations_df['Volume CA (m³)'].fillna(0)).sum()
                
                st.write(f"Volume total disponível nas origens: {total_origem:.2f} m³")
                st.write(f"Volume total necessário nos destinos: {total_destino:.2f} m³")
                st.write(f"Diferença (volume excedente): {total_origem - total_destino:.2f} m³")
            else:
                st.error(message)
                
                total_origem = st.session_state.origins_df['Volume disponível (m³)'].sum()
                total_destino = (st.session_state.destinations_df['Volume CFT (m³)'].fillna(0) + 
                                st.session_state.destinations_df['Volume CA (m³)'].fillna(0)).sum()
                
                st.write(f"Volume total disponível nas origens: {total_origem:.2f} m³")
                st.write(f"Volume total necessário nos destinos: {total_destino:.2f} m³")
                st.write(f"Déficit de volume: {total_destino - total_origem:.2f} m³")
        else:
            st.warning("Carregue os dados das origens e destinos antes de visualizar.")

    # ABA: EXECUTAR OTIMIZAÇÃO
    elif tab == "Executar Otimização":
        st.header("Executar Otimização de Distribuição")
        
        if st.session_state.origins_df is not None and st.session_state.destinations_df is not None:
            # Parâmetros da execução
            st.subheader("Parâmetros da Execução")
            
            favor_cortes = st.checkbox("Favorecer materiais de corte sobre empréstimos", value=True)
            
            time_limit = 1800  # Tempo limite em segundos
            
            use_max_dist = st.checkbox("Limitar distâncias de transporte", value=False)
            max_dist_cortes = None
            max_dist_emprestimos_laterais = None
            max_dist_emprestimos_concentrados = None
            
            # Verifica se existem tipos específicos de empréstimos
            origens = st.session_state.origins_df
            has_laterais = any(origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            has_concentrados = any(origens['Tipo'].str.contains('Empr|empr|EMPR', regex=True) & 
                                 ~origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            
            if use_max_dist:
                max_dist_cortes = st.number_input("Distância máxima para materiais de corte (m)", 
                                               min_value=0.0, value=5000.0, step=100.0)
                
                if has_laterais and has_concentrados:
                    # Configuração separada para cada tipo
                    max_dist_emprestimos_laterais = st.number_input(
                        "Distância máxima para empréstimos laterais (m)", 
                        min_value=0.0, value=2000.0, step=100.0
                    )
                    max_dist_emprestimos_concentrados = st.number_input(
                        "Distância máxima para empréstimos concentrados (m)", 
                        min_value=0.0, value=10000.0, step=100.0
                    )
                else:
                    # Configuração única para todos os empréstimos
                    max_dist_emprestimos = st.number_input(
                        "Distância máxima para empréstimos (m)", 
                        min_value=0.0, value=5000.0, step=100.0
                    )
                    max_dist_emprestimos_laterais = max_dist_emprestimos
                    max_dist_emprestimos_concentrados = max_dist_emprestimos
            
            # Opção para usar empréstimos como bota-fora
            st.subheader("Configurações de Bota-Fora")
            
            usar_emprestimos_bota_fora = st.checkbox(
                "Permitir usar áreas de empréstimo como bota-fora", 
                value=False,
                help="Se marcado, as áreas de empréstimo poderão receber material de bota-fora equivalente ao volume retirado"
            )
            
            # Opções adicionais que só aparecem se usar empréstimos como bota-fora estiver marcado
            if usar_emprestimos_bota_fora:
                priorizar_bf_proximo = st.checkbox(
                    "Priorizar bota-fora mais próximo", 
                    value=True,
                    help="Se marcado, o sistema sempre buscará enviar o material para o bota-fora mais próximo"
                )
                
                max_dist_bota_fora = st.number_input(
                    "Distância máxima para bota-fora em empréstimos (m)",
                    min_value=0.0, 
                    value=5000.0,
                    step=100.0,
                    help="Define a distância máxima permitida para transporte de material para bota-fora"
                )
            else:
                priorizar_bf_proximo = True
                max_dist_bota_fora = None

            # Alocações fixas
            st.subheader("Alocações Fixas")

            st.write("""
                     Defina alocações fixas que o otimizador deve respeitar. 
                     Essas alocações serão mantidas independentemente da otimização.
                     """)

            # Exibir alocações fixas atuais
            if st.session_state.fixed_allocations:
                st.write("Alocações Fixas Atuais:")
                fixed_alloc_df = pd.DataFrame(st.session_state.fixed_allocations)
                st.dataframe(fixed_alloc_df)

                if st.button("Limpar Todas as Alocações Fixas", key="btn_clear_alloc"):
                   st.session_state.fixed_allocations = []
                   st.success("Todas as alocações fixas foram removidas.")
                   st.rerun()

            # Botão para adicionar alocação fixa
            if st.button("Adicionar Alocação Fixa", key="btn_add_alloc"):
                st.session_state.show_add_form = True
                # Inicializar os valores para a nova alocação
                if 'temp_origem_id' not in st.session_state:
                    if st.session_state.origins_df is not None and not st.session_state.origins_df.empty:
                        st.session_state.temp_origem_id = st.session_state.origins_df.index[0]
                if 'temp_destino_id' not in st.session_state:
                    if st.session_state.destinations_df is not None and not st.session_state.destinations_df.empty:
                        st.session_state.temp_destino_id = st.session_state.destinations_df.index[0]
                if 'temp_tipo_material' not in st.session_state:
                    st.session_state.temp_tipo_material = "CFT"
                if 'temp_volume' not in st.session_state:
                    st.session_state.temp_volume = 0.0

            # Formulário para adicionar alocação fixa - DIVIDIDO EM ETAPAS PARA PERMITIR ATUALIZAÇÕES
            if st.session_state.get('show_add_form', False):
                st.write("### Nova Alocação Fixa")
                
                # Lista de origens e destinos
                origens_ids = st.session_state.origins_df.index.tolist()
                destinos_ids = st.session_state.destinations_df.index.tolist()
                
                # Criamos pseudo-formulário com atualizações em tempo real
                col1, col2 = st.columns(2)
                
                # Função para atualizar origem selecionada
                def update_origem():
                    origem_id = st.session_state.origem_selectbox
                    st.session_state.temp_origem_id = origem_id
                    # Atualizar valor sugerido de volume
                    update_volume_sugerido()
                
                # Função para atualizar destino selecionado
                def update_destino():
                    destino_id = st.session_state.destino_selectbox
                    st.session_state.temp_destino_id = destino_id
                    # Atualizar valor sugerido de volume
                    update_volume_sugerido()
                
                # Função para atualizar tipo de material
                def update_tipo():
                    tipo = st.session_state.tipo_material_selectbox
                    st.session_state.temp_tipo_material = tipo
                    # Atualizar valor sugerido de volume
                    update_volume_sugerido()
                    
                # Função para atualizar sugestão de volume
                def update_volume_sugerido():
                    if (st.session_state.temp_origem_id is not None and 
                        st.session_state.temp_destino_id is not None and 
                        st.session_state.origins_df is not None):
                        
                        origem_info = st.session_state.origins_df.loc[st.session_state.temp_origem_id]
                        destino_info = st.session_state.destinations_df.loc[st.session_state.temp_destino_id]
                        
                        # Calcular volume já alocado da origem
                        volume_ja_alocado = sum(
                            float(alloc['volume']) 
                            for alloc in st.session_state.fixed_allocations 
                            if alloc['origem'] == st.session_state.temp_origem_id
                        )
                        
                        vol_disponivel = float(origem_info['Volume disponível (m³)']) - volume_ja_alocado
                        
                        # Volumes necessários e já alocados para este destino
                        vol_cft_alocado = sum(
                            float(alloc['volume']) 
                            for alloc in st.session_state.fixed_allocations 
                            if alloc['destino'] == st.session_state.temp_destino_id and alloc['tipo'] == 'CFT'
                        )
                        vol_ca_alocado = sum(
                            float(alloc['volume']) 
                            for alloc in st.session_state.fixed_allocations 
                            if alloc['destino'] == st.session_state.temp_destino_id and alloc['tipo'] == 'CA'
                        )
                        
                        vol_cft_necessario = float(destino_info['Volume CFT (m³)']) if pd.notna(destino_info['Volume CFT (m³)']) else 0.0
                        vol_ca_necessario = float(destino_info['Volume CA (m³)']) if pd.notna(destino_info['Volume CA (m³)']) else 0.0
                        
                        vol_cft_restante = max(0.0, vol_cft_necessario - vol_cft_alocado)
                        vol_ca_restante = max(0.0, vol_ca_necessario - vol_ca_alocado)
                        
                        if st.session_state.temp_tipo_material == "CFT":
                            st.session_state.temp_volume = min(vol_disponivel, vol_cft_restante)
                        else:  # CA
                            st.session_state.temp_volume = min(vol_disponivel, vol_ca_restante)
                
                # Seletores com callbacks para atualização em tempo real
                with col1:
                    origem_id = st.selectbox(
                        "Origem", 
                        options=origens_ids, 
                        key="origem_selectbox",
                        on_change=update_origem
                    )
                    
                    destino_id = st.selectbox(
                        "Destino", 
                        options=destinos_ids, 
                        key="destino_selectbox",
                        on_change=update_destino
                    )
                
                with col2:
                    tipo_material = st.selectbox(
                        "Tipo de Material", 
                        options=["CFT", "CA"],
                        key="tipo_material_selectbox",
                        on_change=update_tipo
                    )
                    
                    # Entrada do volume
                    volume = st.number_input(
                        "Volume (m³)", 
                        min_value=0.0, 
                        value=float(st.session_state.get('temp_volume', 0.0)),
                        step=10.0,
                        key="volume_input"
                    )
                
                # Exibir informações atualizadas da origem selecionada
                if origem_id is not None:
                    origem_info = st.session_state.origins_df.loc[origem_id]
                    
                    # Calcular volume já alocado da origem
                    volume_ja_alocado = sum(
                        float(alloc['volume']) 
                        for alloc in st.session_state.fixed_allocations 
                        if alloc['origem'] == origem_id
                    )
                    
                    # Calcular volume disponível
                    vol_disponivel = float(origem_info['Volume disponível (m³)']) - volume_ja_alocado
                    
                    # Informações básicas da origem
                    st.info(f"**Origem {origem_id}:** Volume disponível: {float(origem_info['Volume disponível (m³)']):.2f} m³ | Volume restante: {vol_disponivel:.2f} m³ | ISC: {origem_info['ISC']}")

                    if volume_ja_alocado > 0:
                        st.warning(f"Volume já alocado: {volume_ja_alocado:.2f} m³")
                
                # Exibir informações do destino selecionado
                if destino_id is not None:
                    destino_info = st.session_state.destinations_df.loc[destino_id]
                    
                    # Volumes necessários e já alocados para este destino
                    vol_cft_alocado = sum(
                        float(alloc['volume']) 
                        for alloc in st.session_state.fixed_allocations 
                        if alloc['destino'] == destino_id and alloc['tipo'] == 'CFT'
                    )
                    vol_ca_alocado = sum(
                        float(alloc['volume']) 
                        for alloc in st.session_state.fixed_allocations 
                        if alloc['destino'] == destino_id and alloc['tipo'] == 'CA'
                    )
                    
                    vol_cft_necessario = float(destino_info['Volume CFT (m³)']) if pd.notna(destino_info['Volume CFT (m³)']) else 0.0
                    vol_ca_necessario = float(destino_info['Volume CA (m³)']) if pd.notna(destino_info['Volume CA (m³)']) else 0.0
                    
                    vol_cft_restante = max(0.0, vol_cft_necessario - vol_cft_alocado)
                    vol_ca_restante = max(0.0, vol_ca_necessario - vol_ca_alocado)
                    
                    st.info(f"**Destino {destino_id}:** CFT necessário: {vol_cft_necessario:.2f} m³ | CA necessário: {vol_ca_necessario:.2f} m³")
                    st.warning(f"CFT restante: {vol_cft_restante:.2f} m³ | CA restante: {vol_ca_restante:.2f} m³")

                # Verifica compatibilidade de ISC entre origem e destino
                if origem_id is not None and destino_id is not None and 'ISC mínimo exigido' in st.session_state.destinations_df.columns:
                    origem_info = st.session_state.origins_df.loc[origem_id]
                    destino_info = st.session_state.destinations_df.loc[destino_id]
                    
                    isc_origem = float(origem_info['ISC'])
                    isc_min_destino = float(destino_info.get('ISC mínimo exigido', 0)) if pd.notna(destino_info.get('ISC mínimo exigido', None)) else 0.0
                    
                    if isc_origem < isc_min_destino:
                        st.error(f"⚠️ INCOMPATÍVEL: ISC da origem ({isc_origem}) < ISC mínimo exigido ({isc_min_destino})")
                
                # Botões para salvar ou cancelar - FORA do formulário para permitir atualizações em tempo real
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Salvar Alocação", key="btn_save_alloc"):
                        erro = False
                        
                        # Verificações básicas
                        if origem_id is None or destino_id is None:
                            st.error("Selecione origem e destino!")
                            erro = True
                        elif volume <= 0:
                            st.error("O volume deve ser maior que zero!")
                            erro = True
                        else:
                            # Verificações de limites de volume
                            origem_info = st.session_state.origins_df.loc[origem_id]
                            volume_ja_alocado = sum(
                                float(alloc['volume']) 
                                for alloc in st.session_state.fixed_allocations 
                                if alloc['origem'] == origem_id
                            )
                            vol_disponivel = float(origem_info['Volume disponível (m³)']) - volume_ja_alocado
                            
                            if volume > vol_disponivel:
                                st.error(f"Volume excede o disponível! Máximo: {vol_disponivel:.2f} m³")
                                erro = True
                            
                            # Avisos (não impedem o salvamento)
                            if erro == False:
                                destino_info = st.session_state.destinations_df.loc[destino_id]
                                vol_cft_alocado = sum(
                                    float(alloc['volume']) 
                                    for alloc in st.session_state.fixed_allocations 
                                    if alloc['destino'] == destino_id and alloc['tipo'] == 'CFT'
                                )
                                vol_ca_alocado = sum(
                                    float(alloc['volume']) 
                                    for alloc in st.session_state.fixed_allocations 
                                    if alloc['destino'] == destino_id and alloc['tipo'] == 'CA'
                                )
                                
                                vol_cft_necessario = float(destino_info['Volume CFT (m³)']) if pd.notna(destino_info['Volume CFT (m³)']) else 0.0
                                vol_ca_necessario = float(destino_info['Volume CA (m³)']) if pd.notna(destino_info['Volume CA (m³)']) else 0.0
                                
                                vol_cft_restante = max(0.0, vol_cft_necessario - vol_cft_alocado)
                                vol_ca_restante = max(0.0, vol_ca_necessario - vol_ca_alocado)
                                
                                if tipo_material == "CFT" and volume > vol_cft_restante:
                                    st.warning(f"Volume excede CFT necessário! Restante: {vol_cft_restante:.2f} m³")
                                elif tipo_material == "CA" and volume > vol_ca_restante:
                                    st.warning(f"Volume excede CA necessário! Restante: {vol_ca_restante:.2f} m³")
                        
                        if not erro:
                            nova_alocacao = {
                                'origem': origem_id,
                                'destino': destino_id,
                                'tipo': tipo_material,
                                'volume': float(volume)
                            }
                            st.session_state.fixed_allocations.append(nova_alocacao)
                            st.session_state.show_add_form = False  # Fechar o formulário
                            
                            # Limpar variáveis temporárias
                            if 'temp_origem_id' in st.session_state:
                                del st.session_state.temp_origem_id
                            if 'temp_destino_id' in st.session_state:
                                del st.session_state.temp_destino_id
                            if 'temp_tipo_material' in st.session_state:
                                del st.session_state.temp_tipo_material
                            if 'temp_volume' in st.session_state:
                                del st.session_state.temp_volume
                            
                            st.success("Alocação adicionada com sucesso!")
                            st.rerun()
                
                with col2:
                    if st.button("Cancelar", key="btn_cancel"):
                        st.session_state.show_add_form = False
                        
                        # Limpar variáveis temporárias
                        if 'temp_origem_id' in st.session_state:
                            del st.session_state.temp_origem_id
                        if 'temp_destino_id' in st.session_state:
                            del st.session_state.temp_destino_id
                        if 'temp_tipo_material' in st.session_state:
                            del st.session_state.temp_tipo_material
                        if 'temp_volume' in st.session_state:
                            del st.session_state.temp_volume
                            
                        st.rerun()

            # Opção para usar ou não as alocações fixas
            use_fixed_allocations = True
            if st.session_state.fixed_allocations:
                use_fixed_allocations = st.checkbox("Utilizar alocações fixas na otimização", value=True)

                fixed_allocations = st.session_state.fixed_allocations if use_fixed_allocations and st.session_state.fixed_allocations else None
            else:
                fixed_allocations = None
            
            # Salvar e carregar configuração
            st.subheader("Salvar/Carregar Configuração")
            
            col1, col2 = st.columns(2)
            
            with col1:
                config_name = st.text_input("Nome da configuração", value="config_otimizacao")
                
                if st.button("Salvar Configuração", key="btn_save_config"):
                    try:
                        config = {
                            'favor_cortes': favor_cortes,
                            'time_limit': time_limit,
                            'use_max_dist': use_max_dist,
                            'max_dist_cortes': max_dist_cortes if use_max_dist else None,
                            'max_dist_emprestimos_laterais': max_dist_emprestimos_laterais if use_max_dist else None,
                            'max_dist_emprestimos_concentrados': max_dist_emprestimos_concentrados if use_max_dist else None,
                            'usar_emprestimos_bota_fora': usar_emprestimos_bota_fora,
                            'priorizar_bf_proximo': priorizar_bf_proximo if usar_emprestimos_bota_fora else None,
                            'max_dist_bota_fora': max_dist_bota_fora if usar_emprestimos_bota_fora else None,
                            'fixed_allocations': st.session_state.fixed_allocations
                        }
                        
                        # Salvar como JSON
                        with open(f"{config_name}.json", "w") as f:
                            json.dump(config, f, indent=4)
                        
                        st.success(f"Configuração salva com sucesso em '{config_name}.json'")
                    except Exception as e:
                        st.error(f"Erro ao salvar configuração: {str(e)}")
            
            with col2:
                config_file = st.file_uploader("Carregar configuração", type=["json"], key="config_loader")
                
                if config_file is not None:
                    try:
                        config = json.load(config_file)
                        
                        if 'fixed_allocations' in config:
                            st.session_state.fixed_allocations = config['fixed_allocations']
                        
                        st.success("Configuração carregada com sucesso!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao carregar configuração: {str(e)}")
            
            # Verificação de factibilidade
            st.subheader("Verificação de Factibilidade")
            is_feasible, message = check_feasibility(st.session_state.origins_df, st.session_state.destinations_df)
            
            if not is_feasible:
                st.error(f"O problema parece ser infactível: {message}")
                st.warning("Você ainda pode tentar executar a otimização, mas pode não encontrar uma solução válida.")
            
            # Botão para executar a otimização
            if st.button("Executar Otimização", key="btn_execute"):
                cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx = identify_emprestimo_types(
                    st.session_state.origins_df
                )
                st.write("Iniciando otimização... Isso pode levar alguns minutos.")

                # Inicializa barra de progresso
                progress_bar = st.progress(0)
                st.session_state.progress_bar = progress_bar
    
                try:
                    # Adicionar logs de diagnóstico
                    st.write(f"Usar empréstimos como bota-fora: {usar_emprestimos_bota_fora}")
                    
                    # CORREÇÃO: Implementação direta da otimização em duas etapas quando bota-fora em empréstimos está habilitado
                    if usar_emprestimos_bota_fora:
                        st.write("Utilizando otimização em duas etapas para bota-fora em empréstimos...")
                        progress_bar.progress(10)
                        
                        # ETAPA 1: Execute a otimização normal sem bota-fora em empréstimos
                        if has_laterais and has_concentrados:
                            result_step1 = optimize_distribution_advanced(
                                st.session_state.origins_df,
                                st.session_state.destinations_df,
                                time_limit=int(time_limit*0.7),  # 70% do tempo para primeira etapa
                                favor_cortes=favor_cortes,
                                max_dist_cortes=max_dist_cortes if use_max_dist else None,
                                max_dist_emprestimos_laterais=max_dist_emprestimos_laterais if use_max_dist else None,
                                max_dist_emprestimos_concentrados=max_dist_emprestimos_concentrados if use_max_dist else None,
                                fixed_allocations=fixed_allocations,
                                cortes_idx=cortes_idx,
                                emprestimos_laterais_idx=emprestimos_laterais_idx,
                                emprestimos_concentrados_idx=emprestimos_concentrados_idx,
                                usar_emprestimos_bota_fora=False  # Desativado na primeira etapa
                            )
                        else:
                            result_step1 = optimize_distribution(
                                st.session_state.origins_df,
                                st.session_state.destinations_df,
                                time_limit=int(time_limit*0.7),  # 70% do tempo para primeira etapa
                                favor_cortes=favor_cortes,
                                max_dist_cortes=max_dist_cortes if use_max_dist else None,
                                max_dist_emprestimos=max_dist_emprestimos_laterais if use_max_dist else None,
                                fixed_allocations=fixed_allocations,
                                usar_emprestimos_bota_fora=False  # Desativado na primeira etapa
                            )
                        
                        progress_bar.progress(50)
                        
                        # ETAPA 2: Agora otimize a distribuição de bota-fora para empréstimos
                        if result_step1 is not None:
                            # Extrai informações da primeira etapa
                            bota_fora_volumes = result_step1['bota_fora'].copy()
                            emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
                            
                            # Calcula quanto foi retirado de cada empréstimo
                            emprestimos_utilizacao = {}
                            for emp_idx in emprestimos_idx:
                                vol_retirado = result_step1['cft'].loc[emp_idx].sum() + result_step1['ca'].loc[emp_idx].sum()
                                emprestimos_utilizacao[emp_idx] = vol_retirado
                            
                            # Se há material para bota-fora e empréstimos utilizados
                            if bota_fora_volumes.sum() > 0 and sum(emprestimos_utilizacao.values()) > 0:
                                st.write(f"Distribuindo {bota_fora_volumes.sum():.2f} m³ de bota-fora para empréstimos...")
                                
                                # Cria um problema de otimização só para o bota-fora
                                bf_problem = pl.LpProblem("Bota_Fora_Optimization", pl.LpMinimize)
                                
                                # Calcula distâncias entre cortes com bota-fora e empréstimos
                                bf_distances = {}
                                for o_idx, bf_volume in bota_fora_volumes.items():
                                    if bf_volume > 0:
                                        o_row = st.session_state.origins_df.loc[o_idx]
                                        for emp_idx in emprestimos_idx:
                                            if emprestimos_utilizacao.get(emp_idx, 0) > 0:  # Apenas empréstimos utilizados
                                                emp_row = st.session_state.origins_df.loc[emp_idx]
                                                # Calcula distância básica
                                                dist = abs(o_row['Centro de Massa (m)'] - emp_row['Centro de Massa (m)'])
                                                # Adiciona DT fixo do empréstimo, se existir
                                                if 'DT Fixo (m)' in st.session_state.origins_df.columns and pd.notna(st.session_state.origins_df.loc[emp_idx, 'DT Fixo (m)']):
                                                    dist += st.session_state.origins_df.loc[emp_idx, 'DT Fixo (m)']
                                                bf_distances[(o_idx, emp_idx)] = dist
                                
                                # Aplica distância máxima, se definida
                                bf_adjusted_distances = bf_distances.copy()
                                if max_dist_bota_fora is not None:
                                    for key, dist in bf_distances.items():
                                        if dist > max_dist_bota_fora:
                                            bf_adjusted_distances[key] = dist * 1.1  # Penalização
                                
                                # Normaliza distâncias apenas se houver alguma distância
                                if bf_adjusted_distances:
                                    max_dist = max(bf_adjusted_distances.values())
                                    for key in bf_adjusted_distances:
                                        bf_adjusted_distances[key] /= max_dist
                                
                                # Variáveis de decisão: quanto de cada bota-fora vai para cada empréstimo
                                bf_to_emp_vars = {}
                                for o_idx, bf_volume in bota_fora_volumes.items():
                                    if bf_volume > 0:
                                        for emp_idx in emprestimos_idx:
                                            if emprestimos_utilizacao.get(emp_idx, 0) > 0:  # Apenas empréstimos utilizados
                                                bf_to_emp_vars[(o_idx, emp_idx)] = pl.LpVariable(
                                                    f"BF_TO_EMP_{o_idx}_{emp_idx}",
                                                    lowBound=0
                                                )
                                
                                # Variáveis para bota-fora convencional remanescente
                                bf_remaining_vars = {}
                                for o_idx, bf_volume in bota_fora_volumes.items():
                                    if bf_volume > 0:
                                        bf_remaining_vars[o_idx] = pl.LpVariable(
                                            f"BF_REMAIN_{o_idx}",
                                            lowBound=0
                                        )
                                
                                # Se há variáveis de bota-fora para empréstimos
                                if bf_to_emp_vars:
                                    # Função objetivo: minimizar distância/maximizar uso de empréstimos
                                    if priorizar_bf_proximo:
                                        bf_problem += (
                                            10 * pl.lpSum(bf_remaining_vars.values()) +  # Penaliza fortemente o bota-fora convencional
                                            0.1 * pl.lpSum([
                                                bf_adjusted_distances[(o_idx, emp_idx)] * bf_to_emp_vars[(o_idx, emp_idx)]
                                                for (o_idx, emp_idx) in bf_to_emp_vars
                                         ])  
                                        )
                                    else:
                                        # Minimiza o volume enviado para bota-fora convencional
                                        bf_problem += pl.lpSum(bf_remaining_vars.values())
                                    
                                    # Restrição: conservação de volume de bota-fora
                                    for o_idx, bf_volume in bota_fora_volumes.items():
                                        if bf_volume > 0:
                                            bf_problem += (
                                                pl.lpSum([bf_to_emp_vars.get((o_idx, emp_idx), 0) 
                                                       for emp_idx in emprestimos_idx 
                                                       if (o_idx, emp_idx) in bf_to_emp_vars]) + 
                                                bf_remaining_vars[o_idx] == bf_volume,
                                                f"Conservacao_BF_{o_idx}"
                                            )
                                    
                                    # Restrição: capacidade de cada empréstimo para receber bota-fora
                                    for emp_idx in emprestimos_idx:
                                        vol_utilizado = emprestimos_utilizacao.get(emp_idx, 0)
                                        if vol_utilizado > 0:
                                            bf_problem += (
                                                pl.lpSum([bf_to_emp_vars.get((o_idx, emp_idx), 0) 
                                                       for o_idx in bota_fora_volumes.index
                                                       if (o_idx, emp_idx) in bf_to_emp_vars]) <= vol_utilizado,
                                                f"Capacidade_Emprestimo_{emp_idx}"
                                            )
                                    
                                    # Resolver o problema
                                    bf_solver = pl.PULP_CBC_CMD(
                                        msg=True, 
                                        timeLimit=int(time_limit*0.3),  # 30% do tempo para segunda etapa
                                        gapRel=0.01,
                                        options=['presolve on']
                                    )
                                    bf_status = bf_problem.solve(bf_solver)
                                    
                                    # Cria um DataFrame para o resultado de bota-fora em empréstimos
                                    bota_fora_emprestimo = pd.DataFrame(0, index=bota_fora_volumes.index, columns=emprestimos_idx)
                                    
                                    # Extrai resultados
                                    for (o_idx, emp_idx), var in bf_to_emp_vars.items():
                                        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                                            bota_fora_emprestimo.loc[o_idx, emp_idx] = var.value()
                                    
                                    # Atualiza o bota-fora remanescente
                                    bota_fora_atualizado = pd.Series(0, index=bota_fora_volumes.index)
                                    for o_idx, var in bf_remaining_vars.items():
                                        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                                            bota_fora_atualizado[o_idx] = var.value()
                                    
                                    # Calcula momento de bota-fora em empréstimos
                                    momento_bf_emprestimo = sum(
                                        bota_fora_emprestimo.loc[o_idx, emp_idx] * bf_distances.get((o_idx, emp_idx), 0)
                                        for o_idx in bota_fora_emprestimo.index
                                        for emp_idx in bota_fora_emprestimo.columns
                                        if bota_fora_emprestimo.loc[o_idx, emp_idx] > 0
                                    )
                                    
                                    # Volume total de bota-fora em empréstimos
                                    total_bf_emprestimo = bota_fora_emprestimo.sum().sum()
                                    
                                    # Combina os resultados
                                    result = result_step1.copy()
                                    result['bota_fora'] = bota_fora_atualizado
                                    result['bota_fora_emprestimo'] = bota_fora_emprestimo
                                    result['total_bf_emprestimo'] = total_bf_emprestimo
                                    result['usar_emprestimos_bota_fora'] = True
                                    result['priorizar_bf_proximo'] = priorizar_bf_proximo
                                    result['max_dist_bota_fora'] = max_dist_bota_fora
                                    result['momento_bf_emprestimo'] = momento_bf_emprestimo
                                    result['bf_distances'] = bf_distances
                                    
                                    st.write(f"Bota-fora distribuído para empréstimos: {total_bf_emprestimo:.2f} m³")
                                else:
                                    # Se não há variáveis, mantém o resultado da primeira etapa
                                    result = result_step1
                                    result['usar_emprestimos_bota_fora'] = True
                                    result['bota_fora_emprestimo'] = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
                                    result['total_bf_emprestimo'] = 0
                            else:
                                # Se não há material para bota-fora ou não há empréstimos utilizados
                                result = result_step1
                                result['usar_emprestimos_bota_fora'] = True
                                result['bota_fora_emprestimo'] = pd.DataFrame(0, index=cortes_idx, columns=emprestimos_idx)
                                result['total_bf_emprestimo'] = 0
                        else:
                            result = None  # Se a primeira etapa falhou
                        
                        progress_bar.progress(100)
                    else:
                        # Determinar qual função de otimização usar (abordagem original)
                        if has_laterais and has_concentrados:
                           # Versão avançada com tipos específicos de empréstimos
                           result = optimize_distribution_advanced(
                               st.session_state.origins_df,
                               st.session_state.destinations_df,
                               time_limit=time_limit,
                               favor_cortes=favor_cortes,
                               max_dist_cortes=max_dist_cortes if use_max_dist else None,
                               max_dist_emprestimos_laterais=max_dist_emprestimos_laterais if use_max_dist else None,
                               max_dist_emprestimos_concentrados=max_dist_emprestimos_concentrados if use_max_dist else None,
                               fixed_allocations=fixed_allocations,
                               cortes_idx=cortes_idx,
                               emprestimos_laterais_idx=emprestimos_laterais_idx,
                               emprestimos_concentrados_idx=emprestimos_concentrados_idx,
                               usar_emprestimos_bota_fora=False
                            )
                        else:
                            # Versão padrão
                            result = optimize_distribution(
                            st.session_state.origins_df,
                            st.session_state.destinations_df,
                            time_limit=time_limit,
                            favor_cortes=favor_cortes,
                            max_dist_cortes=max_dist_cortes if use_max_dist else None,
                            max_dist_emprestimos=max_dist_emprestimos_laterais if use_max_dist else None,
                            fixed_allocations=fixed_allocations,
                            usar_emprestimos_bota_fora=False
                            )
                        # Atualiza barra de progresso para 80%
                        progress_bar.progress(80)
                        progress_bar.progress(100)

                    # Armazena o resultado
                    st.session_state.optimization_result = result
        
                    if result:
                        st.success("Otimização concluída com sucesso!")
                        
                        # Exibir resumo dos resultados
                        st.subheader("Resumo da Otimização")
            
                        summary = generate_distribution_summary(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        st.text(summary)

                        # Substitua o código do botão por:
                        st.info("Para ver os resultados detalhados, selecione 'Visualizar Resultados' no menu lateral.")
 
                    else:
                        st.error("A otimização não conseguiu encontrar uma solução factível!")
                        st.write("Tente ajustar os parâmetros ou verificar os dados de entrada.")
                
                except Exception as e:
                    st.error(f"Erro durante a otimização: {str(e)}")
                    st.write("Detalhes do erro:")
                    st.write(str(e))
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Carregue os dados das origens e destinos antes de executar a otimização.")
    
    # ABA: VISUALIZAR RESULTADOS
    elif tab == "Visualizar Resultados":
        st.header("Visualização dos Resultados da Otimização")
        
        if 'optimization_result' in st.session_state and st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            
            # Exibir status e métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", result['status'])
            with col2:
                st.metric("DMT (m)", f"{result['dmt']:.2f}")
            with col3:
                momento_total_km = result['momento_total'] / 1000  # Converter para km·m³
                st.metric("Momento Total (km·m³)", f"{momento_total_km:.2f}")
            
            # Se houver bota-fora em empréstimos, exibir métricas adicionais
            if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    bf_convencional = result['bota_fora'].sum()
                    st.metric("Bota-Fora Convencional (m³)", f"{bf_convencional:.2f}")
                with col2:
                    bf_emprestimo = result.get('total_bf_emprestimo', 0)
                    st.metric("Bota-Fora em Empréstimos (m³)", f"{bf_emprestimo:.2f}")
                with col3:
                    bf_total = bf_convencional + bf_emprestimo
                    st.metric("Total Bota-Fora (m³)", f"{bf_total:.2f}")
            
            # Tabs para diferentes visualizações
            result_tabs = st.tabs(["Resumo", "Gráficos", "Detalhes CFT", "Detalhes CA", "Bota-fora"])
            
            # Tab de Resumo
            with result_tabs[0]:
                st.subheader("Resumo da Distribuição")
                
                summary = generate_distribution_summary(
                    result, 
                    st.session_state.origins_df, 
                    st.session_state.destinations_df
                )
                
                st.text(summary)
                
                # Informações sobre os parâmetros utilizados
                st.subheader("Parâmetros Utilizados")
                
                params_text = [
                    f"Favorecimento de cortes: {'Sim' if result.get('favor_cortes') else 'Não'}"
                ]
                
                if result.get('max_dist_cortes') is not None:
                    params_text.append(f"Distância máxima para cortes: {result['max_dist_cortes']:.0f} m")
                
                if 'max_dist_emprestimos_laterais' in result and result['max_dist_emprestimos_laterais'] is not None:
                    params_text.append(f"Distância máxima para empréstimos laterais: {result['max_dist_emprestimos_laterais']:.0f} m")
                
                if 'max_dist_emprestimos_concentrados' in result and result['max_dist_emprestimos_concentrados'] is not None:
                    params_text.append(f"Distância máxima para empréstimos concentrados: {result['max_dist_emprestimos_concentrados']:.0f} m")
                
                if 'max_dist_emprestimos' in result and result['max_dist_emprestimos'] is not None:
                    params_text.append(f"Distância máxima para empréstimos: {result['max_dist_emprestimos']:.0f} m")
                
                # Parâmetros de bota-fora em empréstimos
                if 'usar_emprestimos_bota_fora' in result:
                    params_text.append(f"Usar empréstimos como bota-fora: {'Sim' if result['usar_emprestimos_bota_fora'] else 'Não'}")
                    
                    if result.get('usar_emprestimos_bota_fora'):
                        params_text.append(f"Priorizar bota-fora mais próximo: {'Sim' if result.get('priorizar_bf_proximo', True) else 'Não'}")
                        
                        if 'max_dist_bota_fora' in result and result['max_dist_bota_fora'] is not None:
                            params_text.append(f"Distância máxima para bota-fora em empréstimos: {result['max_dist_bota_fora']:.0f} m")
                
                if 'fixed_allocations' in result and result['fixed_allocations']:
                    params_text.append(f"Alocações fixas utilizadas: {len(result['fixed_allocations'])}")
                
                for param in params_text:
                    st.write(param)
            
            # Tab de Gráficos
            with result_tabs[1]:
                # Utilizar a função de gráficos criada anteriormente
                display_optimization_charts(
                    result,
                    st.session_state.origins_df,
                    st.session_state.destinations_df
                )
            
            # Tab de Detalhes CFT
            with result_tabs[2]:
                st.subheader("Distribuição Detalhada de CFT")
                
                # Mostrar a matriz de distribuição
                st.write("Matriz de distribuição CFT (m³):")
                st.dataframe(result['cft'])
                
                # Informação adicional sobre CFT
                total_cft_necessario = st.session_state.destinations_df['Volume CFT (m³)'].fillna(0).sum()
                total_cft_distribuido = result['cft'].sum().sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Volume CFT Necessário", f"{total_cft_necessario:.2f} m³")
                col2.metric("Volume CFT Distribuído", f"{total_cft_distribuido:.2f} m³")
                col3.metric("Atendimento", f"{(total_cft_distribuido/total_cft_necessario*100):.1f}%" if total_cft_necessario > 0 else "N/A")
                
                # Verificar se há destinos não atendidos
                remaining_cft = result['remaining_cft']
                if remaining_cft.sum() > 0:
                    st.warning("Alguns destinos não foram totalmente atendidos para CFT!")
                    # Listar destinos não atendidos
                    st.write("Destinos com déficit de CFT:")
                    
                    deficit_df = pd.DataFrame({
                        'Destino': remaining_cft.index[remaining_cft > 0],
                        'Volume Faltante (m³)': remaining_cft[remaining_cft > 0].values
                    })
                    
                    st.dataframe(deficit_df)
            
            # Tab de Detalhes CA
            with result_tabs[3]:
                st.subheader("Distribuição Detalhada de CA")
                
                # Mostrar a matriz de distribuição
                st.write("Matriz de distribuição CA (m³):")
                st.dataframe(result['ca'])
                
                # Informação adicional sobre CA
                total_ca_necessario = st.session_state.destinations_df['Volume CA (m³)'].fillna(0).sum()
                total_ca_distribuido = result['ca'].sum().sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Volume CA Necessário", f"{total_ca_necessario:.2f} m³")
                col2.metric("Volume CA Distribuído", f"{total_ca_distribuido:.2f} m³")
                col3.metric("Atendimento", f"{(total_ca_distribuido/total_ca_necessario*100):.1f}%" if total_ca_necessario > 0 else "N/A")
                
                # Verificar se há destinos não atendidos
                remaining_ca = result['remaining_ca']
                if remaining_ca.sum() > 0:
                    st.warning("Alguns destinos não foram totalmente atendidos para CA!")
                    
                    # Listar destinos não atendidos
                    st.write("Destinos com déficit de CA:")
                    
                    deficit_df = pd.DataFrame({
                        'Destino': remaining_ca.index[remaining_ca > 0],
                        'Volume Faltante (m³)': remaining_ca[remaining_ca > 0].values
                    })
                    
                    st.dataframe(deficit_df)
            
            # Tab de Bota-fora e material não utilizado
            with result_tabs[4]:
                st.subheader("Material para Bota-fora")
                
                                # Material para bota-fora convencional
                bota_fora = result['bota_fora']
                if bota_fora.sum() > 0:
                    st.write("Volumes direcionados para bota-fora convencional:")
                    
                    bf_df = pd.DataFrame({
                        'Origem': bota_fora.index[bota_fora > 0],
                        'Volume (m³)': bota_fora[bota_fora > 0].values
                    })
                    
                    st.dataframe(bf_df)
                    
                    # Gráfico de bota-fora
                    if len(bf_df) > 0:
                        st.bar_chart(bf_df, x='Origem', y='Volume (m³)')
                else:
                    st.success("Não há material direcionado para bota-fora convencional.")
                
                # Material para bota-fora em empréstimos, se aplicável
                if 'usar_emprestimos_bota_fora' in result and result['usar_emprestimos_bota_fora'] and 'bota_fora_emprestimo' in result:
                    st.subheader("Material para Bota-fora em Empréstimos")
                    
                    total_bf_emp = result['bota_fora_emprestimo'].sum().sum()
                    
                    if total_bf_emp > 0:
                        # Cria DataFrame para visualização
                        bf_emp_data = []
                        for o_idx in result['bota_fora_emprestimo'].index:
                            for emp_idx in result['bota_fora_emprestimo'].columns:
                                if result['bota_fora_emprestimo'].loc[o_idx, emp_idx] > 0:
                                    bf_emp_data.append({
                                        'Origem (Corte)': o_idx,
                                        'Destino (Empréstimo)': emp_idx,
                                        'Volume (m³)': result['bota_fora_emprestimo'].loc[o_idx, emp_idx]
                                    })
                        
                        bf_emp_df = pd.DataFrame(bf_emp_data)
                        st.dataframe(bf_emp_df)
                        
                        # Gráfico por empréstimo receptor
                        emps = []
                        volumes = []
                        for emp_idx in result['bota_fora_emprestimo'].columns:
                            vol = result['bota_fora_emprestimo'][emp_idx].sum()
                            if vol > 0:
                                emps.append(f"Empréstimo {emp_idx}")
                                volumes.append(vol)
                        
                        if emps:
                            emp_chart_df = pd.DataFrame({
                                'Empréstimo': emps,
                                'Volume Recebido (m³)': volumes
                            })
                            
                            st.write("Volume recebido por empréstimo:")
                            st.bar_chart(emp_chart_df, x='Empréstimo', y='Volume Recebido (m³)')
                    else:
                        st.success("Não há material direcionado para bota-fora em empréstimos.")
                
                # Material de empréstimo não utilizado
                st.subheader("Material de Empréstimo Não Utilizado")
                
                # Verifica qual tipo de resultado temos (avançado ou simples)
                if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
                    emp_lat = result['emprestimos_laterais_nao_utilizados']
                    emp_conc = result['emprestimos_concentrados_nao_utilizados']
                    
                    # Criar DataFrames para cada tipo
                    emp_lat_df = pd.DataFrame({
                        'Origem': list(emp_lat.keys()),
                        'Volume (m³)': list(emp_lat.values()),
                        'Tipo': ['Lateral'] * len(emp_lat)
                    }) if emp_lat else pd.DataFrame()
                    
                    emp_conc_df = pd.DataFrame({
                        'Origem': list(emp_conc.keys()),
                        'Volume (m³)': list(emp_conc.values()),
                        'Tipo': ['Concentrado'] * len(emp_conc)
                    }) if emp_conc else pd.DataFrame()
                    
                    # Combinar os DataFrames
                    emp_df = pd.concat([emp_lat_df, emp_conc_df])
                    
                    if not emp_df.empty:
                        st.write("Volumes de empréstimo não utilizados:")
                        st.dataframe(emp_df)
                        
                        # Gráfico de empréstimos não utilizados
                        st.bar_chart(emp_df, x='Origem', y='Volume (m³)', color='Tipo')
                    else:
                        st.success("Todo o material de empréstimo foi utilizado.")
                    
                elif 'emprestimos_nao_utilizados' in result:
                    emp = result['emprestimos_nao_utilizados']
                    
                    if emp:
                        emp_df = pd.DataFrame({
                            'Origem': list(emp.keys()),
                            'Volume (m³)': list(emp.values())
                        })
                        
                        st.write("Volumes de empréstimo não utilizados:")
                        st.dataframe(emp_df)
                        
                        # Gráfico de empréstimos não utilizados
                        st.bar_chart(emp_df, x='Origem', y='Volume (m³)')
                    else:
                        st.success("Todo o material de empréstimo foi utilizado.")
                else:
                    st.info("Não há informações sobre material de empréstimo não utilizado.")
        else:
            st.warning("Execute a otimização antes de visualizar os resultados.")
    
    # ABA: EXPORTAR
    elif tab == "Exportar":
        st.header("Exportar Resultados da Otimização")
        
        if 'optimization_result' in st.session_state and st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            
            # Resumo da otimização
            st.subheader("Resumo da Otimização")
            
            summary = generate_distribution_summary(
                result, 
                st.session_state.origins_df, 
                st.session_state.destinations_df
            )
            
            st.text(summary)
            
            # Opções de exportação
            st.subheader("Exportar Relatório")
            
            # Escolha do formato
            export_format = st.radio("Escolha o formato de exportação:", 
                                   ["Excel (.xlsx)", "JSON (.json)"])
            
            # Nome do arquivo
            default_filename = f"distribuicao_terraplenagem_{uuid.uuid4().hex[:8]}"
            filename = st.text_input("Nome do arquivo (sem extensão):", value=default_filename)
            
            if st.button("Gerar e Baixar Relatório", key="btn_generate_report"):
                try:
                    if export_format == "Excel (.xlsx)":
                        # Gera relatório Excel
                        excel_file = create_distribution_report(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        
                        # Disponibiliza para download
                        st.download_button(
                            label="Baixar Relatório Excel",
                            data=excel_file,
                            file_name=f"{filename}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="btn_download_excel"
                        )
                        
                        st.success("Relatório Excel gerado com sucesso!")
                    
                    elif export_format == "JSON (.json)":
                        # Gera relatório JSON
                        json_data = export_optimization_results(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        
                        # Disponibiliza para download
                        st.download_button(
                            label="Baixar Relatório JSON",
                            data=json_data,
                            file_name=f"{filename}.json",
                            mime="application/json",
                            key="btn_download_json"
                        )
                        
                        st.success("Relatório JSON gerado com sucesso!")
                
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Relatório por intervalos de distância
            st.subheader("Relatório por Intervalos de Distância de Transporte")
            st.write("""
            Este relatório divide o volume do material distribuído em intervalos específicos de distância,
            conforme categorias padronizadas de serviços de escavação, carga e transporte.
            """)
            # Interface para definição de custos unitários
            st.subheader("Definição de Custos Unitários")
            st.write("Defina o custo unitário (R$/m³) para cada intervalo de distância:")

            # Definimos os intervalos - deve corresponder aos mesmos intervalos da função create_distance_interval_report
            intervals = [
                (0, 50, "ESCAVAÇÃO, CARGA E TRANSPORTE DE MATERIAL DE 1ª CATEGORIA ATÉ 50M"),
                (51, 200, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 51 A 200M)"),
                (201, 400, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 201 A 400M)"),
                (401, 600, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 401 A 600M)"),
                (601, 800, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 601 A 800M)"),
                (801, 1000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 801 A 1.000M)"),
                (1001, 1200, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.001 A 1.200M)"),
                (1201, 1400, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.201 A 1.400M)"),
                (1401, 1600, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.401 A 1.600M)"),
                (1601, 1800, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.601 A 1.800M)"),
                (1801, 2000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 1.801 A 2.000M)"),
                (2001, 3000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 2.001 A 3.000M)"),
                (3001, 5000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 3.001 A 5.000M)"),
                (5001, 10000, "ESCAV., CARGA E TRANSPORTE DE MAT. 1ª CATEG. - C/ ESCAVADEIRA - (DT: 5.001 A 10.000M)"),
                (10001, float('inf'), "ESCAV. E CARGA 1ª CATEG - DMT>10,0KM")
            ]

            # Adicione uma categoria para material sem transporte
            sem_transporte = "ESCAV. E CARGA 1ª CATEG. - SEM TRANSPORTE"

            # Inicializa os custos na session_state, se ainda não existirem
            if 'cost_per_interval' not in st.session_state:
                # Valores padrão - você pode ajustar conforme necessário
                st.session_state.cost_per_interval = {
                   sem_transporte: 5.0,  # Valor padrão para sem transporte
                }
    
                # Valores padrão que aumentam com a distância
                base_cost = 5.0
                for i, (min_dist, max_dist, desc) in enumerate(intervals):
                    # Custo aumenta gradativamente com a distância
                    st.session_state.cost_per_interval[desc] = base_cost + (i * 0.5)

            # Interface para editar os custos
            col1, col2 = st.columns(2)

            with col1:
                # Primeira metade dos intervalos
                for i, (min_dist, max_dist, desc) in enumerate(intervals[:8]):
                    # Usando a descrição resumida na interface para economizar espaço
                    short_desc = f"DT: {min_dist} a {'∞' if max_dist == float('inf') else max_dist}m"
                    st.session_state.cost_per_interval[desc] = st.number_input(
                       f"{short_desc}", 
                       min_value=0.0, 
                       value=st.session_state.cost_per_interval.get(desc, 5.0 + (i * 0.5)),
                       step=0.01,
                       format="%.2f",
                       key=f"cost_{i}"
                    )

                with col2:
                  # Segunda metade dos intervalos
                  for i, (min_dist, max_dist, desc) in enumerate(intervals[8:]):
                      i = i + 8  # Continua a numeração
                      short_desc = f"DT: {min_dist} a {'∞' if max_dist == float('inf') else max_dist}m"
                      st.session_state.cost_per_interval[desc] = st.number_input(
                        f"{short_desc}", 
                        min_value=0.0, 
                        value=st.session_state.cost_per_interval.get(desc, 5.0 + (i * 0.5)),
                        step=0.01,
                        format="%.2f",
                        key=f"cost_{i}"
                     )

                # Item sem transporte
                st.session_state.cost_per_interval[sem_transporte] = st.number_input(
                "Material sem transporte", 
                min_value=0.0, 
                value=st.session_state.cost_per_interval.get(sem_transporte, 5.0),
                step=0.01,
                format="%.2f",
                key=f"cost_sem_transporte"
                )

                # Botão para resetar para valores padrão
                if st.button("Resetar para valores padrão", key="btn_reset_costs"):
                    # Valores padrão - você pode ajustar conforme necessário
                    st.session_state.cost_per_interval = {
                        sem_transporte: 5.0,  # Valor padrão para sem transporte
                    }

                    # Valores padrão que aumentam com a distância
                    base_cost = 5.0
                    for i, (min_dist, max_dist, desc) in enumerate(intervals):
                        # Custo aumenta gradativamente com a distância
                        st.session_state.cost_per_interval[desc] = base_cost + (i * 0.5)

                    st.success("Custos resetados para valores padrão!")
                    st.rerun()

            if st.button("Gerar Relatório por Intervalos", key="btn_generate_interval_report"):
               try:
                # Gera o relatório por intervalos, passando os custos unitários
                report_df, excel_file = create_distance_interval_report(
                    result, 
                    st.session_state.origins_df, 
                    st.session_state.destinations_df,
                    cost_per_interval=st.session_state.cost_per_interval
                )

                if report_df is not None:
                    # Exibe o relatório
                    st.write("### Resultado do Relatório por Intervalos")
                    st.dataframe(report_df)
                
                    # Disponibiliza para download
                    st.download_button(
                        label="Baixar Relatório por Intervalos (Excel)",
                        data=excel_file,
                        file_name=f"{filename}_relatorio_intervalos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="btn_download_interval_excel"
                    )
                
                    # Opção para download como CSV
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="Baixar Relatório por Intervalos (CSV)",
                        data=csv,
                        file_name=f"{filename}_relatorio_intervalos.csv",
                        mime="text/csv",
                        key="btn_download_interval_csv"
                    )
                
                    st.success("Relatório por intervalos gerado com sucesso!")
                else:
                    st.error("Não foi possível gerar o relatório por intervalos.")
               except Exception as e:
                st.error(f"Erro ao gerar relatório por intervalos: {str(e)}")
                import traceback
                st.code(traceback.format_exc()) 
   
    else:
        st.warning("Execute a otimização antes de exportar os resultados.")
            

def main():
    """
    Função principal para executar o aplicativo Streamlit
    """
    create_interface()

if __name__ == "__main__":
    main()
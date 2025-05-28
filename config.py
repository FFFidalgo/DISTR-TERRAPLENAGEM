"""
Configurações globais da aplicação de otimização de terraplenagem
"""

# Configurações de otimização
OPTIMIZATION_CONFIG = {
    'default_time_limit': 1800,  # 30 minutos
    'max_iterations': 10000,
    'tolerance': 1e-6,
    'solver': 'COIN_CMD'  # Solver padrão do PuLP
}

# Configurações de interface
UI_CONFIG = {
    'page_title': "Otimização de Distribuição de Terraplenagem",
    'layout': "wide",
    'sidebar_width': 300,
    'max_file_size_mb': 50
}

# Configurações de dados
DATA_CONFIG = {
    'required_origin_columns': [
        "Tipo", 
        "Centro de Massa (m)", 
        "Volume disponível (m³)", 
        "ISC"
    ],
    'optional_origin_columns': [
        "DT Fixo (m)"
    ],
    'required_destination_columns': [
        "Centro de Massa (m)",
        "ISC mínimo exigido"
    ],
    'optional_destination_columns': [
        "Volume CFT (m³)",
        "Volume CA (m³)"
    ],
    'supported_file_types': ["xlsx", "xls", "csv"],
    'material_types': {
        'CFT': 'Corte, Fundo de Trincheira',
        'CA': 'Corpo do Aterro'
    }
}

# Configurações de validação
VALIDATION_CONFIG = {
    'min_volume': 0.01,  # Volume mínimo em m³
    'max_distance': 10000,  # Distância máxima em metros
    'min_isc': 0,
    'max_isc': 100
}

# Configurações de export
EXPORT_CONFIG = {
    'default_filename': 'resultado_otimizacao',
    'supported_formats': ['xlsx', 'csv'],
    'include_summary': True,
    'include_details': True
}

# Mensagens de erro padrão
ERROR_MESSAGES = {
    'file_upload_error': "Erro ao carregar o arquivo. Verifique o formato e tente novamente.",
    'missing_columns': "Colunas obrigatórias não encontradas: {columns}",
    'invalid_data': "Dados inválidos encontrados na linha {row}: {error}",
    'optimization_failed': "A otimização falhou: {reason}",
    'no_feasible_solution': "Não foi possível encontrar uma solução factível para o problema.",
    'insufficient_volume': "Volume insuficiente disponível nas origens.",
    'invalid_isc': "Valores de ISC inválidos encontrados."
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_optimization_steps': True,
    'log_data_validation': True
}

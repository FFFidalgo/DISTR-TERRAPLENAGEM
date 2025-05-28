# 🏗️ Sistema de Otimização de Distribuição de Terraplenagem

## 📋 Visão Geral

Sistema avançado de otimização para distribuição de materiais em projetos de terraplenagem, utilizando algoritmos de programação linear para encontrar a distribuição ótima entre origens (cortes e empréstimos) e destinos (aterros).

## ✨ Funcionalidades Principais

### 🔧 Funcionalidades Técnicas
- **Algoritmo de Otimização**: Programação linear usando PuLP
- **Validação de Dados**: Verificação automática de integridade e factibilidade
- **Cálculo de Distâncias**: Considera DT fixo para empréstimos
- **Restrições de ISC**: Atende requisitos de California Bearing Ratio
- **Múltiplos Tipos de Material**: CFT (Corte, Fundo de Trincheira) e CA (Corpo do Aterro)

### 🎨 Interface de Usuário
- **Interface Streamlit Moderna**: Design responsivo e intuitivo
- **Upload de Arquivos**: Suporte para Excel (.xlsx, .xls) e CSV
- **Entrada Manual de Dados**: Formulários interativos
- **Visualizações Avançadas**: Gráficos interativos com Plotly
- **Exportação Completa**: Resultados em múltiplos formatos

### 📊 Análises e Relatórios
- **Dashboard Executivo**: Métricas principais em tempo real
- **Análise de Utilização**: Origens e destinos
- **Visualização de Rede**: Diagrama de transporte
- **Análise What-If**: Cenários interativos
- **Relatórios Detalhados**: Exportação completa dos resultados

## 🚀 Melhorias Implementadas

### 1. **Arquitetura Modular**
- ✅ Separação de responsabilidades em módulos especializados
- ✅ Código organizado em pacotes: `optimization/`, `ui/`, `utils/`
- ✅ Configurações centralizadas em `config.py`
- ✅ Facilita manutenção e extensibilidade

### 2. **Performance e Robustez**
- ✅ Otimização de algoritmos de programação linear
- ✅ Validação rigorosa de dados de entrada
- ✅ Tratamento de erros abrangente
- ✅ Caching para operações custosas
- ✅ Indicadores de progresso para operações longas

### 3. **Interface de Usuário Aprimorada**
- ✅ Design moderno e responsivo
- ✅ Navegação intuitiva com abas e seções
- ✅ Feedback visual melhorado
- ✅ Formulários interativos para entrada de dados
- ✅ Visualizações interativas e informativas

### 4. **Funcionalidades Avançadas**
- ✅ Múltiplos formatos de entrada de dados
- ✅ Análise de sensibilidade (What-If)
- ✅ Exportação completa de resultados
- ✅ Histórico de otimizações
- ✅ Métricas de performance detalhadas

### 5. **Qualidade de Código**
- ✅ Documentação abrangente com docstrings
- ✅ Testes automatizados (pytest)
- ✅ Type hints para melhor manutenibilidade
- ✅ Logging estruturado
- ✅ Padrões de código consistentes

## 📁 Estrutura do Projeto

```
/app/
├── main.py                     # Aplicação principal (nova versão)
├── app2.py                     # Aplicação original (mantida para referência)
├── config.py                   # Configurações globais
├── requirements.txt            # Dependências atualizadas
├── test_application.py         # Testes automatizados
│
├── optimization/               # Módulo de otimização
│   ├── __init__.py
│   ├── optimizer.py           # Otimizador principal
│   └── distance_calculator.py # Calculadora de distâncias
│
├── utils/                      # Utilitários
│   ├── __init__.py
│   └── data_validator.py      # Validação e limpeza de dados
│
└── ui/                         # Interface de usuário
    ├── __init__.py
    ├── data_input.py          # Entrada de dados
    ├── results_display.py     # Exibição de resultados
    └── visualization.py       # Visualizações
```

## 🛠️ Instalação e Execução

### 1. **Instalar Dependências**
```bash
cd /app
pip install -r requirements.txt
```

### 2. **Executar a Aplicação**
```bash
# Nova versão otimizada
streamlit run main.py

# Ou a versão original
streamlit run app2.py
```

### 3. **Executar Testes**
```bash
python test_application.py
```

## 📖 Guia de Uso

### 1. **Entrada de Dados**
- **Upload de Arquivos**: Carregue arquivos Excel ou CSV com dados de origens e destinos
- **Entrada Manual**: Use formulários interativos para criar dados de exemplo
- **Dados de Exemplo**: Carregue exemplos pré-configurados

### 2. **Validação**
- Sistema verifica automaticamente a integridade dos dados
- Identifica erros e inconsistências
- Confirma factibilidade do problema

### 3. **Otimização**
- Configure parâmetros avançados (limites de distância, preferências)
- Execute otimização com feedback em tempo real
- Acompanhe métricas de performance

### 4. **Análise de Resultados**
- Dashboard executivo com métricas principais
- Tabelas detalhadas de alocações
- Análise de utilização de origens e atendimento de destinos

### 5. **Visualização**
- Gráficos interativos de distribuição
- Diagrama de rede de transporte
- Análise de custos e eficiência

### 6. **Exportação**
- Exportação em Excel com múltiplas abas
- Formato CSV para análises externas
- Relatórios completos incluindo dados originais

## 🔧 Configurações Avançadas

### Parâmetros de Otimização
- **Tempo Limite**: Controle o tempo máximo de otimização
- **Preferência por Cortes**: Priorize cortes sobre empréstimos
- **Limites de Distância**: Restrinja distâncias máximas por tipo de origem
- **Solver**: Configure o solver de programação linear

### Formatos de Dados Suportados

#### Origens (Obrigatórias)
- `Tipo`: Tipo da origem (Corte, Empréstimo Lateral, Empréstimo Concentrado)
- `Centro de Massa (m)`: Posição da origem
- `Volume disponível (m³)`: Volume disponível para transporte
- `ISC`: California Bearing Ratio

#### Origens (Opcionais)
- `DT Fixo (m)`: Distância de transporte fixa para empréstimos

#### Destinos (Obrigatórias)
- `Centro de Massa (m)`: Posição do destino
- `ISC mínimo exigido`: ISC mínimo aceito

#### Destinos (Opcionais)
- `Volume CFT (m³)`: Volume necessário de CFT
- `Volume CA (m³)`: Volume necessário de CA

## 🧪 Testes e Qualidade

### Testes Automatizados
- **Testes Unitários**: Validação, cálculos, otimização
- **Testes de Integração**: Pipeline completo
- **Testes de Performance**: Verificação de tempos de execução

### Métricas de Qualidade
- Cobertura de código com testes
- Documentação abrangente
- Type hints para todas as funções principais
- Logging estruturado para debugging

## 🔍 Algoritmo de Otimização

### Função Objetivo
Minimizar a soma ponderada das distâncias de transporte:
```
min Σ(i,j) (distância_ij × volume_ij × peso_i)
```

### Restrições Principais
1. **Capacidade das Origens**: Volume transportado ≤ Volume disponível
2. **Demanda dos Destinos**: Volume recebido ≥ Volume necessário
3. **Restrições de ISC**: ISC da origem ≥ ISC mínimo do destino (para CFT)
4. **Limites de Distância**: Distância ≤ Limite máximo por tipo

### Tipos de Material
- **CFT**: Requer ISC mínimo, usado em fundações e trincheiras
- **CA**: Sem restrição de ISC, usado no corpo do aterro

## 📈 Exemplos de Uso

### Exemplo 1: Projeto Simples
- 3 origens (1 corte, 2 empréstimos)
- 3 destinos com necessidades de CFT e CA
- Otimização básica minimizando distâncias

### Exemplo 2: Projeto Complexo
- 6 origens de diferentes tipos
- 6 destinos com restrições variadas de ISC
- Parâmetros avançados com limites de distância

## 🚨 Troubleshooting

### Problemas Comuns
1. **Dados Inválidos**: Verifique formato e colunas obrigatórias
2. **Problema Infactível**: Verifique se há volume suficiente e ISC adequado
3. **Otimização Lenta**: Reduza o tempo limite ou simplifique o problema
4. **Erro de Importação**: Instale todas as dependências

### Logs e Debug
- Logs estruturados disponíveis no console
- Modo debug mostra detalhes técnicos de erros
- Histórico de otimizações para análise de performance

## 🤝 Contribuição

Para contribuir com melhorias:
1. Identifique áreas de melhoria
2. Implemente testes para novas funcionalidades
3. Mantenha documentação atualizada
4. Siga padrões de código estabelecidos

## 📄 Licença

Este projeto utiliza a licença MIT (ver arquivo LICENSE).

---

**Desenvolvido com ❤️ para otimizar projetos de terraplenagem**
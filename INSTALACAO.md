# 🛠️ Guia de Instalação - Sistema de Otimização de Terraplenagem

## ❌ Problema: ModuleNotFoundError: No module named 'pulp'

Este erro indica que as dependências necessárias não estão instaladas em seu ambiente Python.

## 🔧 Soluções Rápidas

### Solução 1: Instalação Automática (Recomendada)

Execute o script de instalação automática:

```bash
python install_dependencies.py
```

### Solução 2: Instalação Manual via requirements.txt

```bash
pip install -r requirements.txt
```

### Solução 3: Instalação Manual Individual

```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install streamlit>=1.28.0
pip install pulp>=2.7.0
pip install openpyxl>=3.1.0
pip install xlsxwriter>=3.1.0
pip install plotly>=5.15.0
pip install pytest>=7.4.0
```

### Solução 4: Instalação com Upgrade (se já tem algumas dependências)

```bash
pip install --upgrade pandas numpy streamlit pulp openpyxl xlsxwriter plotly pytest
```

## 🐍 Problemas com Ambiente Python

### Se você está usando Anaconda/Miniconda:

```bash
conda install pandas numpy streamlit openpyxl xlsxwriter plotly pytest
pip install pulp  # PuLP não está disponível no conda
```

### Se você está usando um ambiente virtual:

```bash
# Ativar ambiente virtual primeiro
# Windows:
venv\Scripts\activate
# ou Linux/Mac:
source venv/bin/activate

# Depois instalar dependências
pip install -r requirements.txt
```

### Criar novo ambiente virtual (se necessário):

```bash
# Criar ambiente virtual
python -m venv terraplenagem_env

# Ativar ambiente
# Windows:
terraplenagem_env\Scripts\activate
# Linux/Mac:
source terraplenagem_env/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## ✅ Verificação da Instalação

Após instalar as dependências, teste se tudo está funcionando:

```bash
python -c "import pandas, numpy, streamlit, pulp, openpyxl, xlsxwriter, plotly; print('✅ Todas as dependências instaladas!')"
```

## 🚀 Executar a Aplicação

Depois que todas as dependências estiverem instaladas:

```bash
streamlit run main.py
```

## 🔍 Diagnóstico de Problemas

Se ainda há problemas, execute o diagnóstico:

```bash
python install_dependencies.py
```

Este script irá:
- ✅ Verificar quais dependências estão instaladas
- ❌ Identificar quais estão faltando
- 🔄 Tentar instalar automaticamente as faltantes
- 📊 Gerar relatório final

## 📋 Dependências Necessárias

| Pacote | Versão Mínima | Descrição |
|--------|---------------|-----------|
| pandas | 2.0.0 | Manipulação de dados |
| numpy | 1.24.0 | Computação numérica |
| streamlit | 1.28.0 | Interface web |
| pulp | 2.7.0 | Otimização linear |
| openpyxl | 3.1.0 | Leitura de Excel |
| xlsxwriter | 3.1.0 | Escrita de Excel |
| plotly | 5.15.0 | Visualizações interativas |
| pytest | 7.4.0 | Testes automatizados |

## 🆘 Se Nada Funcionar

1. **Verifique sua versão do Python:**
   ```bash
   python --version
   ```
   (Requer Python 3.8+)

2. **Atualize o pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Instale uma dependência por vez e teste:**
   ```bash
   pip install pulp
   python -c "import pulp; print('PuLP funcionando!')"
   ```

4. **Use o instalador alternativo:**
   ```bash
   python -m pip install pulp
   ```

## 💡 Dicas Importantes

- **Windows**: Certifique-se de estar executando o prompt como Administrador se necessário
- **Mac/Linux**: Pode precisar usar `pip3` em vez de `pip`
- **Proxy Corporativo**: Use `pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <pacote>`
- **Sem Internet**: Baixe os pacotes .whl manualmente do PyPI

## 📞 Suporte

Se continuar com problemas:
1. Execute `python install_dependencies.py` e copie a saída
2. Verifique se você tem permissões para instalar pacotes
3. Considere usar um ambiente virtual dedicado
4. Verifique se há conflitos com outras instalações Python

Após resolver as dependências, a aplicação deve funcionar perfeitamente! 🎉
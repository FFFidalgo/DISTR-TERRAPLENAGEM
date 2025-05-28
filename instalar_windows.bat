@echo off
echo 🔧 Instalador de Dependências - Otimização de Terraplenagem
echo ==========================================================
echo.

echo 📋 Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python não encontrado! Instale Python 3.8+ primeiro.
    pause
    exit /b 1
)

echo.
echo 📦 Atualizando pip...
python -m pip install --upgrade pip

echo.
echo 📥 Instalando dependências...
echo.

echo 🔄 Instalando pandas...
pip install "pandas>=2.0.0"

echo 🔄 Instalando numpy...
pip install "numpy>=1.24.0"

echo 🔄 Instalando streamlit...
pip install "streamlit>=1.28.0"

echo 🔄 Instalando pulp...
pip install "pulp>=2.7.0"

echo 🔄 Instalando openpyxl...
pip install "openpyxl>=3.1.0"

echo 🔄 Instalando xlsxwriter...
pip install "xlsxwriter>=3.1.0"

echo 🔄 Instalando scipy...
pip install "scipy>=1.10.0"

echo 🔄 Instalando plotly...
pip install "plotly>=5.15.0"

echo 🔄 Instalando pytest...
pip install "pytest>=7.4.0"

echo.
echo 🔍 Verificando instalação...
python -c "import pandas, numpy, streamlit, pulp, openpyxl, xlsxwriter, plotly, scipy, pytest; print('✅ Todas as dependências instaladas com sucesso!')"

if %errorlevel% eq 0 (
    echo.
    echo 🎉 Instalação concluída!
    echo.
    echo 🚀 Para executar a aplicação, digite:
    echo    streamlit run main.py
    echo.
) else (
    echo.
    echo ❌ Algumas dependências falharam na instalação.
    echo    Tente executar: python install_dependencies.py
    echo.
)

pause
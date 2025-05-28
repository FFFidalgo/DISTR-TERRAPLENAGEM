#!/bin/bash

echo "🔧 Instalador de Dependências - Otimização de Terraplenagem"
echo "=========================================================="
echo

echo "📋 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python não encontrado! Instale Python 3.8+ primeiro."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD --version

echo
echo "📦 Atualizando pip..."
$PYTHON_CMD -m pip install --upgrade pip

echo
echo "📥 Instalando dependências..."
echo

echo "🔄 Instalando todas as dependências..."
$PYTHON_CMD -m pip install -r requirements.txt

echo
echo "🔍 Verificando instalação..."
$PYTHON_CMD -c "import pandas, numpy, streamlit, pulp, openpyxl, xlsxwriter, plotly, scipy, pytest; print('✅ Todas as dependências instaladas com sucesso!')"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Instalação concluída!"
    echo
    echo "🚀 Para executar a aplicação, digite:"
    echo "   streamlit run main.py"
    echo
else
    echo
    echo "❌ Algumas dependências falharam na instalação."
    echo "   Tente executar: $PYTHON_CMD install_dependencies.py"
    echo
fi
"""
Script de instalação automática de dependências
para a aplicação de otimização de terraplenagem
"""

import subprocess
import sys
import os

def install_package(package):
    """Instala um pacote usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name):
    """Verifica se um pacote está instalado"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Função principal de instalação"""
    print("🔧 Instalador de Dependências - Otimização de Terraplenagem")
    print("=" * 60)
    
    # Lista de dependências necessárias
    dependencies = [
        ("pandas", "pandas>=2.0.0"),
        ("numpy", "numpy>=1.24.0"),
        ("streamlit", "streamlit>=1.28.0"),
        ("pulp", "pulp>=2.7.0"),
        ("openpyxl", "openpyxl>=3.1.0"),
        ("xlsxwriter", "xlsxwriter>=3.1.0"),
        ("scipy", "scipy>=1.10.0"),
        ("pytest", "pytest>=7.4.0")
    ]
    
    print("📋 Verificando dependências...")
    missing_packages = []
    
    for package_name, package_spec in dependencies:
        if check_package(package_name):
            print(f"✅ {package_name} - OK")
        else:
            print(f"❌ {package_name} - FALTANDO")
            missing_packages.append(package_spec)
    
    if not missing_packages:
        print("\n🎉 Todas as dependências estão instaladas!")
        return
    
    print(f"\n📦 Instalando {len(missing_packages)} pacotes faltantes...")
    
    for package_spec in missing_packages:
        package_name = package_spec.split(">=")[0]
        print(f"\n🔄 Instalando {package_name}...")
        
        if install_package(package_spec):
            print(f"✅ {package_name} instalado com sucesso!")
        else:
            print(f"❌ Falha ao instalar {package_name}")
            print(f"   Tente manualmente: pip install {package_spec}")
    
    print("\n🔍 Verificação final...")
    all_installed = True
    
    for package_name, _ in dependencies:
        if check_package(package_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} - AINDA FALTANDO")
            all_installed = False
    
    if all_installed:
        print("\n🎉 Instalação concluída com sucesso!")
        print("\n🚀 Agora você pode executar a aplicação com:")
        print("   streamlit run main.py")
    else:
        print("\n⚠️ Algumas dependências ainda estão faltando.")
        print("   Tente instalar manualmente ou verifique sua instalação do Python.")

if __name__ == "__main__":
    main()
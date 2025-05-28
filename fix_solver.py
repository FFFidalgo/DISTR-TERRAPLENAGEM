"""
Correção para problemas de solver do PuLP
Resolve o erro: "cannot execute cbc.exe.exe"
"""

import subprocess
import sys
import os
import platform

def install_cbc_solver():
    """Instala o solver CBC para diferentes sistemas operacionais"""
    system = platform.system().lower()
    
    print("🔧 Configurando solver CBC para otimização...")
    
    if system == "windows":
        print("📥 Instalando CBC para Windows...")
        try:
            # Tentar instalar o solver via conda-forge (mais confiável)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp[cbc]"])
            print("✅ CBC instalado via pip[cbc]")
            return True
        except:
            try:
                # Método alternativo
                subprocess.check_call(["conda", "install", "-c", "conda-forge", "coincbc"])
                print("✅ CBC instalado via conda")
                return True
            except:
                print("⚠️ Instalação automática falhou. Configurando solver alternativo...")
                return False
    
    elif system == "linux":
        print("📥 Instalando CBC para Linux...")
        try:
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "coinor-cbc"])
            print("✅ CBC instalado via apt-get")
            return True
        except:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp[cbc]"])
                print("✅ CBC instalado via pip[cbc]")
                return True
            except:
                print("⚠️ Instalação automática falhou. Configurando solver alternativo...")
                return False
    
    elif system == "darwin":  # macOS
        print("📥 Instalando CBC para macOS...")
        try:
            subprocess.check_call(["brew", "install", "cbc"])
            print("✅ CBC instalado via brew")
            return True
        except:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp[cbc]"])
                print("✅ CBC instalado via pip[cbc]")
                return True
            except:
                print("⚠️ Instalação automática falhou. Configurando solver alternativo...")
                return False
    
    return False

def test_solvers():
    """Testa quais solvers estão disponíveis"""
    import pulp as pl
    
    print("🔍 Testando solvers disponíveis...")
    
    solvers = [
        ("PULP_CBC_CMD", pl.PULP_CBC_CMD),
        ("COIN_CMD", pl.COIN_CMD),
        ("CPLEX_CMD", pl.CPLEX_CMD),
        ("GUROBI_CMD", pl.GUROBI_CMD),
        ("GLPK_CMD", pl.GLPK_CMD)
    ]
    
    available_solvers = []
    
    for name, solver_class in solvers:
        try:
            solver = solver_class()
            if solver.available():
                print(f"✅ {name} - Disponível")
                available_solvers.append((name, solver_class))
            else:
                print(f"❌ {name} - Não disponível")
        except Exception as e:
            print(f"❌ {name} - Erro: {str(e)}")
    
    return available_solvers

def create_simple_test():
    """Cria um teste simples de otimização para verificar se funciona"""
    import pulp as pl
    
    print("🧪 Testando otimização simples...")
    
    # Problema simples de teste
    prob = pl.LpProblem("Teste", pl.LpMinimize)
    x = pl.LpVariable("x", lowBound=0)
    y = pl.LpVariable("y", lowBound=0)
    
    prob += x + y, "Função Objetivo"
    prob += x + 2*y >= 3, "Restrição 1"
    prob += 2*x + y >= 3, "Restrição 2"
    
    # Testar diferentes solvers
    available_solvers = test_solvers()
    
    for name, solver_class in available_solvers:
        try:
            print(f"\n🔄 Testando com {name}...")
            solver = solver_class(msg=False)
            prob.solve(solver)
            
            if prob.status == pl.LpStatusOptimal:
                print(f"✅ {name} funcionou! Resultado: x={x.varValue}, y={y.varValue}")
                return name, solver_class
            else:
                print(f"⚠️ {name} resolveu mas não encontrou solução ótima")
        except Exception as e:
            print(f"❌ {name} falhou: {str(e)}")
    
    return None, None

def main():
    """Função principal de correção"""
    print("🔧 Corretor de Problemas do Solver PuLP")
    print("=" * 50)
    
    # Tentar instalar CBC
    cbc_installed = install_cbc_solver()
    
    # Testar solvers disponíveis
    working_solver, solver_class = create_simple_test()
    
    if working_solver:
        print(f"\n🎉 Solver funcionando encontrado: {working_solver}")
        
        # Criar arquivo de configuração
        config_content = f'''
# Configuração do solver para a aplicação
WORKING_SOLVER = "{working_solver}"
SOLVER_CLASS = "{solver_class.__name__}"

# Para usar na aplicação:
# from solver_config import WORKING_SOLVER, SOLVER_CLASS
'''
        
        with open('solver_config.py', 'w') as f:
            f.write(config_content)
        
        print("📄 Arquivo de configuração criado: solver_config.py")
        
    else:
        print("\n❌ Nenhum solver funcionando encontrado!")
        print("📋 Soluções manuais:")
        print("1. Instale o Anaconda e use: conda install -c conda-forge coincbc")
        print("2. Baixe o CBC manualmente de: https://github.com/coin-or/Cbc/releases")
        print("3. Use um solver online ou em nuvem")
    
    return working_solver is not None

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Agora você pode executar a aplicação!")
        print("   streamlit run main.py")
    else:
        print("\n🆘 Se o problema persistir, use a versão com solver manual.")
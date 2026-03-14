@echo off
REM ============================================================
REM run_pipeline.bat — Executa o pipeline completo do projeto
REM Penguins MLOps: download, split, train, evaluate
REM ============================================================

REM Garante que o script e executado a partir da pasta do projeto
cd /d "%~dp0.."

echo.
echo ============================================================
echo   PENGUINS MLOPS — Pipeline Completo
echo ============================================================

echo.
echo [1/4] Baixando dataset...
venv\Scripts\python main.py download
if errorlevel 1 (
    echo ERRO: Falha no download. Verifique a conexao.
    pause
    exit /b 1
)

echo.
echo [2/4] Dividindo em treino e teste...
venv\Scripts\python main.py split
if errorlevel 1 (
    echo ERRO: Falha na divisao dos dados.
    pause
    exit /b 1
)

echo.
echo [3/4] Treinando modelos (Classificador + Regressor)...
venv\Scripts\python main.py train
if errorlevel 1 (
    echo ERRO: Falha no treinamento.
    pause
    exit /b 1
)

echo.
echo [4/4] Avaliando desempenho no conjunto de teste...
venv\Scripts\python main.py evaluate
if errorlevel 1 (
    echo ERRO: Falha na avaliacao.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Pipeline concluido! Confira os graficos em reports/
echo ============================================================
pause

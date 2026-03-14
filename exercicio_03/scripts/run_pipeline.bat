@echo off
REM Pipeline completo do Exercicio 03 — Wine Classifier
REM Executa: download -> features -> train -> evaluate
REM Uso: scripts\run_pipeline.bat

SET ROOT=%~dp0..
SET PYTHON=%ROOT%\venv\Scripts\python.exe

echo ==============================================
echo  Wine Classifier — Pipeline Completo
echo ==============================================

echo.
echo [1/4] Baixando dataset...
"%PYTHON%" "%ROOT%\main.py" download
IF %ERRORLEVEL% NEQ 0 ( echo ERRO no download & exit /b 1 )

echo.
echo [2/4] Normalizando features...
"%PYTHON%" "%ROOT%\main.py" features
IF %ERRORLEVEL% NEQ 0 ( echo ERRO nas features & exit /b 1 )

echo.
echo [3/4] Treinando modelo...
"%PYTHON%" "%ROOT%\main.py" train
IF %ERRORLEVEL% NEQ 0 ( echo ERRO no treino & exit /b 1 )

echo.
echo [4/4] Avaliando modelo...
"%PYTHON%" "%ROOT%\main.py" evaluate
IF %ERRORLEVEL% NEQ 0 ( echo ERRO na avaliacao & exit /b 1 )

echo.
echo ==============================================
echo  Pipeline concluido com sucesso!
echo  Relatorio salvo em: data\evaluation.txt
echo ==============================================

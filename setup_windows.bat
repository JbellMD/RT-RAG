@echo off
echo Creating Python virtual environment (venv)...
python -m venv venv

echo Activating virtual environment and installing dependencies...
call .\venv\Scripts\activate.bat && pip install -r requirements.txt

echo Setup complete.
echo To activate the virtual environment in a new terminal, run: .\venv\Scripts\activate.bat
echo Then, to run the assistant: python rag_assistant.py
pause

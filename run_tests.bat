@echo off
set PYTHONPATH=src
python -m pytest tests/ -v

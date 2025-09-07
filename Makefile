# Makefile

# Default target
.PHONY: all
all: db eval backend frontend

# Run initialization script for database
.PHONY: db
db:
	python -m backend.init_db

# Run backend
.PHONY: backend
backend:
	uvicorn backend.main:app --reload

# Run frontend
.PHONY: frontend
frontend:
	streamlit run frontend/app.py

# Run ground truth evaluation
.PHONY: eval
eval:
	python -m tests.evaluation

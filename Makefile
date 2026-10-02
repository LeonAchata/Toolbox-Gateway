.PHONY: up down logs test lint format

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail 100

test:
	cd toolbox && python -m pytest -q
	cd llm-gateway && python -m pytest -q
	cd agents && python -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

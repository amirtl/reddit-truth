.PHONY: install test run worker migrate shell docker-up docker-down docker-logs lint

## — Local development (uses uv) ——————————————————————————

install:
	uv venv
	uv pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "\n✓ Environment ready. Edit .env with your credentials, then run: make migrate"

migrate:
	uv run python manage.py migrate

shell:
	uv run python manage.py shell

run:
	uv run python manage.py runserver

worker:
	uv run celery -A reddit_truth worker --loglevel=info

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --ignore=tests/test_embedder_clusterer.py

## — Docker ——————————————————————————————————————————————

docker-up:
	docker-compose up --build

docker-up-d:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-migrate:
	docker-compose exec web python manage.py migrate

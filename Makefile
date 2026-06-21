.PHONY: install test run worker migrate shell smoke docker-up docker-down docker-logs lint

QUERY ?= Sony WH-1000XM5

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
	# macOS: the prefork pool fork()s workers, which aborts (SIGABRT) once
	# PyTorch/sentence-transformers is loaded. Use the solo pool locally and mark
	# native libs fork-safe. The Docker/Linux worker keeps prefork concurrency.
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES TOKENIZERS_PARALLELISM=false \
		uv run celery -A reddit_truth worker --pool=solo --loglevel=info

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --ignore=tests/test_embedder_clusterer.py

# End-to-end smoke test against a running server + worker. Override the product
# with: make smoke QUERY="AirPods Pro"
smoke:
	uv run python scripts/smoke_test.py "$(QUERY)"

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

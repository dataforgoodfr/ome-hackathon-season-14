SHELL=/bin/bash

APP_NAME?=$(shell grep -m 1 name pyproject.toml | cut -d'"' -f2)
APP_VERSION?=$(shell grep -m 1 version pyproject.toml | sed 's|^version = "\(.*\)"|\1|')

BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD | sed 's|^.*\/||g' | sed 's|_|-|g' | tr '[:upper:]' '[:lower:]')
COMMIT_HASH=$(shell git rev-parse --short HEAD)
PROD_VERSION=${APP_VERSION}
DEV_VERSION=${APP_VERSION}-${BRANCH_NAME}-${COMMIT_HASH}-$(shell whoami)


DOCKER_REGISTRY?=""
DOCKER_IMAGE?=$(if $(value $(DOCKER_REGISTRY)),$(DOCKER_REGISTRY)/$(APP_NAME),$(APP_NAME))
DOCKER_TAG?=${DEV_VERSION}


PYTHON_VERSION?=3.12
RUN?=uv run

.DEFAULT_GOAL := all

## help: Display list of commands (from gazr.io)
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## all: Run all targets
.PHONY: all
all: init quality test build

# ----------------------------------------------------------------------
#			 Versions
# ----------------------------------------------------------------------

## version: Print the semantic app version (aliases are app-version, prod-version)
.PHONY: version prod-version app-version
version prod-version app-version:
	@echo ${APP_VERSION}

## dev-version: Print the version as <semver>-<branchname>-<commithash>-<user>
.PHONY: dev-version
dev-version:
	@echo ${DEV_VERSION}


## image: Print the docker image without tag
image:
	@echo ${DOCKER_IMAGE}


## info: Print information about the installation
.PHONY: info
info:
	@echo "App name       : ${APP_NAME}"
	@echo "App version    : ${APP_VERSION}"
	@echo "Python version : ${PYTHON_VERSION} (used is $(shell $(RUN) python --version 2> /dev/null | grep -Eo '[0-9\.]+' || echo 'unknown'))"

	@echo "Docker image   : ${DOCKER_IMAGE}"


# ----------------------------------------------------------------------
#			Local setup
# ----------------------------------------------------------------------

## init: Install local setup with uv and pre-commit
.PHONY: init
init:
	uv python install "$(PYTHON_VERSION)"
	uv venv --quiet --allow-existing --python "$(PYTHON_VERSION)"
	uv sync --all-extras --all-groups --all-packages
	uv pip check
	uv run pre-commit install

## clean: Clean cache files
.PHONY: clean
clean:
	find . -type d -name __pycache__ -delete || true
	rm -rf .ruff_cache .mypy_cache || true
	rm -rf dist || true

## deinit: Clean and uninstall local setup
.PHONY: deinit
deinit: clean
	uv run pre-commit uninstall
	rm -rf ".venv" || true

## update: Update python and pre-commit dependencies
.PHONY: update
update: update-dependencies update-precommit

## update-dependencies: Update python dependencies (alias update-deps)
.PHONY: update-dependencies update-deps
update-dependencies update-deps:
	uv sync --upgrade --all-extras --all-groups --all-packages
	uv tree -q --outdated --depth=1 | grep latest || echo "Everything is up to date !"

## update-precommit: Update pre-commit dependencies
.PHONY: update-precommit
update-precommit:
	$(RUN) pre-commit autoupdate -j 10

## export-requirements: Export requirements.txt from uv configuration
.PHONY: export-requirements
export-requirements:
	@uv export --quiet --format requirements.txt --no-hashes --no-annotate > requirements.txt

## run: Run the app locally
.PHONY: run
run:
	@echo "Not implemented"

# ----------------------------------------------------------------------
#			Quality & Test
# ----------------------------------------------------------------------

## quality: Check and fix lint and code styling rules (aliases are lint, format)
.PHONY: quality lint format
quality lint format:
	$(RUN) pre-commit run -a

## lint-parallel: Run all pre-commit hooks in parallel
.PHONY: lint-parallel
lint-parallel:
	@grep "id: " .pre-commit-config.yaml \
		| awk '{ print $$3 }' \
		| xargs -P0 -I% bash -c 'RESULT="$$(uv run pre-commit run -a %)"; EXIT_CODE=$$?; echo "$${RESULT}"; exit "$${EXIT_CODE}"'
# NOTE: Trick to run all hooks concurrently for faster CI,
# 		need $(...) for coherent output, might mix files that are changed but yolo


## test: Run all registered tests (unit, functional and integration).
.PHONY: test
test:
	$(RUN) python -m pytest tests --cov

## test-debug: Run all registered tests (unit, functional and integration) with ipdb on error
.PHONY: test-debug
test-debug:
	$(RUN) python -m pytest tests --pdb

## check-versions-coherence: Check coherence between major tools versions
.PHONY: check-versions-coherence
check-versions-coherence:
	@[[ "$(shell grep -c -E "^python\w* = \"\^?${PYTHON_VERSION}\"" pyproject.toml)" == "1" ]] \
		|| (echo "Incoherence for PYTHON_VERSION in pyproject.toml"; exit 1)
	@echo "Python version ok for ${PYTHON_VERSION}"

# ----------------------------------------------------------------------
#			Build & Publish
# ----------------------------------------------------------------------

# ## build-package: Build the Python package
# .PHONY: build-package
# build-package:
# 	uv build --all-packages

# ## publish-package: Publish the Python package
# .PHONY: publish-package
# publish-package:
# 	uv publish


# ## build-docker: Build the docker image
# .PHONY: build-docker
# build-docker:
# 	docker build . \
# 		--pull \
# 		--file Dockerfile \
# 		--tag "${DOCKER_IMAGE}:${DOCKER_TAG}" \
# 		--build-arg "APP_NAME=${APP_NAME}" \
# 		--build-arg "APP_VERSION=${APP_VERSION}" \
# 		--build-arg "COMMIT_HASH=${COMMIT_HASH}" \
# 		--build-arg "PYTHON_VERSION=$(PYTHON_VERSION)"

# ## publish-docker: Push the docker image
# .PHONY: publish-docker
# publish-docker:
# 	docker push "${DOCKER_IMAGE}:${DOCKER_TAG}"

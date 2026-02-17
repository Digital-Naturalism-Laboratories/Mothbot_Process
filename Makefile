SHELL := /bin/bash

ORG ?= Digital-Naturalism-Laboratories
REPO ?= mothbot-detect
VISIBILITY ?= public

.PHONY: help setup run run-watch gpu-setup dev-help release-help venv install-cpu install-gpu-cuda118 run-local run-local-watch clean build-macos build-linux build-windows package-macos init-git publish-org

help:
	@echo "Mothbot Make entrypoint"
	@echo ""
	@echo "Local testing (recommended):"
	@echo "  make setup                  Install default CPU dependencies"
	@echo "  make run                    Run local Gradio UI"
	@echo "  make run-watch              Run local Gradio with auto-refresh"
	@echo ""
	@echo "Optional local:"
	@echo "  make gpu-setup              Install optional CUDA 11.8 dependencies"
	@echo "  make dev-help               Show all dev targets"
	@echo ""
	@echo "Release / packaging:"
	@echo "  make release-help           Show all release + cleanup targets"
	@echo "  make build-macos            Build macOS app bundle"
	@echo "  make build-linux            Build Linux app folder"
	@echo "  make build-windows          Build Windows app folder"
	@echo "  make package-macos          Create macOS zip+dmg from existing build"
	@echo "  make clean                  Remove packaging artifacts and caches"
	@echo ""
	@echo "Git / publishing:"
	@echo "  make init-git               Init local git repo on branch main"
	@echo "  make publish-org            Create/push GitHub repo $(ORG)/$(REPO)"
	@echo ""
	@echo "Direct file usage:"
	@echo "  make -f make/dev.mk <target>"
	@echo "  make -f make/release.mk <target>"

setup:
	@$(MAKE) -f make/dev.mk install-cpu

run:
	@$(MAKE) -f make/dev.mk run-local

run-watch:
	@$(MAKE) -f make/dev.mk run-local-watch

gpu-setup:
	@$(MAKE) -f make/dev.mk install-gpu-cuda118

dev-help:
	@$(MAKE) -f make/dev.mk help

release-help:
	@$(MAKE) -f make/release.mk help

venv install-cpu install-gpu-cuda118 run-local run-local-watch:
	@$(MAKE) -f make/dev.mk $@

clean build-macos build-linux build-windows package-macos:
	@$(MAKE) -f make/release.mk $@

init-git:
	@if [ ! -d .git ]; then git init; fi
	@git branch -M main
	@echo "Initialized git repo on branch main"

publish-org: init-git
	@if ! command -v gh >/dev/null 2>&1; then echo "GitHub CLI (gh) is required."; exit 1; fi
	@gh auth status >/dev/null
	@if git remote get-url origin >/dev/null 2>&1; then \
		echo "origin already configured: $$(git remote get-url origin)"; \
	else \
		gh repo create "$(ORG)/$(REPO)" --source=. --remote=origin --$(VISIBILITY) --description "Moth detection workflows and desktop tooling"; \
	fi
	@git add .
	@git commit -m "Initial import for mothbot-detect" || true
	@git push -u origin main


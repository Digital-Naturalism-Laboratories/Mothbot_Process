SHELL := /bin/bash

.PHONY: help build-macos build-linux build-windows package-macos clean

help:
	@echo "Mothbot release targets"
	@echo ""
	@echo "Build:"
	@echo "  make -f make/release.mk build-macos"
	@echo "  make -f make/release.mk build-linux"
	@echo "  make -f make/release.mk build-windows"
	@echo ""
	@echo "Package:"
	@echo "  make -f make/release.mk package-macos"
	@echo ""
	@echo "Cleanup:"
	@echo "  make -f make/release.mk clean"

build-macos:
	bash apps/scripts/build_desktop_macos.sh

build-linux:
	bash apps/scripts/build_desktop_linux.sh

build-windows:
	@if [ "$$OS" != "Windows_NT" ]; then \
		echo "build-windows must be run on Windows (PowerShell)."; \
		echo "Current host is non-Windows."; \
		echo "Use a Windows machine/runner, or run make build-macos / make build-linux on this host."; \
		exit 1; \
	fi
	@if ! command -v pwsh >/dev/null 2>&1; then \
		echo "PowerShell (pwsh) is required. Install PowerShell 7+ and re-run."; \
		exit 1; \
	fi
	pwsh -File apps/scripts/build_desktop_windows.ps1

package-macos:
	bash apps/scripts/package_release_macos.sh

clean:
	rm -rf apps/build apps/dist apps/release __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +


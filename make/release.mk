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
	pwsh -File apps/scripts/build_desktop_windows.ps1

package-macos:
	bash apps/scripts/package_release_macos.sh

clean:
	rm -rf apps/build apps/dist apps/release __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +


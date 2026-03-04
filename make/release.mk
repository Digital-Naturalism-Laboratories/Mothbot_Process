SHELL := /bin/bash

.PHONY: help build-macos build-linux build-windows package-macos release-bump clean

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
	@echo "Release:"
	@echo "  make -f make/release.mk release-bump"
	@echo "    - prompts for major/minor/patch, previews next vX.Y.Z, confirms, then pushes tag"
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

release-bump:
	@if [ ! -t 0 ] || [ ! -t 1 ]; then \
		echo "release-bump is interactive and requires a TTY."; \
		exit 1; \
	fi
	@if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "Not inside a git repository."; \
		exit 1; \
	fi
	@if ! git remote get-url origin >/dev/null 2>&1; then \
		echo "Missing git remote 'origin'."; \
		echo "Add one with: git remote add origin <url>"; \
		exit 1; \
	fi
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Working tree is not clean. Commit/stash changes before bumping release."; \
		exit 1; \
	fi
	@latest_tag="$$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n 1)"; \
	if [ -z "$$latest_tag" ]; then \
		major=1; minor=0; patch=0; \
		echo "No existing SemVer tags found. Initial base version: v1.0.0"; \
	else \
		version="$${latest_tag#v}"; \
		major="$${version%%.*}"; \
		rest="$${version#*.}"; \
		minor="$${rest%%.*}"; \
		patch="$${rest##*.}"; \
		echo "Latest SemVer tag: $$latest_tag"; \
	fi; \
	echo ""; \
	echo "Select bump type:"; \
	echo "  1) patch ($$major.$$minor.$$((patch + 1)))"; \
	echo "  2) minor ($$major.$$((minor + 1)).0)"; \
	echo "  3) major ($$((major + 1)).0.0)"; \
	read -r -p "Choice [1-3]: " choice; \
	case "$$choice" in \
		1) new_major=$$major; new_minor=$$minor; new_patch=$$((patch + 1)); bump_name="patch" ;; \
		2) new_major=$$major; new_minor=$$((minor + 1)); new_patch=0; bump_name="minor" ;; \
		3) new_major=$$((major + 1)); new_minor=0; new_patch=0; bump_name="major" ;; \
		*) echo "Invalid choice. Cancelled."; exit 1 ;; \
	esac; \
	next_tag="v$${new_major}.$${new_minor}.$${new_patch}"; \
	if git rev-parse "$$next_tag" >/dev/null 2>&1; then \
		echo "Tag $$next_tag already exists locally."; \
		exit 1; \
	fi; \
	if git ls-remote --exit-code --tags origin "refs/tags/$$next_tag" >/dev/null 2>&1; then \
		echo "Tag $$next_tag already exists on origin."; \
		exit 1; \
	fi; \
	echo ""; \
	echo "Bump type: $$bump_name"; \
	echo "Next tag:  $$next_tag"; \
	read -r -p "Create and push this tag? [y/N]: " confirm; \
	case "$$confirm" in \
		y|Y|yes|YES) ;; \
		*) echo "Cancelled."; exit 0 ;; \
	esac; \
	git tag -a "$$next_tag" -m "Release $$next_tag"; \
	git push origin "$$next_tag"; \
	echo "Release tag pushed: $$next_tag"

clean:
	rm -rf apps/build apps/dist apps/release __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +


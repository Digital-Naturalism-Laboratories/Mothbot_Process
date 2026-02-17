# Publish to GitHub Organization

This project can be published to:

- `Digital-Naturalism-Laboratories/mothbot-detect`

## Prerequisites

- GitHub CLI installed (`gh`)
- Logged in via `gh auth login`
- Permission to create repositories in the organization

## One-command flow

From project root:

- `make publish-org`

This will:

1. Initialize git (if needed) and set branch to `main`
2. Create the GitHub repo and set `origin` if missing
3. Stage and commit local files
4. Push to `origin/main`

## Override org/repo name

- `make publish-org ORG=Digital-Naturalism-Laboratories REPO=mothbot-detect`

## Optional manual flow

```bash
git init
git branch -M main
git add .
git commit -m "Initial import for mothbot-detect"
gh repo create Digital-Naturalism-Laboratories/mothbot-detect --source=. --remote=origin --public
git push -u origin main
```

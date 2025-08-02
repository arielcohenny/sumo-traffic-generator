# Branching Workflow Guide

This document explains the feature branch workflow for the SUMO Traffic Generator project.

## Overview

We use a feature branch workflow to keep the `main` branch stable and enable safe collaboration:

- `main` branch: Production-ready code, protected by CI tests
- Feature branches: Development work, merged via Pull Requests
- GitHub Actions: Automatic testing on PRs before merge

## Daily Workflow

### 1. Create a Feature Branch

Always create a new branch for each feature or bug fix:

```bash
# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Your Changes

Work normally on your feature branch:

```bash
# Make your changes to files
# Then stage and commit
git add .
git commit -m "Add your meaningful commit message"
```

### 3. Push and Create PR

Push your branch and create a Pull Request:

```bash
# Push your feature branch
git push -u origin feature/your-feature-name

# Create PR using GitHub CLI (after authentication)
gh pr create --title "Your PR Title" --body "Description of changes"

# Or create PR manually on GitHub web interface
```

### 4. CI Validation

GitHub Actions will automatically:
- Run smoke tests (< 1 minute)
- Run scenario tests (2-5 minutes)
- Report results on the PR

### 5. Merge to Main

Once CI passes and you're satisfied:
- Merge the PR on GitHub
- Delete the feature branch
- Switch back to main locally

```bash
# Switch back to main and pull latest
git checkout main
git pull origin main

# Delete the merged feature branch
git branch -d feature/your-feature-name
```

## Branch Protection Rules

To set up branch protection for `main`:

1. Go to GitHub repository settings
2. Navigate to "Branches" section
3. Add protection rule for `main` branch:
   - ✅ Require status checks before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators (you can bypass as owner)
   - ✅ Allow force pushes (for emergency fixes)
   - ✅ Allow deletions (for maintenance)

## Benefits

- **Safety**: `main` always has passing tests
- **History**: Clean merge commits show feature completions
- **Collaboration**: Easy to review changes before merge
- **Rollback**: Simple to revert problematic features
- **CI Confidence**: Tests run before merge, not after

## Quick Commands Reference

```bash
# Start new feature
git checkout main && git pull && git checkout -b feature/new-feature

# Commit and push
git add . && git commit -m "Description" && git push -u origin HEAD

# Create PR (after gh auth login)
gh pr create --title "Feature: Description" --body "Changes made..."

# After merge, cleanup
git checkout main && git pull && git branch -d feature/new-feature
```
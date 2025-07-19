# Git Workflow Command

When you type `/commit`, I will:

1. Analyze current git changes and generate a descriptive commit message
2. Show you the proposed commit message and ask for approval
3. After your approval, execute these commands in sequence:
   - `git add .`
   - `git commit -m "your approved description"`
   - `git push`

## Usage
```
/commit [optional custom description]
```

## Examples
```
/commit                                 # I analyze changes and propose message
/commit Fix lane assignment bug         # Use your custom message
```

This is a simple command that uses Claude's terminal tools directly - no GitHub App required.
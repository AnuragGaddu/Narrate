# Git Cleanup — Voice Model Files — 2026-02-17

Record of removing the `voices/` directory from git tracking and purging it
from history. The directory contains the Piper TTS ONNX model (~61 MB) which
should not be stored in the repository.

---

## 1. Problem

The `voices/` directory was committed to the repository on February 15 during
the TTS tab implementation. It contained:

```
voices/en_US-lessac-medium.onnx       61 MB   (neural network weights)
voices/en_US-lessac-medium.onnx.json  4.8 KB  (model config)
```

A 61 MB binary file inflates the repository size, slows clones, and cannot
be meaningfully diffed. Voice models should be downloaded during setup, not
shipped in the repo.

---

## 2. Fix

### 2.1 Added `voices/` to `.gitignore`

```diff
 models/
+voices/
 static/audio/
```

This prevents future `git add` from re-tracking the directory.

### 2.2 Removed from git index

```bash
git rm --cached -r voices/
```

This untracks the files without deleting them from disk. The working copy
retains the model files for the running application.

### 2.3 Purged from git history

```bash
git filter-repo --path voices/ --invert-paths
```

`git filter-repo` rewrites every commit to exclude any path matching
`voices/`. This removes the 61 MB blob from all historical commits,
reclaiming repository space.

**Side effect:** Rewriting history changes all commit hashes from the point
where `voices/` was first added. A force push (`git push --force`) is required
to update the remote.

---

## 3. Setup Instructions

After cloning, voice models must be downloaded manually:

```bash
mkdir -p voices
wget -O voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## 4. Troubleshooting -- Force Push Failed

### 4.1 Symptom

After `git filter-repo` rewrote history, the force push to GitHub failed:

```
$ git push --force origin master
fatal: could not read Username for 'https://github.com': No such device or address
```

### 4.2 Diagnosis

Three authentication methods were checked:

```bash
# 1. Credential helper -- not configured
$ git config credential.helper
(empty)

# 2. SSH keys -- none set up
$ ssh -T git@github.com
Permission denied (publickey).

# 3. GitHub CLI -- not installed
$ gh --version
bash: gh: command not found
```

No authentication method was available on the Pi.

### 4.3 Additional issue -- remote stripped by filter-repo

`git filter-repo` removes all remotes as a safety measure (to prevent
accidentally pushing rewritten history to the wrong place). The remote had
to be re-added:

```bash
git remote add origin https://github.com/AnuragGaddu/Narrate.git
```

### 4.4 Resolution options

Three approaches were identified:

1. **Personal access token**: Generate a PAT on GitHub, then push with the
   token embedded in the URL:
   ```bash
   git push --force https://<TOKEN>@github.com/AnuragGaddu/Narrate.git master
   ```

2. **SSH key setup**: Generate an SSH keypair, add the public key to GitHub,
   and switch the remote to SSH:
   ```bash
   ssh-keygen -t ed25519
   git remote set-url origin git@github.com:AnuragGaddu/Narrate.git
   ```

3. **GitHub CLI**: Install `gh`, authenticate, then push normally:
   ```bash
   sudo apt install gh
   gh auth login
   git push --force origin master
   ```

---

## 5. Lesson

Large binary assets (ONNX models, HEF files, WAV samples) should be excluded
from git from the start. The `.gitignore` now covers:

| Pattern | Contents |
|---------|----------|
| `models/` | Hailo HEF and Vosk model |
| `voices/` | Piper ONNX voice model |
| `static/audio/` | Generated WAV files |

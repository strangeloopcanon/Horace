#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run(cmd, cwd=None, check=True):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        sys.stderr.write(res.stderr)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)
    return res


def ensure_git_repo(root: Path):
    if not (root / '.git').exists():
        run(['git', 'init'], cwd=root)
        # Create initial commit if none
        run(['git', 'add', '-A'], cwd=root)
        run(['git', 'commit', '-m', 'Initial commit'], cwd=root, check=False)


def create_github_repo(token: str, full_name: str, private: bool = True):
    import requests  # type: ignore
    if '/' in full_name:
        owner, name = full_name.split('/', 1)
    else:
        owner, name = None, full_name
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github+json'}
    if owner:
        url = f'https://api.github.com/orgs/{owner}/repos'
        body = {'name': name, 'private': private, 'auto_init': False}
    else:
        url = 'https://api.github.com/user/repos'
        body = {'name': name, 'private': private, 'auto_init': False}
    r = requests.post(url, headers=headers, json=body)
    if r.status_code in (201, 202):
        data = r.json()
        print(f"Created repo: {data.get('full_name')}")
        return data
    elif r.status_code == 422 and 'already exists' in r.text:
        print('Repo already exists; continuing')
        # fetch data
        if owner:
            r2 = requests.get(f'https://api.github.com/repos/{owner}/{name}', headers=headers)
        else:
            # need to know your username
            rme = requests.get('https://api.github.com/user', headers=headers)
            user = rme.json().get('login')
            r2 = requests.get(f'https://api.github.com/repos/{user}/{name}', headers=headers)
        r2.raise_for_status()
        return r2.json()
    else:
        print(f"GitHub API error: {r.status_code} {r.text}")
        r.raise_for_status()


def push_repo(root: Path, remote_url: str, default_branch: str = 'main'):
    ensure_git_repo(root)
    # checkout main
    run(['git', 'checkout', '-B', default_branch], cwd=root)
    # add everything (respecting .gitignore)
    run(['git', 'add', '-A'], cwd=root)
    run(['git', 'commit', '-m', 'Update project'], cwd=root, check=False)
    # set remote
    remotes = run(['git', 'remote'], cwd=root, check=False).stdout.strip().split()
    if 'origin' in remotes:
        run(['git', 'remote', 'set-url', 'origin', remote_url], cwd=root)
    else:
        run(['git', 'remote', 'add', 'origin', remote_url], cwd=root)
    # push
    run(['git', 'push', '-u', 'origin', default_branch], cwd=root)


def main():
    ap = argparse.ArgumentParser(description='Create a private GitHub repo and push the current workspace')
    ap.add_argument('--full-name', required=True, help='Target repo full name: <owner>/<name> (owner=org or user)')
    ap.add_argument('--private', action='store_true', default=True, help='Create as private (default)')
    ap.add_argument('--branch', default='main')
    ap.add_argument('--token', default=None, help='GitHub token (or set GITHUB_TOKEN env)')
    ap.add_argument('--https', action='store_true', help='Use https remote with token (default). If not set, prints ssh alternative.')
    args = ap.parse_args()

    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        print('Error: Provide a GitHub token via --token or GITHUB_TOKEN env var (scopes: repo).')
        sys.exit(2)

    try:
        import requests  # noqa: F401
    except Exception:
        print('Installing requests...')
        run([sys.executable, '-m', 'pip', 'install', '-q', 'requests'])

    data = create_github_repo(token, args.full_name, private=args.private)
    # Prepare remote URL
    if args.https:
        remote = f"https://{token}@github.com/{data['full_name']}.git"
    else:
        # ssh option (requires your ssh keys configured)
        remote = f"git@github.com:{data['full_name']}.git"
        print('Info: using SSH remote; ensure your SSH key is configured with GitHub')
    push_repo(Path('.').resolve(), remote_url=remote, default_branch=args.branch)


if __name__ == '__main__':
    main()


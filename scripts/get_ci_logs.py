#!/usr/bin/env python3
"""Download GitHub Actions CI logs and grep for errors.

Auto-detects latest failed run, downloads build artifacts, and shows errors.

Usage:
  # Default: latest failed run, grep for Rust errors
  python3 scripts/get_ci_logs.py

  # Specific run
  python3 scripts/get_ci_logs.py --run 12345678

  # Show warnings too
  python3 scripts/get_ci_logs.py --warnings

  # Custom grep pattern
  python3 scripts/get_ci_logs.py --grep "unused"

  # Save artifacts zip
  python3 scripts/get_ci_logs.py --save-zip logs.zip
"""

import argparse
import getpass
import json
import os
import re
import sys
import tempfile
import zipfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError

OWNER = "99degree"
REPO = "MNN"


def get_token():
    """Get GitHub token from env or prompt."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        try:
            import getpass
            token = getpass.getpass("GitHub PAT: ")
        except Exception:
            pass
    if not token:
        print("❌ No GitHub token found. Set GITHUB_TOKEN or GH_TOKEN env var.", file=sys.stderr)
        sys.exit(1)
    return token


def api_get(url, token):
    """Make authenticated GET request to GitHub API."""
    req = Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode()
        print(f"❌ HTTP {e.code}: {body[:300]}", file=sys.stderr)
        sys.exit(1)


def api_download(url, token, output_path):
    """Download a file from GitHub API (follows redirects, Azure-compatible)."""
    # Use curl via subprocess for proper redirect handling across hosts
    import subprocess
    cmd = [
        "curl", "-sSfL", "--max-time", "30",
        "-H", f"Authorization: Bearer {token}",
        "-o", output_path,
        url
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed (exit {e.returncode}): {e.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
        return None


def get_latest_failed_run(token, branch="master"):
    """Get the latest failed workflow run on the given branch."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs?per_page=10&branch={branch}&status=completed"
    data = api_get(url, token)
    for run in data.get("workflow_runs", []):
        if run["conclusion"] in ("failure", "cancelled"):
            return run
    print("❌ No failed runs found.", file=sys.stderr)
    sys.exit(1)


def get_artifact(token, run_id, name_pattern="build-debug-logs"):
    """Find an artifact by name pattern in a workflow run."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}/artifacts"
    data = api_get(url, token)
    for art in data.get("artifacts", []):
        if name_pattern in art["name"]:
            return art
    return None


def download_artifact(token, artifact):
    """Download an artifact zip and return temp file path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    path = api_download(artifact["archive_download_url"], token, tmp.name)
    if not path:
        print(f"❌ Failed to download artifact {artifact['name']}", file=sys.stderr)
        sys.exit(1)
    return path


def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*[mK]', '', text)


def find_errors(content, show_warnings=False):
    """Extract Rust compilation errors with file locations and messages."""
    clean = strip_ansi(content)
    results = []

    # Find Rust errors - match error[XXXX] and surrounding context
    error_pattern = re.compile(
        r'(?:error|warning)(?:\[(?P<code>[A-Z0-9]+)\])?:(?P<msg>[^\n]*)'
    )

    lines = clean.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for error[XXXX]
        m = re.search(r'error\[([A-Z0-9]+)\]', line)
        if m:
            code = m.group(1)
            # Get file path and line number from next line
            file_path = ""
            line_no = 0
            msg = line.strip()
            for j in range(i, min(i + 5, len(lines))):
                fl = re.match(r'\s*-->\s+(.+?):(\d+):(\d+)', lines[j])
                if fl:
                    file_path = fl.group(1)
                    line_no = fl.group(2)
                    break
            
            # Collect error body (until next blank line or next error)
            body = []
            for j in range(i + 1, min(i + 20, len(lines))):
                if re.match(r'^\s*$', lines[j]) and body:
                    break
                if re.search(r'^(?:error|warning)\[', lines[j]):
                    break
                body.append(lines[j].strip())
            
            results.append({
                'type': 'error',
                'code': code,
                'file': file_path,
                'line': line_no,
                'message': msg,
                'body': '\n'.join(body[:10]),
            })
            i += 1
            continue
        
        # Check for warnings (if requested)
        if show_warnings:
            m = re.search(r'warning\[([A-Z0-9]+)\]', line)
            if m:
                code = m.group(1)
                file_path = ""
                for j in range(i, min(i + 5, len(lines))):
                    fl = re.match(r'\s*-->\s+(.+?):(\d+):(\d+)', lines[j])
                    if fl:
                        file_path = fl.group(1)
                        line_no = fl.group(2)
                        break
                
                results.append({
                    'type': 'warning',
                    'code': code,
                    'file': file_path,
                    'line': line_no,
                    'message': line.strip(),
                    'body': '',
                })
        
        # Also match "error: could not compile" lines
        m2 = re.search(r'error: could not compile `(\S+)`', line)
        if m2:
            results.append({
                'type': 'error',
                'code': 'BUILD_FAIL',
                'file': '',
                'line': '',
                'message': line.strip(),
                'body': f"Failed crate: {m2.group(1)}",
            })
        
        i += 1
    
    return results


def extract_errors_from_artifact(artifact_zip, show_warnings=False):
    """Extract error details from a downloaded artifact zip."""
    extract_dir = tempfile.mkdtemp(prefix="ci_errors_")
    with zipfile.ZipFile(artifact_zip, "r") as zf:
        zf.extractall(extract_dir)
    
    all_errors = {}
    for root, dirs, files in os.walk(extract_dir):
        for fn in sorted(files):
            fpath = os.path.join(root, fn)
            try:
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            errors = find_errors(content, show_warnings)
            if errors:
                # Deduplicate by (code, file, line)
                seen = set()
                unique = []
                for e in errors:
                    key = (e['code'], e['file'], e['line'])
                    if key not in seen:
                        seen.add(key)
                        unique.append(e)
                all_errors[fn] = unique
    
    return all_errors, extract_dir


def get_job_logs(token, run_id):
    """Get all job logs for a workflow run as a zip."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}/logs"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    path = api_download(url, token, tmp.name)
    if not path:
        print("❌ Failed to download logs", file=sys.stderr)
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser(description="CI error inspector for softisp")
    parser.add_argument("--run", type=int, help="Workflow run ID (default: latest failed)")
    parser.add_argument("--branch", default="master", help="Branch (default: master)")
    parser.add_argument("--grep", help="Custom grep pattern (default: show structured errors)")
    parser.add_argument("--warnings", action="store_true", help="Include warnings")
    parser.add_argument("--save-zip", help="Save downloaded zip to path")
    parser.add_argument("--artifact", default="build-debug-logs", help="Artifact name pattern")
    args = parser.parse_args()

    token = get_token()

    # Get the run
    if args.run:
        run_id = args.run
        run = api_get(f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}", token)
    else:
        run = get_latest_failed_run(token, args.branch)
        run_id = run["id"]
    
    sha = run["head_sha"][:8]
    msg = run.get("head_commit", {}).get("message", "")[:60]
    print(f"\n{'='*60}")
    print(f"  Run #{run_id} | {sha} | {run['conclusion']}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

    # If custom grep, use the full logs approach
    if args.grep:
        print(f"📥 Downloading full logs...")
        zip_path = get_job_logs(token, run_id)
        if args.save_zip:
            import shutil
            shutil.copy(zip_path, args.save_zip)
            print(f"   Saved to {args.save_zip}")
        
        extract_dir = tempfile.mkdtemp(prefix="ci_logs_")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        
        pat = re.compile(args.grep, re.IGNORECASE)
        print(f"🔍 Grepping for '{args.grep}'...\n")
        total = 0
        for fn in sorted(os.listdir(extract_dir)):
            fpath = os.path.join(extract_dir, fn)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue
            matches = []
            for i, line in enumerate(lines, 1):
                clean = strip_ansi(line)
                if pat.search(clean):
                    matches.append((i, clean.rstrip()))
            if matches:
                print(f"── {fn} ──")
                for ln, txt in matches:
                    print(f"  L{ln}: {txt}")
                total += len(matches)
                print()
        
        print(f"Total matches: {total}")
        os.unlink(zip_path)
        return

    # Default: find and parse artifact
    print(f"🔍 Looking for artifact '{args.artifact}'...")
    artifact = get_artifact(token, run_id, args.artifact)
    
    if artifact:
        print(f"   Found: {artifact['name']} ({artifact['size_in_bytes']}B)")
        print(f"📥 Downloading artifact...")
        zip_path = download_artifact(token, artifact)
        
        if args.save_zip:
            import shutil
            shutil.copy(zip_path, args.save_zip)
        
        print(f"🔍 Parsing errors...\n")
        errors_by_file, _ = extract_errors_from_artifact(zip_path, args.warnings)
        
        total_errors = 0
        total_warnings = 0
        for fn, errors in sorted(errors_by_file.items()):
            errs = [e for e in errors if e['type'] == 'error']
            warns = [e for e in errors if e['type'] == 'warning']
            if not errs and not warns:
                continue
            
            print(f"╔══ {fn} ══╗")
            
            for e in errs:
                total_errors += 1
                loc = f"{e['file']}:{e['line']}" if e['file'] else ""
                print(f"  ❌ [{e['code']}] {e['message']}")
                if loc:
                    print(f"     at {loc}")
                if e['body']:
                    print(f"     {e['body']}")
                print()
            
            for w in warns:
                total_warnings += 1
                loc = f"{w['file']}:{w['line']}" if w['file'] else ""
                print(f"  ⚠️  [{w['code']}] {w['message']}")
                if loc:
                    print(f"     at {loc}")
                print()
        
        summary = f"\n{'='*60}\n"
        if total_errors > 0:
            summary += f"  ❌ {total_errors} error(s)"
        if total_warnings > 0:
            summary += f"  ⚠️  {total_warnings} warning(s)"
        if total_errors == 0 and total_warnings == 0:
            summary += "  ✅ No errors or warnings found in artifact!"
        summary += f"\n{'='*60}\n"
        print(summary)
        
        os.unlink(zip_path)
    else:
        print(f"   No artifact '{args.artifact}' found for run #{run_id}")
        print(f"   Falling back to full log download...")
        zip_path = get_job_logs(token, run_id)
        extract_dir = tempfile.mkdtemp(prefix="ci_logs_")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        
        # Find files with errors
        for fn in sorted(os.listdir(extract_dir)):
            fpath = os.path.join(extract_dir, fn)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            clean = strip_ansi(content)
            if "error[" in clean:
                errors = find_errors(content, args.warnings)
                if errors:
                    print(f"\n── {fn} ──")
                    for e in errors:
                        loc = f"{e['file']}:{e['line']}" if e['file'] else ""
                        icon = "❌" if e['type'] == 'error' else "⚠️"
                        print(f"  {icon} [{e['code']}] {e['message']}")
                        if loc:
                            print(f"     at {loc}")
                        if e['body']:
                            print(f"     {e['body']}")
                        print()


if __name__ == "__main__":
    main()

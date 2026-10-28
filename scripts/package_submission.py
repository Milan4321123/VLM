"""Create a clean source + report ZIP, without local weights, secrets or caches."""
from pathlib import Path
import argparse
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Explicit allowlist keeps unrelated untracked files and private local state out.
FILES = ['README.md', 'Dockerfile', 'docker-compose.yml', 'docker-compose.gpu.yml',
         '.dockerignore', '.gitignore', 'cli.py', 'inference.py']
DIRECTORIES = ['vlm_fo1', 'detect_tools', 'demo', 'docker', 'assets', 'docs', 'scripts', 'tests']
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.cache', '.DS_Store'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.so', '.safetensors', '.pth', '.bin', '.ckpt', '.log', '.mp4'}
NOTICE_PATTERNS = ('LICENSE*', 'LICENCE*', 'NOTICE*', 'COPYING*', 'AUTHORS*')
REQUIRED_FILES = FILES + ['docker/entrypoint.sh', 'requirements.txt', 'requirements-gpu.txt']


def package(output=None, root=ROOT):
    root = Path(root).resolve()
    output = Path(output) if output is not None else root / 'output/submission/vlm-fo1-qwen35-submission.zip'
    report = root / 'output/pdf/project_report.pdf'
    if not report.is_file():
        raise SystemExit('Build the report first: python scripts/build_report.py')
    missing = [name for name in REQUIRED_FILES
               if not (root / name).is_file() or (root / name).is_symlink()]
    if missing:
        raise SystemExit('Missing submission files: ' + ', '.join(missing))
    paths = [root / name for name in FILES]
    paths.extend(root.glob('requirements*.txt'))
    for pattern in NOTICE_PATTERNS:
        paths.extend(root.glob(pattern))
    for folder in DIRECTORIES:
        paths.extend((root / folder).rglob('*'))
    paths.append(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part in SKIP_PARTS or part == '.env' or part.startswith('.env.')
                   for part in relative.parts) or path.suffix in SKIP_SUFFIXES:
                continue
            archive.write(path, Path('vlm-fo1-qwen35') / relative)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
    print(f'{output}\nSHA256 {digest}\nSize {output.stat().st_size / 1024**2:.1f} MiB')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the ZIP to this path instead of output/submission/.')
    package(parser.parse_args().output)

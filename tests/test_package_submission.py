"""The extracted handover must include its container entrypoint and source notices."""
import hashlib
import zipfile

import pytest

from scripts.package_submission import REQUIRED_FILES, package


def make_source(tmp_path):
    source = tmp_path / 'source'
    for name in REQUIRED_FILES + ['output/pdf/project_report.pdf', 'LICENSE',
                                  'detect_tools/upn/ops/setup.py',
                                  'tests/fixtures/fo1/model.safetensors.index.json']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    return source


def test_archive_preserves_build_inputs_and_excludes_local_state(tmp_path):
    source = make_source(tmp_path)
    for name in ['docker/.env', 'docker/.env.local', 'demo/local.pth',
                 'demo/__pycache__/module.pyc']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('local state')
    output = package(tmp_path / 'review.zip', root=source)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        archive.extractall(tmp_path / 'extracted')
    extracted = tmp_path / 'extracted/vlm-fo1-qwen35'
    assert (extracted / 'docker/entrypoint.sh').is_file()
    assert (extracted / 'detect_tools/upn/ops/setup.py').is_file()
    assert (extracted / 'LICENSE').is_file()
    assert (extracted / 'tests/fixtures/fo1/model.safetensors.index.json').is_file()
    assert not any('.env' in name or '__pycache__' in name or name.endswith('.pth') for name in names)
    assert output.with_suffix('.zip.sha256').read_text().split()[0] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_missing_entrypoint_fails_before_creating_archive(tmp_path):
    source = make_source(tmp_path)
    (source / 'docker/entrypoint.sh').unlink()
    output = tmp_path / 'review.zip'
    with pytest.raises(SystemExit, match='docker/entrypoint.sh'):
        package(output, root=source)
    assert not output.exists()

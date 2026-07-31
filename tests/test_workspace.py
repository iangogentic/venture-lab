"""The workspace tree is the artifact store of record; its shape is load-bearing."""

from app.utils.paths import STAGE_DIRECTORIES, WorkspacePaths, get_workspace_paths


def test_ensure_creates_every_stage_directory() -> None:
    paths = get_workspace_paths()
    paths.ensure()

    for directory in paths.all_dirs():
        assert directory.is_dir()


def test_ensure_is_idempotent(workspace: WorkspacePaths) -> None:
    workspace.ensure()
    assert workspace.root.is_dir()


def test_stage_directories_match_attributes(workspace: WorkspacePaths) -> None:
    """`STAGE_DIRECTORIES` and the dataclass fields must not drift apart."""
    named = {directory.name for directory in workspace.all_dirs()}
    assert named == set(STAGE_DIRECTORIES)

    for name in STAGE_DIRECTORIES:
        assert getattr(workspace, name) == workspace.for_directory(name)


def test_paths_are_absolute(workspace: WorkspacePaths) -> None:
    assert workspace.root.is_absolute()

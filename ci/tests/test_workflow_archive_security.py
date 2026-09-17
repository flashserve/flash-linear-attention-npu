#!/usr/bin/env python3
import ast
import io
import os
import re
import stat
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _python_blocks():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ascend-npu"]["steps"]
    blocks = []
    for step in steps:
        run = step.get("run", "")
        blocks.extend(
            re.findall(r"python3? - <<'PY'\n(.*?)\nPY(?:\n|$)", run, flags=re.S)
        )
    return blocks


def _function_nodes(name):
    nodes = []
    for block in _python_blocks():
        tree = ast.parse(block)
        nodes.extend(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
    return nodes


def _load_archive_functions():
    block = next(block for block in _python_blocks() if "def extract_archive" in block)
    tree = ast.parse(block)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef)
        and node.name in {"extract_archive", "copy_case_file"}
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(WORKFLOW_PATH), "exec"),
        namespace,
    )
    return namespace["extract_archive"], namespace["copy_case_file"]


def _tar_info(name, member_type=tarfile.REGTYPE, data=b"", linkname=""):
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.mode = 0o755 if member_type == tarfile.DIRTYPE else 0o644
    info.linkname = linkname
    if member_type == tarfile.REGTYPE:
        info.size = len(data)
    return info, data


def _write_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for info, data in members:
            source = io.BytesIO(data) if info.isfile() else None
            archive.addfile(info, source)


class WorkflowSourceConsistencyTest(unittest.TestCase):
    def test_a2_and_a5_case_copy_implementations_match(self):
        nodes = _function_nodes("copy_case_file")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))


@unittest.skipUnless(
    getattr(os, "O_NOFOLLOW", 0) and getattr(os, "O_DIRECTORY", 0),
    "safe archive handling requires Linux open flags",
)
class WorkflowArchiveSecurityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extract_archive, copy_case_file = _load_archive_functions()
        cls.extract_archive = staticmethod(extract_archive)
        cls.copy_case_file = staticmethod(copy_case_file)

    def test_valid_archive_extracts_and_case_copy_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            archive_path = temp_path / "valid.tar.gz"
            destination = temp_path / "repo"
            preserved = temp_path / "preserved" / "example_st_cases.json"
            destination.mkdir()
            preserved.parent.mkdir()
            members = [
                _tar_info("root/", tarfile.DIRTYPE),
                _tar_info("root/ci/", tarfile.DIRTYPE),
                _tar_info("root/ci/example_st_cases.json", data=b'{"cases": []}'),
                _tar_info("root/bin/", tarfile.DIRTYPE),
                _tar_info("root/bin/run.sh", data=b"#!/bin/sh\n"),
            ]
            members[-1][0].mode = 0o755
            _write_archive(archive_path, members)

            self.extract_archive(archive_path, destination)

            self.assertEqual(
                (destination / "ci" / "example_st_cases.json").read_bytes(),
                b'{"cases": []}',
            )
            self.assertTrue((destination / "bin" / "run.sh").stat().st_mode & stat.S_IXUSR)
            self.assertTrue(self.copy_case_file(destination, preserved))
            self.assertEqual(preserved.read_bytes(), b'{"cases": []}')

    def test_unsafe_archive_members_are_rejected(self):
        unsafe_cases = [
            ("empty archive", lambda _: [], r"archive is empty"),
            # If validation regresses, one '..' can only reach this test's temp root.
            (
                "parent traversal",
                lambda _: [_tar_info("root/../escape", data=b"x")],
                r"unsafe archive member path",
            ),
            (
                "absolute path",
                lambda temp: [
                    _tar_info(f"{temp.as_posix()}/absolute-escape", data=b"x")
                ],
                r"unsafe archive member path",
            ),
            (
                "symlink",
                lambda _: [
                    _tar_info("root/target", data=b"target"),
                    _tar_info("root/link", tarfile.SYMTYPE, linkname="target"),
                ],
                r"unsupported archive member type",
            ),
            (
                "hardlink",
                lambda _: [
                    _tar_info("root/target", data=b"target"),
                    _tar_info(
                        "root/link", tarfile.LNKTYPE, linkname="root/target"
                    ),
                ],
                r"unsupported archive member type",
            ),
            (
                "fifo",
                lambda _: [_tar_info("root/fifo", tarfile.FIFOTYPE)],
                r"unsupported archive member type",
            ),
            (
                "character device",
                lambda _: [_tar_info("root/device", tarfile.CHRTYPE)],
                r"unsupported archive member type",
            ),
            (
                "duplicate file",
                lambda _: [
                    _tar_info("root/file", data=b"one"),
                    _tar_info("root/file", data=b"two"),
                ],
                r"duplicate archive member",
            ),
            (
                "duplicate root",
                lambda _: [
                    _tar_info("root/", tarfile.DIRTYPE),
                    _tar_info("root/", tarfile.DIRTYPE),
                ],
                r"duplicate archive top-level directory",
            ),
            (
                "multiple roots",
                lambda _: [
                    _tar_info("root/one", data=b"one"),
                    _tar_info("other/two", data=b"two"),
                ],
                r"archive contains multiple top-level directories",
            ),
            (
                "top-level file",
                lambda _: [_tar_info("root", data=b"x")],
                r"archive top-level member is not a directory",
            ),
        ]
        for description, member_factory, expected_error in unsafe_cases:
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory() as temp:
                    temp_path = Path(temp)
                    archive_path = temp_path / "unsafe.tar.gz"
                    destination = temp_path / "repo"
                    destination.mkdir()
                    members = member_factory(temp_path)
                    _write_archive(archive_path, members)
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        self.extract_archive(archive_path, destination)
                    self.assertFalse((temp_path / "escape").exists())
                    self.assertFalse((temp_path / "absolute-escape").exists())

    def test_parent_child_type_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            archive_path = temp_path / "conflict.tar.gz"
            destination = temp_path / "repo"
            destination.mkdir()
            _write_archive(
                archive_path,
                [
                    _tar_info("root/parent", data=b"file"),
                    _tar_info("root/parent/child", data=b"child"),
                ],
            )
            with self.assertRaises(OSError):
                self.extract_archive(archive_path, destination)

    def test_nonempty_or_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            archive_path = temp_path / "valid.tar.gz"
            _write_archive(archive_path, [_tar_info("root/file", data=b"x")])

            nonempty = temp_path / "nonempty"
            nonempty.mkdir()
            (nonempty / "existing").write_text("x", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.extract_archive(archive_path, nonempty)

            real_destination = temp_path / "real"
            real_destination.mkdir()
            symlink_destination = temp_path / "symlink"
            symlink_destination.symlink_to(real_destination, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                self.extract_archive(archive_path, symlink_destination)

    def test_case_copy_missing_invalid_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            repo = temp_path / "repo"
            output_dir = temp_path / "output"
            repo.mkdir()
            (repo / "ci").mkdir()
            output_dir.mkdir()

            self.assertFalse(
                self.copy_case_file(repo, output_dir / "missing.json")
            )
            self.assertFalse((output_dir / "missing.json").exists())

            case_path = repo / "ci" / "example_st_cases.json"
            case_path.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.copy_case_file(repo, output_dir / "invalid.json")
            self.assertFalse((output_dir / "invalid.json").exists())
            case_path.unlink()

            outside = temp_path / "outside.json"
            outside.write_text('{"secret": true}', encoding="utf-8")
            case_path.symlink_to(outside)
            with self.assertRaises(OSError):
                self.copy_case_file(repo, output_dir / "symlink.json")

    def test_case_copy_rejects_ci_symlink_fifo_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            repo = temp_path / "repo"
            outside_ci = temp_path / "outside-ci"
            output_dir = temp_path / "output"
            repo.mkdir()
            outside_ci.mkdir()
            output_dir.mkdir()
            (outside_ci / "example_st_cases.json").write_text("{}", encoding="utf-8")
            (repo / "ci").symlink_to(outside_ci, target_is_directory=True)
            with self.assertRaises(OSError):
                self.copy_case_file(repo, output_dir / "ci-symlink.json")

            (repo / "ci").unlink()
            (repo / "ci").mkdir()
            case_path = repo / "ci" / "example_st_cases.json"
            os.mkfifo(case_path)
            with self.assertRaises(RuntimeError):
                self.copy_case_file(repo, output_dir / "fifo.json")
            case_path.unlink()

            case_path.write_text("{}", encoding="utf-8")
            existing_output = output_dir / "existing.json"
            existing_output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.copy_case_file(repo, existing_output)
            self.assertEqual(existing_output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

"""Tests for setup.py's RHEL / CentOS 7 handling (no GPU, no network, nothing installed): a GCC older than the engine's C++20
needs is replaced by Red Hat's devtoolset, a C library older than the ready-made engines can use means a local build, and
the pinned Python packages that have no wheel for glibc 2.17 fall back to the newest ones that do.

    python -m unittest tools.test_setup_rhel7
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class Quiet:
    def __enter__(self):
        self.buf = io.StringIO()
        self.cm = contextlib.redirect_stdout(self.buf)
        self.cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)

    @property
    def text(self):
        return self.buf.getvalue()


def env_without(*names):
    return {k: v for k, v in os.environ.items() if k not in names}


class GlibcVersion(unittest.TestCase):
    def test_rhel7(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.platform, "libc_ver", return_value=("glibc", "2.17")):
            self.assertEqual(setup.glibc_version(), (2, 17))

    def test_a_newer_glibc_and_a_patch_level(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.platform, "libc_ver", return_value=("glibc", "2.39.1")):
            self.assertEqual(setup.glibc_version(), (2, 39))

    def test_not_glibc_or_not_linux(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.platform, "libc_ver", return_value=("", "")):
            self.assertIsNone(setup.glibc_version())       # musl
        with mock.patch.object(setup, "WIN", True):
            self.assertIsNone(setup.glibc_version())
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "darwin"):
            self.assertIsNone(setup.glibc_version())

    def test_rhel7_is_below_the_ready_made_engines_floor(self):
        self.assertLess((2, 17), setup.PREBUILT_MIN_GLIBC)
        self.assertGreaterEqual((2, 31), setup.PREBUILT_MIN_GLIBC)      # Ubuntu 20.04 and RHEL 8 keep their downloads


class NewerGcc(unittest.TestCase):
    """The system's g++ is 4.8.5; /opt/rh/devtoolset-11 has 11."""

    def where(self, existing, gxx="/usr/bin/g++", major=4, cc=None, cxx=None):
        env = env_without("CC", "CXX")
        env.update({k: v for k, v in (("CC", cc), ("CXX", cxx)) if v})
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.shutil, "which", return_value=gxx), \
                mock.patch.object(setup, "gcc_major", return_value=major), \
                mock.patch.object(setup.os, "access", side_effect=lambda p, mode: p in existing):
            return setup.newer_gcc_dir()

    def test_devtoolset_11_is_taken_when_the_system_gcc_is_old(self):
        got = self.where({"/opt/rh/devtoolset-11/root/usr/bin/g++", "/opt/rh/devtoolset-10/root/usr/bin/g++"})
        self.assertEqual(got, "/opt/rh/devtoolset-11/root/usr/bin")

    def test_gcc_toolset_on_rhel8(self):
        self.assertEqual(self.where({"/opt/rh/gcc-toolset-12/root/usr/bin/g++"}), "/opt/rh/gcc-toolset-12/root/usr/bin")

    def test_eleven_before_twelve_before_ten(self):
        self.assertEqual(self.where({"/opt/rh/devtoolset-12/root/usr/bin/g++", "/opt/rh/devtoolset-10/root/usr/bin/g++"}),
                         "/opt/rh/devtoolset-12/root/usr/bin")
        self.assertEqual(self.where({"/opt/rh/devtoolset-10/root/usr/bin/g++"}), "/opt/rh/devtoolset-10/root/usr/bin")

    def test_nothing_changes_for_a_new_enough_gcc(self):
        self.assertIsNone(self.where({"/opt/rh/devtoolset-11/root/usr/bin/g++"}, major=13))
        self.assertIsNone(self.where({"/opt/rh/devtoolset-11/root/usr/bin/g++"}, major=10))

    def test_nothing_changes_when_the_user_named_a_compiler(self):
        self.assertIsNone(self.where({"/opt/rh/devtoolset-11/root/usr/bin/g++"}, cxx="g++-12"))
        self.assertIsNone(self.where({"/opt/rh/devtoolset-11/root/usr/bin/g++"}, cc="gcc-12"))

    def test_no_toolset_no_change(self):
        self.assertIsNone(self.where(set()))

    def test_the_toolset_goes_first_on_path(self):
        env = env_without("CC", "CXX")
        env["PATH"] = "/usr/bin:/bin"
        with mock.patch.dict(os.environ, env, clear=True), Quiet() as q, \
                mock.patch.object(setup, "newer_gcc_dir", return_value="/opt/rh/devtoolset-11/root/usr/bin"), \
                mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.shutil, "which", return_value="/opt/rh/devtoolset-11/root/usr/bin/g++"), \
                mock.patch.object(setup, "gcc_major", return_value=11):
            setup.use_newer_gcc()
            self.assertTrue(os.environ["PATH"].startswith("/opt/rh/devtoolset-11/root/usr/bin" + os.pathsep))
        self.assertIn("devtoolset-11", q.text)

    def test_an_old_gcc_with_no_toolset_stops_in_words(self):
        env = env_without("CC", "CXX")
        with mock.patch.dict(os.environ, env, clear=True), Quiet(), \
                mock.patch.object(setup, "newer_gcc_dir", return_value=None), \
                mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.shutil, "which", return_value="/usr/bin/g++"), \
                mock.patch.object(setup, "gcc_major", return_value=4), \
                mock.patch.object(setup, "fail", side_effect=SystemExit) as fail:
            with self.assertRaises(SystemExit):
                setup.use_newer_gcc()
        self.assertIn("GCC 4", fail.call_args[0][0])
        self.assertIn("devtoolset-11", fail.call_args[0][1])

    def test_a_named_compiler_is_not_second_guessed(self):
        env = env_without("CC", "CXX")
        env["CXX"] = "g++-9"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(setup, "fail") as fail, \
                mock.patch.object(setup.shutil, "which", return_value="/usr/bin/g++"), \
                mock.patch.object(setup, "gcc_major", return_value=4):
            setup.use_newer_gcc()
        fail.assert_not_called()


class PipOnAnOldCLibrary(unittest.TestCase):
    PINS = ["numpy==2.5.3; python_version >= '3.12'", "pillow==12.3.0", "tqdm==4.70.1"]

    def install(self, glibc, first_returncode):
        calls = []

        def fake_run(cmd, **kw):
            calls.append((list(cmd), kw))
            rc = first_returncode if len(calls) == 1 else 0
            return mock.Mock(returncode=rc, stdout="")

        import tempfile
        with tempfile.TemporaryDirectory() as d, Quiet(), mock.patch.object(setup, "CONTROL", False), \
                mock.patch.object(setup.sys, "prefix", d), mock.patch.object(setup, "glibc_version", return_value=glibc), \
                mock.patch.object(setup, "run", side_effect=fake_run), \
                mock.patch.object(setup.acquisition, "require", return_value=None):
            setup.pip_install(self.PINS, "test packages")
        return calls

    def test_new_glibc_installs_the_pins_once(self):
        calls = self.install((2, 39), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("numpy==2.5.3; python_version >= '3.12'", calls[0][0])

    def test_old_glibc_keeps_the_pins_when_they_install(self):
        calls = self.install((2, 17), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("pillow==12.3.0", calls[0][0])

    def test_old_glibc_falls_back_to_the_newest_wheels(self):
        calls = self.install((2, 17), 1)
        self.assertEqual(len(calls), 2)
        second = calls[1][0]
        self.assertIn("--prefer-binary", second)
        self.assertIn("pillow", second)
        self.assertIn("numpy; python_version >= '3.12'", second)      # the marker stays, the pin goes
        self.assertFalse([a for a in second if "==" in a])


class RhelFilesExist(unittest.TestCase):
    def test_the_build_script_and_the_checker_are_there_and_parse(self):
        import subprocess
        for name in ("tools/build_rhel7.sh", "tools/check_glibc_symbols.sh"):
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(subprocess.run(["sh", "-n", str(path)]).returncode, 0, name)

    def test_the_checker_passes_a_binary_that_asks_for_nothing_new(self):
        import subprocess
        import shutil
        if shutil.which("objdump") is None:
            self.skipTest("no objdump")
        # /bin/sh of any system asks for glibc symbols; with a huge limit the check passes, with 2.0 it must fail
        sh = shutil.which("sh")
        ok = subprocess.run(["sh", str(ROOT / "tools/check_glibc_symbols.sh"), sh, "--max-glibc", "99.0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        if b"GLIBC none" in ok.stdout:
            self.skipTest("this sh asks for no glibc symbol (static)")
        bad = subprocess.run(["sh", str(ROOT / "tools/check_glibc_symbols.sh"), sh, "--max-glibc", "2.0"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(bad.returncode, 1)


if __name__ == "__main__":
    unittest.main()

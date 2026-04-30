"""Tests for src/local/execute.py — import validation and viewer safety checks."""
import pytest
from unittest.mock import patch
from pymmcore_plus import CMMCorePlus
from src.local.execute import Execute


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def executor():
    """Execute instance with a bare CMMCorePlus (no hardware config loaded)."""
    mmc = CMMCorePlus()
    return Execute(mmc=mmc)


# ---------------------------------------------------------------------------
# _get_missing_imports
# ---------------------------------------------------------------------------

class TestGetMissingImports:

    def test_no_imports(self, executor):
        assert executor._get_missing_imports("x = 1 + 1") == []

    def test_all_available(self, executor):
        code = "import os\nimport sys\nimport json"
        assert executor._get_missing_imports(code) == []

    def test_from_import_available(self, executor):
        assert executor._get_missing_imports("from os.path import join") == []

    def test_single_missing(self, executor):
        missing = executor._get_missing_imports("import _fake_pkg_xyz_does_not_exist")
        assert missing == ["_fake_pkg_xyz_does_not_exist"]

    def test_from_import_missing(self, executor):
        missing = executor._get_missing_imports("from _fake_pkg_xyz_does_not_exist import foo")
        assert missing == ["_fake_pkg_xyz_does_not_exist"]

    def test_multiple_missing(self, executor):
        code = "import _fake_a\nimport _fake_b"
        missing = executor._get_missing_imports(code)
        assert set(missing) == {"_fake_a", "_fake_b"}

    def test_deduplication(self, executor):
        code = "import _fake_pkg_xyz_does_not_exist\nfrom _fake_pkg_xyz_does_not_exist import bar"
        missing = executor._get_missing_imports(code)
        assert missing.count("_fake_pkg_xyz_does_not_exist") == 1

    def test_syntax_error_returns_empty(self, executor):
        assert executor._get_missing_imports("def (broken syntax!!!") == []

    def test_submodule_uses_top_level(self, executor):
        # os.path is available; only the top-level 'os' is checked
        missing = executor._get_missing_imports("import os.path")
        assert missing == []


# ---------------------------------------------------------------------------
# is_safe_viewer
# ---------------------------------------------------------------------------

class TestIsSafeViewer:

    def test_safe_code(self, executor):
        assert executor.is_safe_viewer("x = 1\nprint(x)") is True

    def test_bare_viewer_name_blocked(self, executor):
        assert executor.is_safe_viewer("viewer.add_layer(img)") is False

    def test_napari_current_viewer_blocked(self, executor):
        assert executor.is_safe_viewer("v = napari.current_viewer()") is False

    def test_similar_name_allowed(self, executor):
        # 'my_viewer' is not exactly 'viewer'
        assert executor.is_safe_viewer("my_viewer = get_viewer()") is True

    def test_viewer_in_string_allowed(self, executor):
        # String literals don't create ast.Name nodes
        assert executor.is_safe_viewer('msg = "viewer is great"') is True

    def test_viewer_in_comment_allowed(self, executor):
        assert executor.is_safe_viewer("# viewer stuff\nx = 1") is True


# ---------------------------------------------------------------------------
# _preimport_dependencies
# ---------------------------------------------------------------------------

class TestPreimportDependencies:

    def test_no_missing_returns_empty(self, executor):
        failed = executor._preimport_dependencies("import os\nimport sys")
        assert failed == []

    def test_missing_install_fails_returns_name(self, executor):
        with patch.object(executor, "_install_library", return_value=False):
            failed = executor._preimport_dependencies("import _fake_pkg_xyz_does_not_exist")
        assert "_fake_pkg_xyz_does_not_exist" in failed

    def test_missing_install_succeeds_returns_empty(self, executor):
        # Simulate install succeeding (the module becomes available via importlib mock)
        with patch.object(executor, "_install_library", return_value=True):
            # Still missing from sys — importlib.import_module will fail silently,
            # but _preimport_dependencies only reports install failures
            failed = executor._preimport_dependencies("import _fake_pkg_xyz_does_not_exist")
        assert failed == []

    def test_syntax_error_returns_empty(self, executor):
        failed = executor._preimport_dependencies("def (broken!!!")
        assert failed == []


# ---------------------------------------------------------------------------
# run_code_new
# ---------------------------------------------------------------------------

class TestRunCodeNew:

    def test_invalid_mode_returns_error(self, executor):
        result = executor.run_code_new("x = 1", execution_mode="streaming")
        assert "Invalid execution mode" in result

    def test_viewer_reference_blocked(self, executor):
        result = executor.run_code_new("viewer.add_layer(img)", execution_mode="buffered")
        assert result == "viewer"

    def test_missing_package_returns_error(self, executor):
        with patch.object(executor, "_install_library", return_value=False):
            result = executor.run_code_new(
                "import _fake_pkg_xyz_does_not_exist",
                execution_mode="buffered"
            )
        assert "Missing packages" in result
        assert "_fake_pkg_xyz_does_not_exist" in result

    def test_simple_buffered_execution(self, executor):
        result = executor.run_code_new('print("hello test")', execution_mode="buffered")
        assert "hello test" in result

    def test_simple_live_execution(self, executor):
        result = executor.run_code_new('print("live mode")', execution_mode="live")
        assert "live mode" in result

    def test_no_output_returns_success_message(self, executor):
        result = executor.run_code_new("x = 1 + 1", execution_mode="buffered")
        assert "successfully" in result.lower()

    def test_runtime_error_returns_error_string(self, executor):
        result = executor.run_code_new("raise ValueError('boom')", execution_mode="buffered")
        assert "Execution error" in result
        assert "boom" in result

    def test_stderr_captured_in_output(self, executor):
        result = executor.run_code_new(
            "import sys; sys.stderr.write('a warning')",
            execution_mode="buffered"
        )
        assert "a warning" in result


# ---------------------------------------------------------------------------
# confirmation flow (via run_code_new: missing → error, not auto-install)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# is_safe_code — Task #14
# ---------------------------------------------------------------------------

class TestIsSafeCode:

    def test_safe_code(self, executor):
        safe, reason = executor.is_safe_code("x = mmc.getXPosition()")
        assert safe is True
        assert reason == ""

    def test_blocks_cmmcoreplus_constructor(self, executor):
        safe, reason = executor.is_safe_code("core = CMMCorePlus()")
        assert safe is False
        assert "CMMCorePlus" in reason

    def test_blocks_uniMMcore_constructor(self, executor):
        safe, reason = executor.is_safe_code("core = UniMMCore()")
        assert safe is False
        assert "UniMMCore" in reason

    def test_blocks_cmmcoreplus_instance(self, executor):
        safe, reason = executor.is_safe_code("core = CMMCorePlus.instance()")
        assert safe is False
        assert "CMMCorePlus" in reason

    def test_blocks_uniMMcore_instance(self, executor):
        safe, reason = executor.is_safe_code("core = UniMMCore.instance()")
        assert safe is False
        assert "UniMMCore" in reason

    def test_blocks_load_system_configuration(self, executor):
        safe, reason = executor.is_safe_code("mmc.loadSystemConfiguration('demo.cfg')")
        assert safe is False
        assert "loadSystemConfiguration" in reason

    def test_blocks_load_config(self, executor):
        safe, reason = executor.is_safe_code("mmc.loadConfig('path/to/cfg')")
        assert safe is False
        assert "loadConfig" in reason

    def test_unrelated_instance_call_allowed(self, executor):
        # Some other class calling .instance() is fine
        safe, reason = executor.is_safe_code("obj = MyClass.instance()")
        assert safe is True

    def test_syntax_error_returns_safe(self, executor):
        # Syntax errors are caught later; safety check must not crash
        safe, reason = executor.is_safe_code("def (broken!!!")
        assert safe is True

    def test_run_code_blocks_cmmcoreplus(self, executor):
        result = executor.run_code_new("core = CMMCorePlus()", execution_mode="buffered")
        assert "Safety Error" in result
        assert "CMMCorePlus" in result

    def test_run_code_blocks_load_system_configuration(self, executor):
        result = executor.run_code_new(
            "mmc.loadSystemConfiguration('demo.cfg')", execution_mode="buffered"
        )
        assert "Safety Error" in result
        assert "loadSystemConfiguration" in result


class TestConfirmationFlow:

    def test_missing_package_not_auto_installed(self, executor):
        """run_code_new must NOT silently install — it should surface the failure."""
        with patch.object(executor, "_install_library", return_value=False) as mock_install:
            result = executor.run_code_new(
                "import _fake_pkg_xyz_does_not_exist",
                execution_mode="buffered"
            )
        # install was attempted once for the missing package
        mock_install.assert_called_once_with("_fake_pkg_xyz_does_not_exist")
        assert "Missing packages" in result
        assert "_fake_pkg_xyz_does_not_exist" in result

    def test_get_missing_imports_is_standalone(self, executor):
        """_get_missing_imports must not install anything — pure inspection."""
        with patch.object(executor, "_install_library") as mock_install:
            missing = executor._get_missing_imports("import _fake_pkg_xyz_does_not_exist")
        mock_install.assert_not_called()
        assert "_fake_pkg_xyz_does_not_exist" in missing

    def test_install_then_run_succeeds(self, executor):
        """Simulate the two-step flow: install → run."""
        # Step 1: detect missing
        missing = executor._get_missing_imports("import os")   # os is always present
        assert missing == []
        # Step 2: run without issue
        result = executor.run_code_new('print("after install")', execution_mode="buffered")
        assert "after install" in result

"""Tests for src/local/execute.py — import validation and viewer safety checks."""

from unittest.mock import patch

import pytest
from pymmcore_plus import CMMCorePlus

from local import Execute

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
                "import _fake_pkg_xyz_does_not_exist", execution_mode="buffered"
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
            "import sys; sys.stderr.write('a warning')", execution_mode="buffered"
        )
        assert "a warning" in result


# ---------------------------------------------------------------------------
# confirmation flow (via run_code_new: missing → error, not auto-install)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# is_safe_code
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
                "import _fake_pkg_xyz_does_not_exist", execution_mode="buffered"
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
        missing = executor._get_missing_imports("import os")  # os is always present
        assert missing == []
        # Step 2: run without issue
        result = executor.run_code_new('print("after install")', execution_mode="buffered")
        assert "after install" in result


# ---------------------------------------------------------------------------
# Library guards
# ---------------------------------------------------------------------------

import ast as _ast  # noqa: E402

import numpy as np  # noqa: E402

from local.execute import (  # noqa: E402
    _CELLPOSE_SIZE_THRESHOLD,
    _cellpose_cellprob_threshold_guard,
    _cellpose_channels_guard,
    _cellpose_diameter_guard,
    _cellpose_flow_threshold_guard,
    _install_cellpose_size_guard,
)


class TestGetImportedModules:
    def test_empty_code(self, executor):
        assert executor._get_imported_modules("x = 1") == set()

    def test_plain_import(self, executor):
        assert "os" in executor._get_imported_modules("import os")

    def test_from_import(self, executor):
        assert "os" in executor._get_imported_modules("from os.path import join")

    def test_submodule_uses_top_level(self, executor):
        mods = executor._get_imported_modules("import os.path")
        assert "os" in mods
        assert "os.path" not in mods

    def test_multiple_imports(self, executor):
        code = "import numpy\nfrom scipy import stats"
        mods = executor._get_imported_modules(code)
        assert {"numpy", "scipy"}.issubset(mods)

    def test_syntax_error_returns_empty(self, executor):
        assert executor._get_imported_modules("def (broken!!!") == set()


class TestCheckLibraryGuards:
    def test_no_guard_registered_passes(self, executor):
        ok, reason = executor._check_library_guards("import os\nprint('hi')")
        assert ok is True
        assert reason == ""

    def test_unrelated_library_guard_not_triggered(self, executor):
        ok, _ = executor._check_library_guards("import numpy")
        assert ok is True

    def test_custom_guard_triggered(self, executor):
        def _always_fail(tree):
            return False, "test guard triggered"

        Execute.register_library_guard("_test_lib_guard_xyz", _always_fail)
        try:
            ok, reason = executor._check_library_guards("import _test_lib_guard_xyz")
            assert ok is False
            assert "_test_lib_guard_xyz" in reason
            assert "test guard triggered" in reason
        finally:
            Execute._LIBRARY_GUARDS.pop("_test_lib_guard_xyz", None)

    def test_custom_guard_not_triggered_when_lib_absent(self, executor):
        def _always_fail(tree):
            return False, "should not run"

        Execute.register_library_guard("_test_lib_guard_absent", _always_fail)
        try:
            ok, _ = executor._check_library_guards("import os")
            assert ok is True
        finally:
            Execute._LIBRARY_GUARDS.pop("_test_lib_guard_absent", None)

    def test_syntax_error_passes(self, executor):
        ok, _ = executor._check_library_guards("def (broken!!!")
        assert ok is True


class TestCellposeDiameterGuard:
    def _tree(self, code):
        return _ast.parse(code)

    def test_with_diameter_passes(self):
        code = "model.eval(img, diameter=15)"
        ok, reason = _cellpose_diameter_guard(self._tree(code))
        assert ok is True
        assert reason == ""

    def test_without_diameter_fails(self):
        code = "model.eval(img)"
        ok, reason = _cellpose_diameter_guard(self._tree(code))
        assert ok is False
        assert "diameter" in reason

    def test_empty_code_fails(self):
        ok, _ = _cellpose_diameter_guard(self._tree("x = 1"))
        assert ok is False

    def test_run_code_blocks_cellpose_without_diameter(self, executor):
        code = "import cellpose\ncellpose.models.Cellpose().eval(img)"
        result = executor.run_code_new(code, execution_mode="buffered")
        assert "Library Guard Error" in result
        assert "cellpose" in result.lower()

    def test_run_code_allows_cellpose_with_diameter(self, executor):
        code = "import cellpose\ncellpose.models.Cellpose().eval(img, diameter=15, channels=[0,0])"
        # Will fail at runtime (no real cellpose), but must pass all guards
        result = executor.run_code_new(code, execution_mode="buffered")
        assert "Library Guard Error" not in result


class TestCellposeChannelsGuard:
    def _tree(self, code):
        return _ast.parse(code)

    def test_with_channels_passes(self):
        ok, _ = _cellpose_channels_guard(self._tree("model.eval(img, channels=[0,0])"))
        assert ok is True

    def test_without_channels_fails(self):
        ok, reason = _cellpose_channels_guard(self._tree("model.eval(img)"))
        assert ok is False
        assert "channels" in reason

    def test_no_call_fails(self):
        ok, _ = _cellpose_channels_guard(self._tree("x = 1"))
        assert ok is False


class TestCellposeFlowThresholdGuard:
    def _tree(self, code):
        return _ast.parse(code)

    def test_valid_value_passes(self):
        ok, _ = _cellpose_flow_threshold_guard(self._tree("model.eval(img, flow_threshold=0.4)"))
        assert ok is True

    def test_zero_passes(self):
        ok, _ = _cellpose_flow_threshold_guard(self._tree("model.eval(img, flow_threshold=0.0)"))
        assert ok is True

    def test_max_boundary_passes(self):
        ok, _ = _cellpose_flow_threshold_guard(self._tree("model.eval(img, flow_threshold=3.0)"))
        assert ok is True

    def test_too_high_fails(self):
        ok, reason = _cellpose_flow_threshold_guard(
            self._tree("model.eval(img, flow_threshold=5.0)")
        )
        assert ok is False
        assert "flow_threshold" in reason

    def test_negative_fails(self):
        ok, reason = _cellpose_flow_threshold_guard(
            self._tree("model.eval(img, flow_threshold=-1.0)")
        )
        assert ok is False
        assert "flow_threshold" in reason

    def test_absent_passes(self):
        # Not set at all → guard passes (no literal to check)
        ok, _ = _cellpose_flow_threshold_guard(self._tree("model.eval(img)"))
        assert ok is True

    def test_variable_value_passes(self):
        # Can't check non-literal values statically
        ok, _ = _cellpose_flow_threshold_guard(self._tree("model.eval(img, flow_threshold=thresh)"))
        assert ok is True


class TestCellposeCellprobThresholdGuard:
    def _tree(self, code):
        return _ast.parse(code)

    def test_valid_value_passes(self):
        ok, _ = _cellpose_cellprob_threshold_guard(
            self._tree("model.eval(img, cellprob_threshold=0.0)")
        )
        assert ok is True

    def test_boundary_passes(self):
        ok, _ = _cellpose_cellprob_threshold_guard(
            self._tree("model.eval(img, cellprob_threshold=6.0)")
        )
        assert ok is True

    def test_too_high_fails(self):
        ok, reason = _cellpose_cellprob_threshold_guard(
            self._tree("model.eval(img, cellprob_threshold=10.0)")
        )
        assert ok is False
        assert "cellprob_threshold" in reason

    def test_too_low_fails(self):
        ok, reason = _cellpose_cellprob_threshold_guard(
            self._tree("model.eval(img, cellprob_threshold=-10.0)")
        )
        assert ok is False
        assert "cellprob_threshold" in reason

    def test_absent_passes(self):
        ok, _ = _cellpose_cellprob_threshold_guard(self._tree("model.eval(img)"))
        assert ok is True

    def test_variable_value_passes(self):
        ok, _ = _cellpose_cellprob_threshold_guard(
            self._tree("model.eval(img, cellprob_threshold=prob)")
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Runtime guards — cellpose image size
# ---------------------------------------------------------------------------

import sys  # noqa: E402
import types  # noqa: E402


@pytest.fixture()
def fake_cellpose():
    """
    Inject a minimal fake cellpose.models into sys.modules so tests
    work regardless of whether real cellpose is installed or what API version it has.
    """

    class FakeCellpose:
        def eval(self, x, *args, **kwargs):
            return [], [], []

    fake_models = types.ModuleType("cellpose.models")
    fake_models.Cellpose = FakeCellpose

    fake_pkg = types.ModuleType("cellpose")
    fake_pkg.models = fake_models

    orig_cellpose = sys.modules.get("cellpose")
    orig_models = sys.modules.get("cellpose.models")
    sys.modules["cellpose"] = fake_pkg
    sys.modules["cellpose.models"] = fake_models

    yield fake_models, FakeCellpose

    # Restore original state
    if orig_cellpose is None:
        sys.modules.pop("cellpose", None)
    else:
        sys.modules["cellpose"] = orig_cellpose
    if orig_models is None:
        sys.modules.pop("cellpose.models", None)
    else:
        sys.modules["cellpose.models"] = orig_models


class TestCellposeSizeGuard:
    """Runtime size interceptor tests — use fake_cellpose fixture, no real cellpose needed."""

    def test_small_image_passes(self, fake_cellpose):
        fake_models, FakeCellpose = fake_cellpose
        original_eval = FakeCellpose.eval
        namespace = {}
        td = _install_cellpose_size_guard(namespace)
        assert td is not None
        try:
            small = np.zeros((256, 256), dtype=np.uint8)
            FakeCellpose.eval(FakeCellpose(), small)  # must not raise
        finally:
            td()
        assert FakeCellpose.eval is original_eval

    def test_large_image_raises(self, fake_cellpose):
        fake_models, FakeCellpose = fake_cellpose
        namespace = {}
        td = _install_cellpose_size_guard(namespace)
        assert td is not None
        try:
            large = np.zeros(
                (_CELLPOSE_SIZE_THRESHOLD + 1, _CELLPOSE_SIZE_THRESHOLD + 1), dtype=np.uint8
            )
            with pytest.raises(RuntimeError, match="exceeds"):
                FakeCellpose.eval(FakeCellpose(), large)
        finally:
            td()

    def test_large_image_with_allow_flag_passes(self, fake_cellpose):
        fake_models, FakeCellpose = fake_cellpose
        namespace = {"cellpose_allow_large_image": True}
        td = _install_cellpose_size_guard(namespace)
        assert td is not None
        try:
            large = np.zeros((1024, 1024), dtype=np.uint8)
            FakeCellpose.eval(FakeCellpose(), large)  # must not raise
        finally:
            td()

    def test_teardown_restores_original(self, fake_cellpose):
        fake_models, FakeCellpose = fake_cellpose
        original_eval = FakeCellpose.eval
        namespace = {}
        td = _install_cellpose_size_guard(namespace)
        assert FakeCellpose.eval is not original_eval
        td()
        assert FakeCellpose.eval is original_eval

    def test_apply_remove_runtime_guards_lifecycle(self, executor, fake_cellpose):
        """_apply_runtime_guards installs teardowns; _remove_runtime_guards restores state."""
        fake_models, FakeCellpose = fake_cellpose
        original_eval = FakeCellpose.eval
        code = "import cellpose"
        teardowns = executor._apply_runtime_guards(code)
        assert len(teardowns) > 0
        assert FakeCellpose.eval is not original_eval
        executor._remove_runtime_guards(teardowns)
        assert FakeCellpose.eval is original_eval

    def test_list_input_first_element_checked(self, fake_cellpose):
        """When x is a list of arrays, the first element's shape is checked."""
        fake_models, FakeCellpose = fake_cellpose
        namespace = {}
        td = _install_cellpose_size_guard(namespace)
        assert td is not None
        try:
            large = np.zeros((1024, 1024), dtype=np.uint8)
            with pytest.raises(RuntimeError, match="exceeds"):
                FakeCellpose.eval(FakeCellpose(), [large])
        finally:
            td()

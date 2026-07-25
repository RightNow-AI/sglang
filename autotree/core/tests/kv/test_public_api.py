import autotree_core
import autotree_core.kv


def test_package_root_reexports_tree_kv_public_api() -> None:
    assert set(autotree_core.__all__) == {
        *autotree_core.kv.__all__,
        "__version__",
    }
    for name in autotree_core.kv.__all__:
        assert getattr(autotree_core, name) is getattr(autotree_core.kv, name)


def test_package_version_is_public() -> None:
    assert autotree_core.__version__ == "0.1.0"

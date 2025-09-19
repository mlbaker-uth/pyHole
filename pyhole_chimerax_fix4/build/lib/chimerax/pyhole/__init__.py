# src/chimerax/pyhole/__init__.py
from chimerax.core.toolshed import BundleAPI

class _API(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        from .tool import PyHoleTool
        # IMPORTANT: must RETURN a ToolInstance
        return PyHoleTool(session, ti.name)

    # Optional; helps session restore
    @staticmethod
    def get_class(class_name):
        if class_name == "PyHoleTool":
            from .tool import PyHoleTool
            return PyHoleTool
        raise ValueError(class_name)

# IMPORTANT: ChimeraX looks for this name
bundle_api = _API()

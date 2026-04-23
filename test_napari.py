

if __name__ == "__main__":
    import napari
    from pymmcore_plus.experimental.unicore import UniMMCore

    core = UniMMCore()
    core.loadSystemConfiguration()

    viewer = napari.Viewer()

    viewer.window.add_plugin_dock_widget(plugin_name="napari-micromanager")

    napari.run()
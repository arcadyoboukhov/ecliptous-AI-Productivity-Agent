from PySide6.QtWidgets import QApplication
import main_window
from pathlib import Path

print('MATPLOTLIB_AVAILABLE=', getattr(main_window, 'MATPLOTLIB_AVAILABLE', None))
app = QApplication.instance() or QApplication([])
try:
    w = main_window.BasicWindow()
    try:
        w._ensure_graph_canvases()
    except Exception as e:
        print('ensure_graph_canvases error:', e)
    gc = getattr(w, 'graph_canvases', None)
    if not gc:
        print('graph_canvases is None')
    else:
        print('graph_canvases keys=', list(gc.keys()))
        for k, v in gc.items():
            try:
                axes = getattr(v, 'figure', None) and getattr(v.figure, 'axes', None)
                count = len(axes) if axes is not None else 'no-axes'
                print(k, 'canvas_type=', type(v), 'fig_axes_count=', count)
            except Exception as e:
                print(k, 'inspect error', e)
except Exception as e:
    print('BasicWindow instantiation error:', e)
print('done')

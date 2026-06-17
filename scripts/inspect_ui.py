from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
# ensure project root is on sys.path so imports of main_window work when run from scripts/
proj_root = str(Path(__file__).parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import main_window
from datetime import datetime

print('MATPLOTLIB_AVAILABLE=', getattr(main_window, 'MATPLOTLIB_AVAILABLE', None))
app = QApplication.instance() or QApplication([])
w = main_window.BasicWindow()
# ensure canvases
w._ensure_graph_canvases()
print('Graph canvases keys:', list(getattr(w, 'graph_canvases', {}).keys()))
overview = w.graph_canvases.get('Overview') if getattr(w, 'graph_canvases', None) else None
print('Overview exists:', overview is not None)
if overview is not None:
    try:
        print('Overview widget visible:', overview.isVisible())
        print('Overview size:', overview.size())
        print('Overview minimumSize:', overview.minimumSize())
        print('Overview sizeHint:', overview.sizeHint())
        try:
            axes = overview.figure.axes
            print('Overview figure axes count:', len(axes))
        except Exception as e:
            print('Error reading axes:', e)
    except Exception as e:
        print('Error inspecting overview:', e)
else:
    print('No overview canvas')

# Print layout children for Home area
from PySide6.QtCore import Qt

try:
    central = w.centralWidget()
    print('central widget children count:', central.layout().count())
    for i in range(central.layout().count()):
        item = central.layout().itemAt(i)
        widget = item.widget()
        print('child', i, type(widget), getattr(widget, 'objectName', None))
except Exception as e:
    print('Error listing central children:', e)

print('done at', datetime.now())

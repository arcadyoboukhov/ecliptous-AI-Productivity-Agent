import json
import sys
from pathlib import Path

proj_root = str(Path(__file__).parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from agent.analytics.graph import render_analytics_graph

model = render_analytics_graph(
    graph_type='time_series',
    metric='focus',
    time_window=30,
    comparison_mode=None,
    options={'moving_average': 7},
)
print(json.dumps(model, indent=2))

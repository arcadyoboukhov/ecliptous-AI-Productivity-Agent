import sys
import threading
import math
import traceback
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSystemTrayIcon,
    QMenu,
    QTabBar,
    QHBoxLayout,
    QFrame,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QHeaderView,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QAction, QCloseEvent
import io
from PySide6.QtGui import QPixmap

# Matplotlib embedding (optional dependency)
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def _create_test_pixmap(name: str, width: int = 380, height: int = 220):
    """Create a simple deterministic test plot as a QPixmap for fallback rendering."""
    try:
        from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QFont
        pix = QPixmap(width, height)
        pix.fill(QColor('#ffffff'))
        p = QPainter(pix)
        try:
            pen = QPen(QColor('#dddddd'))
            p.setPen(pen)
            # border/axes
            p.drawRect(8, 20, width - 16, height - 36)
            # title
            font = QFont('Sans', 9)
            p.setFont(font)
            p.setPen(QPen(QColor('#333333')))
            p.drawText(12, 14, name)
            # draw a simple waveform as demo data
            import math
            pen = QPen(QColor('#2a9df4'))
            pen.setWidth(2)
            p.setPen(pen)
            pts = []
            for i in range(0, width - 24):
                x = 12 + i
                y = int((height - 36) / 2 + math.sin(i / 10.0) * ((height - 36) / 3)) + 22
                pts.append((x, y))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        finally:
            p.end()
        return pix
    except Exception:
        return None


def _render_model_to_pixmap(model: dict, width: int = 760, height: int = 380) -> Optional[QPixmap]:
    """Render a simple time_series `model` to a QPixmap using Matplotlib Agg backend.

    Returns None on failure so callers can fallback to deterministic pixmap.
    """
    try:
        # Use a local import so we don't require Matplotlib at module import time
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as AggCanvas
        fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, tight_layout=True)
        canvas = AggCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#fafafa')
        gtype = model.get('graph_type')
        if gtype == 'time_series':
            import matplotlib.dates as mdates
            from datetime import datetime, timezone
            for series in model.get('series', []):
                pts = series.get('points', []) or []
                dates = []
                vals = []
                for p in pts:
                    d = p.get('date')
                    v = p.get('value')
                    try:
                        if isinstance(d, str):
                            dt = datetime.fromisoformat(d)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = d
                        dates.append(dt)
                        vals.append(v if v is not None else float('nan'))
                    except Exception:
                        continue
                if dates:
                    ax.plot(dates, vals, label=series.get('label'), color='#2a9df4')
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
        elif gtype == 'comparison':
            series = model.get('series', [])
            labels = [s.get('label') for s in series]
            vals = [s.get('value') or 0 for s in series]
            ax.bar(range(len(vals)), vals, color=['#888', '#2a9df4'][: len(vals)])
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels)
        else:
            ax.text(0.5, 0.5, 'Unsupported graph type', ha='center', va='center')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        data = buf.read()
        pix = QPixmap()
        if pix.loadFromData(data):
            return pix
        return None
    except Exception:
        return None


class BasicWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ecliptous AI")
        self.setGeometry(100, 100, 800, 520)

        # Agent thread placeholders
        self.agent_thread = None
        self.agent_stop_flag = threading.Event()

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        # Tabs
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Home")
        self.tab_bar.addTab("Tasks")
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tab_bar)

        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        layout.addWidget(self.content_widget)

        # Tray
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exit_application)
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

        # Build initial Home
        try:
            self.build_home_view()
        except Exception:
            placeholder = QLabel("Home view unavailable")
            placeholder.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(placeholder)

        # Start agent thread
        self.start_agent_thread()
        # Run an initial re-prediction using all available data
        try:
            from main_window import run_reprediction as _run_reprediction
            self._latest_prediction = _run_reprediction()
        except Exception:
            self._latest_prediction = None

        # Home refresh disabled: prevents rebuilding Home view and re-rendering graphs
        # To re-enable periodic refresh, uncomment below.
        # try:
        #     self.home_refresh_timer = QTimer(self)
        #     self.home_refresh_timer.timeout.connect(self._maybe_refresh_home)
        #     self.home_refresh_timer.start(3000)
        # except Exception:
        #     pass

        # Update live status every second
        try:
            self.status_update_timer = QTimer(self)
            self.status_update_timer.timeout.connect(self.update_live_status)
            self.status_update_timer.start(1000)
        except Exception:
            pass

        # Graph update disabled - graphs render once on Home tab build
        # To update graphs manually, call self.update_graphs()
        # Uncomment below to re-enable periodic updates every 3s:
        # try:
        #     self.graphs_update_timer = QTimer(self)
        #     self.graphs_update_timer.timeout.connect(self.update_graphs)
        #     self.graphs_update_timer.start(3000)
        # except Exception:
        #     pass
        
        # Update live task prediction periodically (check every 2s, predictions calculated every 60s)
        try:
            self.live_prediction_timer = QTimer(self)
            self.live_prediction_timer.timeout.connect(self.update_live_task_prediction)
            self.live_prediction_timer.start(2000)
        except Exception:
            pass

    def start_agent_thread(self):
        try:
            from main import main as agent_main

            self.agent_stop_flag = threading.Event()

            def run_agent():
                try:
                    agent_main(stop_flag=self.agent_stop_flag)
                except Exception:
                    pass

            self.agent_thread = threading.Thread(target=run_agent, daemon=True)
            self.agent_thread.start()
        except Exception:
            pass

    def repredict_on_setup(self):
        """Build behavioral model from DB sessions and run predictions for UI setup."""
        from agent.analytics.behavioral_model import BehavioralModel
        from agent.analytics.prediction_hooks import PredictionHooks
        from agent.analytics.persistence import load_sessions
        from datetime import datetime, timezone

        self._latest_prediction = None
        try:
            sessions = load_sessions()
        except Exception:
            sessions = []

        model = BehavioralModel()

        # Build model from all sessions
        for s in sessions:
            try:
                # Prefer assigned/inferred task id when updating per-task baselines
                task_id = getattr(s, 'inferred_task_id', None)
                if not task_id:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        task_id = assign.get('task_id')
                model.update_from_session(s, task_id)
            except Exception:
                try:
                    model.update_from_session(s, None)
                except Exception:
                    pass

        # Create prediction hooks
        self._prediction_hooks = PredictionHooks(behavioral_model=model)

        # Find active session/task to predict for
        active_task = None
        active_duration = 0.0
        for s in sessions:
            try:
                if getattr(s, 'in_progress', False) or getattr(s, 'end', None) is None:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        active_task = assign.get('task_id')
                    elif getattr(s, 'inferred_task_id', None):
                        active_task = getattr(s, 'inferred_task_id')
                    if getattr(s, 'start', None):
                        active_duration = (datetime.now(timezone.utc) - s.start).total_seconds() / 60.0
                    break
            except Exception:
                continue

        # Run a periodic update prediction for the active task (if any)
        if active_task:
            try:
                res = self._prediction_hooks.on_periodic_update(active_task, active_duration)
                self._latest_prediction = res.get('completion_estimate')
            except Exception:
                self._latest_prediction = None
        else:
            self._latest_prediction = None

    def exit_application(self):
        try:
            if hasattr(self, 'agent_stop_flag'):
                self.agent_stop_flag.set()
        finally:
            QApplication.quit()

    def closeEvent(self, event: QCloseEvent):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage("Ecliptous AI", "Application minimized to tray", QSystemTrayIcon.MessageIcon.Information, 2000)

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()

    def on_tab_changed(self, index: int):
        name = self.tab_bar.tabText(index) if index >= 0 else ""
        try:
            if name == "Home":
                self.build_home_view()
            elif name == "Tasks":
                self.build_tasks_view()
            else:
                self.build_placeholder_view(name)
        except Exception as e:
            self.clear_content()
            err = QLabel(f"Error building view: {e}")
            err.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(err)

    def clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    def build_home_view(self):
        try:
            print('[UI-GRAPH] build_home_view (primary) invoked', flush=True)
        except Exception:
            pass
        self.clear_content()

        # Top area: state + key info
        top = QWidget()
        top_l = QHBoxLayout()
        top.setLayout(top_l)
        try:
            top.setMaximumHeight(160)
        except Exception:
            pass

        state_frame = QFrame()
        state_frame.setFrameShape(QFrame.StyledPanel)
        state_layout = QVBoxLayout()
        state_frame.setLayout(state_layout)
        self.current_state_label = QLabel("Current State: Unknown")
        self.current_state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        state_layout.addWidget(self.current_state_label)
        self.current_task_label = QLabel("Current Task: Unknown")
        state_layout.addWidget(self.current_task_label)
        
        # Live intensity score under the task name
        self.current_intensity_label = QLabel("Intensity: N/A")
        self.current_intensity_label.setStyleSheet("color: #555;")
        state_layout.addWidget(self.current_intensity_label)
        # One-line status (intensity | state | app | task)
        self.live_status_label = QLabel("")
        self.live_status_label.setStyleSheet("font-family: monospace; color: #333; padding-top:6px;")
        state_layout.addWidget(self.live_status_label)
        state_layout.addStretch()

        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout()
        info_frame.setLayout(info_layout)
        info_title = QLabel("Key Info")
        info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        info_layout.addWidget(info_title)
        info_layout.addWidget(QLabel(f"Peak Focus Hours: {self.compute_peak_focus_hours()}"))
        info_layout.addWidget(QLabel(f"Most Productive Session Durations: {self.compute_most_productive_session_durations()}"))
        info_layout.addWidget(QLabel(f"Predicted Current Task Completion Time: {self.compute_predicted_current_task_completion()}"))
        info_layout.addStretch()

        top_l.addWidget(state_frame, 1)
        top_l.addWidget(info_frame, 1)
        self.content_layout.addWidget(top, 1)

        # Hard placeholder box that must be visible below the two boxes
        force_box = QFrame()
        force_box.setFrameShape(QFrame.StyledPanel)
        try:
            force_box.setStyleSheet("background:#111; border: 3px solid #ffeb3b;")
        except Exception:
            pass
        force_layout = QVBoxLayout()
        force_box.setLayout(force_layout)
        # Render PNG and place it on top of the placeholder
        # Render graphs once when building Home view (no periodic updates)
        if True:  # Always render once on Home tab build
            try:
                from agent.analytics.graph import render_analytics_graph
                model = render_analytics_graph(
                    graph_type="time_series",
                    metric="focus",
                    time_window=30,
                    comparison_mode=None,
                    options={"moving_average": 7},
                )
                self._home_graph_pixmap = _render_model_to_pixmap(model, width=760, height=380)
            except Exception:
                self._home_graph_pixmap = None

        rp = self._home_graph_pixmap

        if rp:
            img = QLabel()
            img.setPixmap(rp)
            img.setScaledContents(True)
            try:
                img.setMinimumSize(760, 380)
            except Exception:
                pass
            force_layout.addWidget(img)
        # No text label; image only
        try:
            force_box.setMinimumHeight(420)
            force_box.setMaximumHeight(1200)
        except Exception:
            pass
        self.content_layout.addWidget(force_box, 5)
        try:
            self.content_layout.setStretch(0, 0)
            self.content_layout.setStretch(1, 1)
        except Exception:
            pass

        # Graph area (three embedded plots). Use Matplotlib if available,
        # otherwise fallback to simple labeled frames.
        graphs = QWidget()
        graphs_l = QHBoxLayout()
        graphs.setLayout(graphs_l)
        try:
            graphs.setMinimumHeight(260)
        except Exception:
            pass

        # Ensure canvases exist
        self._ensure_graph_canvases()
        try:
            graphs.setFixedHeight(420)
        except Exception:
            pass

        # Replace the three small graph boxes with a single large overview box
        # Create a single large graphs container
        graphs_l = QVBoxLayout()
        graphs.setLayout(graphs_l)
        try:
            graphs.setMinimumHeight(420)
        except Exception:
            pass

        # Ensure canvases exist
        self._ensure_graph_canvases()

        # Prefer the large Overview canvas if available, otherwise pick any existing canvas
        chosen_canvas = None
        if MATPLOTLIB_AVAILABLE and getattr(self, 'graph_canvases', None):
            chosen_canvas = self.graph_canvases.get('Overview') or None
            if not chosen_canvas:
                for key in ("Current vs Baseline", "Current vs Long-Term", "Session Trend"):
                    c = self.graph_canvases.get(key)
                    if c:
                        chosen_canvas = c
                        break

        if chosen_canvas:
            # Prefer rendering to a static pixmap for reliable display.
            pixmap_widget = None
            try:
                from agent.analytics.graph import render_analytics_graph
                model = render_analytics_graph(
                    graph_type="time_series",
                    metric="focus",
                    time_window=30,
                    comparison_mode=None,
                    options={"moving_average": 7},
                )
                rp = _render_model_to_pixmap(model, width=760, height=380)
                if rp:
                    lbl = QLabel()
                    lbl.setPixmap(rp)
                    lbl.setScaledContents(True)
                    lbl.setStyleSheet("border: 2px solid #2a9df4; background: #f0f6ff;")
                    try:
                        lbl.setMinimumSize(760, 380)
                    except Exception:
                        pass
                    pixmap_widget = lbl
                    try:
                        print('[UI-GRAPH] primary pixmap rendered size=', rp.size(), flush=True)
                    except Exception:
                        pass
            except Exception as e:
                self._log_ui_error(e, 'render_overview_pixmap')

            if pixmap_widget:
                # Stop here so only the two top boxes + placeholder are shown
                return

    def _maybe_refresh_home(self):
        try:
            if getattr(self, 'tab_bar', None) and self.tab_bar.currentIndex() == 0:
                self.build_home_view()
        except Exception:
            pass

    def update_live_status(self):
        """Read latest STATE_CHANGE event and update the one-line status label."""
        try:
            from agent.storage import db
            events = db.get_events(event_type='STATE_CHANGE', limit=1)
            if not events:
                self.live_status_label.setText("")
                return

            ts_str, evt_type, payload = events[0]
            # Format timestamp
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(ts_str)
                timestr = ts.strftime('%H:%M:%S')
            except Exception:
                timestr = ts_str

            intensity = None
            state = None
            app = None
            task = None

            if isinstance(payload, dict):
                intensity = payload.get('intensity') or payload.get('input_activity_score')
                state = payload.get('to') or payload.get('state')
                app = payload.get('active_app') or payload.get('app')
                task = payload.get('active_task') or payload.get('task')

            intensity_str = f"{float(intensity):.1f}" if intensity is not None else "N/A"
            state_str = str(state).upper() if state else "UNKNOWN"
            app_str = str(app) if app else "N/A"
            task_str = str(task) if task else "N/A"

            line = f"[INTENSITY] {timestr} - Intensity: {intensity_str} | State: {state_str} | App: {app_str} | Task: {task_str}"
            try:
                self.live_status_label.setText(line)
            except Exception:
                pass

            # Also update the separate intensity/state/task labels for compatibility
            try:
                self.current_intensity_label.setText(f"Intensity: {intensity_str}")
            except Exception:
                pass
            try:
                # Update state text and color to reflect live state
                self.current_state_label.setText(f"Current State: {state_str}")
                # choose color mapping similar to build_home_view
                color = None
                try:
                    s_low = (state_str or "").lower()
                    if 'aligned' in s_low or 'active_aligned' in s_low:
                        color = 'green'
                    elif 'unalign' in s_low or 'drift' in s_low or 'active_unaligned' in s_low:
                        color = 'orange'
                    elif 'active' in s_low:
                        color = 'green'
                    elif 'paused' in s_low or 'pause' in s_low:
                        color = '#666'
                    elif 'idle' in s_low:
                        color = '#888'
                    else:
                        color = '#333'
                except Exception:
                    color = '#333'

                if color:
                    self.current_state_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
            except Exception:
                pass
            try:
                self.current_task_label.setText(f"Current Task: {task_str}")
            except Exception:
                pass

        except Exception:
            pass

    # ------------------ Graphing helpers ------------------
    def _ensure_graph_canvases(self):
        """Create Matplotlib FigureCanvas objects cached on the window."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            print('[UI-GRAPH] _ensure_graph_canvases called, MATPLOTLIB_AVAILABLE=', MATPLOTLIB_AVAILABLE, flush=True)
        except Exception:
            pass
        if getattr(self, 'graph_canvases', None):
            return
        try:
            self.graph_canvases = {}
            for name in ("Current vs Baseline", "Current vs Long-Term", "Session Trend"):
                fig = Figure(figsize=(4, 3), tight_layout=True)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.set_title(name)
                ax.set_facecolor('#fafafa')
                self.graph_canvases[name] = canvas
            # Add a larger Overview canvas for the Home free space
            try:
                fig = Figure(figsize=(8, 4), tight_layout=True)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.set_title('Overview')
                ax.set_facecolor('#fafafa')
                self.graph_canvases['Overview'] = canvas
            except Exception:
                pass
            try:
                print('[UI-GRAPH] created canvases:', list(self.graph_canvases.keys()), flush=True)
            except Exception:
                pass
        except Exception:
            self.graph_canvases = None

    def update_graphs(self):
        """Populate graphs with recent data. Resilient if Matplotlib missing."""
        if not MATPLOTLIB_AVAILABLE or not getattr(self, 'graph_canvases', None):
            return
        try:
            # Fetch recent STATE_CHANGE events with intensity
            from agent.storage import db
            from datetime import datetime, timezone
            events = []
            try:
                events = db.get_events(event_type='STATE_CHANGE', limit=240) or []
            except Exception:
                events = []

            times = []
            intensities = []
            for item in reversed(events):
                try:
                    ts_str = item[0]
                    payload = item[2]
                    if isinstance(ts_str, datetime):
                        ts = ts_str
                    else:
                        ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                val = None
                if isinstance(payload, dict):
                    val = payload.get('intensity') or payload.get('input_activity_score')
                elif isinstance(payload, (int, float)):
                    val = payload
                if val is None:
                    continue
                try:
                    times.append(ts)
                    intensities.append(float(val))
                except Exception:
                    continue

            # If no state-change intensity events, try extracting timeline intensity from sessions
            if not intensities:
                try:
                    sessions = db.load_sessions_from_db(days_back=30) or []
                    timeline_points = []
                    for s in sessions:
                        # First, try per-session intensity_scores table
                        sid = getattr(s, 'session_id', None) or getattr(s, 'id', None)
                        if sid:
                            try:
                                scores = db.get_intensity_scores(sid, limit=500)
                                for ts, score in reversed(scores):
                                    timeline_points.append((ts, float(score)))
                            except Exception:
                                pass

                        # Next, try session.timeline dict if present
                        tl = getattr(s, 'timeline', None) or {}
                        if isinstance(tl, dict):
                            for k, v in tl.items():
                                try:
                                    t = k if isinstance(k, datetime) else datetime.fromisoformat(k)
                                    if getattr(v, 'get', None):
                                        iv = v.get('intensity') or v.get('input_activity_score')
                                    elif isinstance(v, (int, float)):
                                        iv = v
                                    else:
                                        iv = None
                                    if iv is not None:
                                        timeline_points.append((t, float(iv)))
                                except Exception:
                                    continue
                    if timeline_points:
                        timeline_points.sort(key=lambda x: x[0])
                        times = [p[0] for p in timeline_points]
                        intensities = [p[1] for p in timeline_points]
                except Exception:
                    pass

            # Also render the analytics model into the large Overview canvas (if present)
            try:
                overview = self.graph_canvases.get('Overview')
                if overview:
                    try:
                        from agent.analytics.graph import render_analytics_graph
                        model = render_analytics_graph(
                            graph_type='time_series',
                            metric='focus',
                            time_window=30,
                            comparison_mode=None,
                            options={'moving_average': 7},
                        )
                        self._render_graph_model(overview, model)
                    except Exception as e:
                        self._log_ui_error(e, 'render_overview')
            except Exception:
                pass

            # Current vs Baseline: plot intensity and mean baseline
            cvs = self.graph_canvases.get('Current vs Baseline')
            if cvs:
                ax = cvs.figure.axes[0]
                ax.clear()
                if times and intensities:
                    try:
                        import matplotlib.dates as mdates
                        ax.plot(times, intensities, label='Intensity', color='#2a9df4')
                        mean = sum(intensities) / len(intensities)
                        ax.hlines(mean, times[0], times[-1], colors='#888', linestyles='--', label=f'Baseline {mean:.2f}')
                        ax.legend()
                        ax.set_ylabel('Intensity')
                        ax.set_xlabel('Time')
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        for lbl in cvs.figure.axes[0].get_xticklabels():
                            lbl.set_rotation(30)
                    except Exception:
                        ax.plot(times, intensities, label='Intensity', color='#2a9df4')
                else:
                    ax.text(0.5, 0.5, 'No recent intensity data', ha='center', va='center')
                cvs.draw_idle()

            # Current vs Long-Term: short vs long moving averages
            cvs2 = self.graph_canvases.get('Current vs Long-Term')
            if cvs2:
                ax = cvs2.figure.axes[0]
                ax.clear()
                if intensities and times:
                    try:
                        import numpy as _np
                        arr = _np.array(intensities)
                        if arr.size >= 3:
                            short = _np.convolve(arr, _np.ones(5)/5, mode='same')
                            long = _np.convolve(arr, _np.ones(min(30, arr.size))/min(30, arr.size), mode='same')
                        else:
                            short = long = arr
                        ax.plot(times, short, label='Short MA (5)', color='#2ca02c')
                        ax.plot(times, long, label='Long MA (30)', color='#d62728')
                        ax.legend()
                        ax.set_ylabel('Intensity MA')
                    except Exception:
                        ax.plot(times, intensities, label='Intensity', color='#2a9df4')
                else:
                    ax.text(0.5, 0.5, 'No recent intensity data', ha='center', va='center')
                cvs2.draw_idle()

            # Session Trend: bar chart of last 14 sessions durations
            cvs3 = self.graph_canvases.get('Session Trend')
            if cvs3:
                ax = cvs3.figure.axes[0]
                ax.clear()
                try:
                    sessions = db.load_sessions_from_db(days_back=60) or []
                    durations = []
                    labels = []
                    now = datetime.now(timezone.utc)
                    for s in sessions[-14:]:
                        start = getattr(s, 'start', None)
                        end = getattr(s, 'end', None)
                        if start:
                            try:
                                if end is None:
                                    end = now
                                dur = (end - start).total_seconds() / 60.0
                                durations.append(dur)
                                labels.append(start.strftime('%m-%d'))
                            except Exception:
                                continue
                    if durations:
                        ax.bar(range(len(durations)), durations, color='#8c564b')
                        ax.set_xticks(range(len(durations)))
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.set_ylabel('Minutes')
                    else:
                        ax.text(0.5, 0.5, 'No sessions', ha='center', va='center')
                except Exception:
                    ax.text(0.5, 0.5, 'Session data unavailable', ha='center', va='center')
                cvs3.draw_idle()

        except Exception as e:
            try:
                print(f"[UI-GRAPH] update_graphs error: {e}", flush=True)
            except Exception:
                pass
            # Don't let graph errors crash UI
            self._log_ui_error(e, 'update_graphs')
            pass

    def _log_ui_error(self, exc: Exception, context: str = "ui"):
        """Log UI errors to stdout and append to `ui_err.txt` for diagnosis.

        Prints the traceback and appends it to `ui_err.txt` in the repository root.
        """
        try:
            print(f"[UI-ERROR] Context={context}: {exc}", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        try:
            p = Path(__file__).parent.parent / 'ui_err.txt'
            with open(p, 'a', encoding='utf-8') as f:
                from datetime import datetime
                f.write(f"[{datetime.now().isoformat()}] Context={context}: {repr(exc)}\n")
                try:
                    traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
                except Exception:
                    f.write('Traceback unavailable\n')
        except Exception:
            pass

    def _render_graph_model(self, canvas, model: dict):
        """Render a graph-agnostic model (from agent.analytics.graph.render_analytics_graph)
        into a Matplotlib canvas already created by `_ensure_graph_canvases`.
        This method is defensive and will not raise exceptions to the caller.
        """
        if not MATPLOTLIB_AVAILABLE or canvas is None or not model:
            return
        try:
            ax = canvas.figure.axes[0]
            ax.clear()
            gtype = model.get('graph_type')
            if gtype == 'time_series':
                import matplotlib.dates as mdates
                from datetime import datetime, timezone

                for series in model.get('series', []):
                    pts = series.get('points', []) or []
                    dates = []
                    vals = []
                    for p in pts:
                        d = p.get('date')
                        v = p.get('value')
                        try:
                            if isinstance(d, str):
                                dt = datetime.fromisoformat(d)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                            else:
                                dt = d
                            dates.append(dt)
                            vals.append(v if v is not None else float('nan'))
                        except Exception:
                            continue
                    if dates:
                        ax.plot(dates, vals, label=series.get('label'))
                ax.legend()
                ax.set_ylabel(model.get('metric', ''))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(30)

            elif gtype == 'comparison':
                # simple bar representation for baseline/current
                series = model.get('series', [])
                labels = [s.get('label') for s in series]
                vals = [s.get('value') or 0 for s in series]
                ax.bar(range(len(vals)), vals, color=['#888', '#2a9df4'][: len(vals)])
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(labels)
                ax.set_ylabel(model.get('metric', ''))
                for i, s in enumerate(series):
                    dp = s.get('delta_percent')
                    if dp is not None:
                        ax.text(i, vals[i], f"{dp:+.1f}%", ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'Unsupported graph type', ha='center', va='center')

            canvas.draw_idle()
        except Exception:
            try:
                ax = canvas.figure.axes[0]
                ax.clear()
                ax.text(0.5, 0.5, 'Graph render error', ha='center', va='center')
                canvas.draw_idle()
            except Exception:
                pass
    
    def update_live_task_prediction(self):
        """Update current task label with live task prediction.
        
        Task predictions are calculated every 60 seconds, starting 1 minute after
        a session is started. Shows countdown during initial collection period.
        """
        try:
            from agent.storage.db import get_latest_live_prediction
            from agent.session.gate import get_session_gate
            from datetime import datetime, timezone
            
            gate = get_session_gate()
            
            # Check if session is active
            if not gate.is_active():
                # No active session, reset to default
                if hasattr(self, 'current_task_label'):
                    self.current_task_label.setText("Current Task: Unknown")
                    self.current_task_label.setStyleSheet("")
                return
            
            # Get session ID and session object
            try:
                session_id = gate.get_active_session_id()
                # Import session manager to get session details
                from agent.session.manager_v2 import SessionManager
                session_mgr = SessionManager()
                session = session_mgr.get_session(session_id)
            except Exception:
                return
            
            if not session_id or not session:
                return
            
            # Check if we're in the initial 1-minute collection period
            if session.started_at:
                now = datetime.now(timezone.utc)
                time_since_start = (now - session.started_at).total_seconds()
                
                if time_since_start < 60:
                    # Still in initial collection period - show countdown
                    seconds_remaining = int(60 - time_since_start)
                    if hasattr(self, 'current_task_label'):
                        self.current_task_label.setText(
                            f"Current Task: Collecting data... ({seconds_remaining}s remaining)"
                        )
                        self.current_task_label.setStyleSheet("color: #0066cc;")
                    return
            
            # Past 1-minute mark - check for predictions
            prediction = get_latest_live_prediction(session_id)
            
            if not prediction or not hasattr(self, 'current_task_label'):
                # No prediction yet, but should be coming soon
                self.current_task_label.setText("Current Task: (collecting data...)")
                self.current_task_label.setStyleSheet("color: #0066cc;")
                return
            
            # Update label with live prediction as current task
            task_id = prediction.get('task_id')
            confidence = prediction.get('confidence')
            
            if task_id and confidence is not None:
                conf_pct = int(confidence * 100)
                # Display as: "Current Task: task_id (XX%)"
                self.current_task_label.setText(
                    f"Current Task: {task_id} ({conf_pct}%)"
                )
                
                # Color based on confidence: red (low) → yellow (medium) → green (high)
                if confidence >= 0.70:
                    color = "#00cc00"  # Green - high confidence
                    weight = "bold"
                elif confidence >= 0.50:
                    color = "#ffcc00"  # Yellow - medium confidence
                    weight = "600"
                else:
                    color = "#ff6600"  # Orange - low confidence
                    weight = "normal"
                
                self.current_task_label.setStyleSheet(f"color: {color}; font-weight: {weight};")
            else:
                # No valid prediction yet
                self.current_task_label.setText("Current Task: (collecting data...)")
                self.current_task_label.setStyleSheet("color: #0066cc;")
        
        except Exception as e:
            try:
                if hasattr(self, 'current_task_label'):
                    self.current_task_label.setText("Current Task: (error)")
            except Exception:
                pass

    def build_tasks_view(self):
        self.clear_content()
        tree = QTreeWidget()
        tree.setColumnCount(6)
        tree.setHeaderLabels(["Task ID", "Session ID", "Start", "End", "Duration (m)", "Confidence"])
        tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=30)
        except Exception:
            sessions = []

        for s in sessions:
            sess_id = getattr(s, 'session_id', None) or getattr(s, 'id', None)
            segments = getattr(s, 'intra_session_tasks', []) or []
            for seg in segments:
                start = getattr(seg, 'start_time', None)
                end = getattr(seg, 'end_time', None)
                dur = None
                try:
                    if start and end:
                        dur = int((end - start).total_seconds() / 60)
                except Exception:
                    dur = None
                conf = getattr(seg, 'confidence', None)
                top = QTreeWidgetItem([
                    str(getattr(seg, 'task_id', 'unknown')),
                    str(sess_id),
                    start.isoformat() if start else '',
                    end.isoformat() if end else '',
                    str(dur) if dur is not None else '',
                    f"{conf:.2f}" if isinstance(conf, float) else str(conf or ''),
                ])
                child = QTreeWidgetItem()
                top.addChild(child)
                tree.addTopLevelItem(top)
                widget = QTextEdit()
                widget.setReadOnly(True)
                try:
                    import json as _json
                    detail = { 'task_id': getattr(seg, 'task_id', None) }
                    widget.setText(_json.dumps(detail, indent=2, default=str))
                except Exception:
                    widget.setText(str(getattr(seg, 'task_id', None)))
                tree.setItemWidget(child, 0, widget)

        self.content_layout.addWidget(tree if tree.topLevelItemCount() else QLabel("No tasks found in the last 30 days."))

    def build_placeholder_view(self, name: str):
        # Placeholder view removed — leave content area empty.
        self.clear_content()
        return

    def compute_peak_focus_hours(self):
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=14)
            hours = {}
            for s in sessions:
                if hasattr(s, 'start') and s.start:
                    h = s.start.hour
                    dur = ((s.end - s.start).total_seconds()/60.0) if s.end and s.start else 0
                    hours[h] = hours.get(h, 0) + dur
            if not hours:
                return 'N/A'
            top = sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3]
            return ", ".join(f"{h}:00({int(m)}m)" for h, m in top)
        except Exception:
            return 'N/A'

    def compute_most_productive_session_durations(self):
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=30)
            durations = []
            for s in sessions:
                if hasattr(s, 'start') and hasattr(s, 'end') and s.start and s.end:
                    minutes = (s.end - s.start).total_seconds()/60.0
                    durations.append(minutes)
            if not durations:
                return 'N/A'
            top = sorted(durations, reverse=True)[:3]
            return ", ".join(f"{int(m)}m" for m in top)
        except Exception:
            return 'N/A'

    def compute_predicted_current_task_completion(self):
        # Use cached prediction from setup if available, otherwise attempt a quick re-predict
        est = getattr(self, '_latest_prediction', None)
        if est is None:
            try:
                from main_window import run_reprediction as _run_reprediction
                est = _run_reprediction()
            except Exception:
                est = None

        if not est:
            return 'N/A'

        try:
            mins = est.get('estimated_minutes_remaining')
            conf = est.get('confidence')
            if mins is None:
                return 'Completed'
            return f"{int(round(mins))}m remaining (conf {int(conf*100)}%)" if conf is not None else f"{int(round(mins))}m remaining"
        except Exception:
            return 'N/A'


def main():
    app = QApplication(sys.argv)
    window = BasicWindow()
    window.show()
    window.tray_icon.showMessage("Ecliptous AI", "Application started in system tray", QSystemTrayIcon.MessageIcon.Information, 2000)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# --- Compatibility: expose graph helpers to whichever BasicWindow class is bound at import time ---
def _module_ensure_graph_canvases(self):
    if not MATPLOTLIB_AVAILABLE:
        return
    if getattr(self, 'graph_canvases', None):
        return
    try:
        self.graph_canvases = {}
        for name in ("Current vs Baseline", "Current vs Long-Term", "Session Trend"):
            fig = Figure(figsize=(4, 3), tight_layout=True)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_title(name)
            ax.set_facecolor('#fafafa')
            self.graph_canvases[name] = canvas
    except Exception:
        self.graph_canvases = None


def _module_update_graphs(self):
    if not MATPLOTLIB_AVAILABLE or not getattr(self, 'graph_canvases', None):
        return
    try:
        from agent.storage import db
        from datetime import datetime, timezone
        events = []
        try:
            events = db.get_events(event_type='STATE_CHANGE', limit=240) or []
        except Exception:
            events = []
        try:
            print('[UI-GRAPH] _module_update_graphs: events_fetched=', len(events), flush=True)
        except Exception:
            pass

        times = []
        intensities = []
        for item in reversed(events):
            try:
                ts_str = item[0]
                payload = item[2]
                if isinstance(ts_str, datetime):
                    ts = ts_str
                else:
                    ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            val = None
            if isinstance(payload, dict):
                val = payload.get('intensity') or payload.get('input_activity_score')
            elif isinstance(payload, (int, float)):
                val = payload
            if val is None:
                continue
            try:
                times.append(ts)
                intensities.append(float(val))
            except Exception:
                continue

        if not intensities:
            try:
                sessions = db.load_sessions_from_db(days_back=30) or []
                timeline_points = []
                for s in sessions:
                    # First, try per-session intensity_scores table
                    sid = getattr(s, 'session_id', None) or getattr(s, 'id', None)
                    if sid:
                        try:
                            scores = db.get_intensity_scores(sid, limit=500)
                            for ts, score in reversed(scores):
                                timeline_points.append((ts, float(score)))
                        except Exception:
                            pass

                    # Next, try session.timeline dict if present
                    tl = getattr(s, 'timeline', None) or {}
                    if isinstance(tl, dict):
                        for k, v in tl.items():
                            try:
                                t = k if isinstance(k, datetime) else datetime.fromisoformat(k)
                                if getattr(v, 'get', None):
                                    iv = v.get('intensity') or v.get('input_activity_score')
                                elif isinstance(v, (int, float)):
                                    iv = v
                                else:
                                    iv = None
                                if iv is not None:
                                    timeline_points.append((t, float(iv)))
                            except Exception:
                                continue
                if timeline_points:
                    timeline_points.sort(key=lambda x: x[0])
                    times = [p[0] for p in timeline_points]
                    intensities = [p[1] for p in timeline_points]
            except Exception:
                pass

        cvs = self.graph_canvases.get('Current vs Baseline')
        if cvs:
            ax = cvs.figure.axes[0]
            ax.clear()
            if times and intensities:
                try:
                    import matplotlib.dates as mdates
                    ax.plot(times, intensities, label='Intensity', color='#2a9df4')
                    mean = sum(intensities) / len(intensities)
                    ax.hlines(mean, times[0], times[-1], colors='#888', linestyles='--', label=f'Baseline {mean:.2f}')
                    ax.legend()
                    ax.set_ylabel('Intensity')
                    ax.set_xlabel('Time')
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    for lbl in cvs.figure.axes[0].get_xticklabels():
                        lbl.set_rotation(30)
                except Exception:
                    ax.plot(times, intensities, label='Intensity', color='#2a9df4')
            else:
                ax.text(0.5, 0.5, 'No recent intensity data', ha='center', va='center')
            cvs.draw_idle()

        cvs2 = self.graph_canvases.get('Current vs Long-Term')
        if cvs2:
            ax = cvs2.figure.axes[0]
            ax.clear()
            if intensities and times:
                try:
                    import numpy as _np
                    arr = _np.array(intensities)
                    if arr.size >= 3:
                        short = _np.convolve(arr, _np.ones(5)/5, mode='same')
                        long = _np.convolve(arr, _np.ones(min(30, arr.size))/min(30, arr.size), mode='same')
                    else:
                        short = long = arr
                    ax.plot(times, short, label='Short MA (5)', color='#2ca02c')
                    ax.plot(times, long, label='Long MA (30)', color='#d62728')
                    ax.legend()
                except Exception:
                    ax.plot(times, intensities, label='Intensity', color='#2a9df4')
            else:
                ax.text(0.5, 0.5, 'No recent intensity data', ha='center', va='center')
            cvs2.draw_idle()

        cvs3 = self.graph_canvases.get('Session Trend')
        if cvs3:
            ax = cvs3.figure.axes[0]
            ax.clear()
            try:
                sessions = db.load_sessions_from_db(days_back=60) or []
                durations = []
                labels = []
                now = datetime.now(timezone.utc)
                for s in sessions[-14:]:
                    start = getattr(s, 'start', None)
                    end = getattr(s, 'end', None)
                    if start:
                        try:
                            if end is None:
                                end = now
                            dur = (end - start).total_seconds() / 60.0
                            durations.append(dur)
                            labels.append(start.strftime('%m-%d'))
                        except Exception:
                            continue
                if durations:
                    ax.bar(range(len(durations)), durations, color='#8c564b')
                    ax.set_xticks(range(len(durations)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                else:
                    ax.text(0.5, 0.5, 'No sessions', ha='center', va='center')
            except Exception:
                ax.text(0.5, 0.5, 'Session data unavailable', ha='center', va='center')
            cvs3.draw_idle()

    except Exception as e:
        try:
            print(f"[UI-GRAPH] _module_update_graphs error: {e}", flush=True)
        except Exception:
            pass
        pass


# Attach to BasicWindow if present
try:
    BasicWindow._ensure_graph_canvases = _module_ensure_graph_canvases
    BasicWindow.update_graphs = _module_update_graphs
    BasicWindow.build_home_view = _module_build_home_view
except Exception:
    pass


# Force a consistent Home view with a visible Overview graph (used when multiple
# BasicWindow definitions exist in this file).
def _module_build_home_view(self):
    try:
        print('[UI-GRAPH] _module_build_home_view invoked', flush=True)
    except Exception:
        pass
    try:
        if hasattr(self, 'clear_content'):
            self.clear_content()
    except Exception:
        pass

    # Top info area
    top = QWidget()
    top_l = QHBoxLayout()
    top.setLayout(top_l)
    try:
        top.setMaximumHeight(160)
    except Exception:
        pass

    state_frame = QFrame()
    state_frame.setFrameShape(QFrame.StyledPanel)
    state_layout = QVBoxLayout()
    state_frame.setLayout(state_layout)
    self.current_state_label = QLabel("Current State: Unknown")
    self.current_state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
    state_layout.addWidget(self.current_state_label)
    self.current_task_label = QLabel("Current Task: Unknown")
    state_layout.addWidget(self.current_task_label)
    self.current_intensity_label = QLabel("Intensity: N/A")
    self.current_intensity_label.setStyleSheet("color: #555;")
    state_layout.addWidget(self.current_intensity_label)
    state_layout.addStretch()

    info_frame = QFrame()
    info_frame.setFrameShape(QFrame.StyledPanel)
    info_layout = QVBoxLayout()
    info_frame.setLayout(info_layout)
    info_title = QLabel("Key Info")
    info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
    info_layout.addWidget(info_title)
    try:
        peak_focus = self.compute_peak_focus_hours() if hasattr(self, 'compute_peak_focus_hours') else 'N/A'
    except Exception:
        peak_focus = 'N/A'
    try:
        most_prod = self.compute_most_productive_session_durations() if hasattr(self, 'compute_most_productive_session_durations') else 'N/A'
    except Exception:
        most_prod = 'N/A'
    info_layout.addWidget(QLabel(f"Peak Focus Hours: {peak_focus}"))
    info_layout.addWidget(QLabel(f"Most Productive Session Durations: {most_prod}"))
    info_layout.addStretch()

    top_l.addWidget(state_frame, 1)
    top_l.addWidget(info_frame, 1)

    try:
        self.content_layout.addWidget(top)
    except Exception:
        return

    # Graph area (Overview)
    graphs = QWidget()
    graphs_l = QVBoxLayout()
    graphs.setLayout(graphs_l)
    try:
        graphs.setMinimumHeight(420)
    except Exception:
        pass

    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    fl = QVBoxLayout()
    frame.setLayout(fl)

    pixmap_widget = None
    try:
        from agent.analytics.graph import render_analytics_graph
        model = render_analytics_graph(
            graph_type="time_series",
            metric="focus",
            time_window=30,
            comparison_mode=None,
            options={"moving_average": 7},
        )
        rp = _render_model_to_pixmap(model, width=760, height=380)
        if rp:
            try:
                print('[UI-GRAPH] overview pixmap rendered', flush=True)
            except Exception:
                pass
            lbl = QLabel()
            lbl.setPixmap(rp)
            lbl.setScaledContents(True)
            try:
                lbl.setMinimumSize(760, 380)
            except Exception:
                pass
            pixmap_widget = lbl
    except Exception:
        try:
            print('[UI-GRAPH] overview pixmap render failed', flush=True)
        except Exception:
            pass
        pixmap_widget = None

    if not pixmap_widget:
        pix = _create_test_pixmap("Overview", width=760, height=380)
        if pix:
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setScaledContents(True)
        else:
            lbl = QLabel("Overview Graph Placeholder")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#f5f7ff; border: 1px solid #ddd; font-size:16px; padding:24px;")
        pixmap_widget = lbl

    fl.addWidget(pixmap_widget)
    graphs_l.addWidget(frame)
    self.content_layout.addWidget(graphs, 2)


try:
    BasicWindow.build_home_view = _module_build_home_view
except Exception:
    pass


# Ensure the active BasicWindow class has prediction helpers (covers duplicate/nested class definitions)
def _repredict_on_setup_impl(self):
    try:
        from agent.analytics.behavioral_model import BehavioralModel
        from agent.analytics.prediction_hooks import PredictionHooks
        from agent.analytics.persistence import load_sessions
        from datetime import datetime, timezone

        self._latest_prediction = None
        try:
            sessions = load_sessions()
        except Exception:
            sessions = []

        model = BehavioralModel()
        for s in sessions:
            try:
                task_id = getattr(s, 'inferred_task_id', None)
                if not task_id:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        task_id = assign.get('task_id')
                model.update_from_session(s, task_id)
            except Exception:
                try:
                    model.update_from_session(s, None)
                except Exception:
                    pass

        self._prediction_hooks = PredictionHooks(behavioral_model=model)

        active_task = None
        active_duration = 0.0
        for s in sessions:
            try:
                if getattr(s, 'in_progress', False) or getattr(s, 'end', None) is None:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        active_task = assign.get('task_id')
                    elif getattr(s, 'inferred_task_id', None):
                        active_task = getattr(s, 'inferred_task_id')
                    if getattr(s, 'start', None):
                        active_duration = (datetime.now(timezone.utc) - s.start).total_seconds() / 60.0
                    break
            except Exception:
                continue

        if active_task:
            try:
                res = self._prediction_hooks.on_periodic_update(active_task, active_duration)
                self._latest_prediction = res.get('completion_estimate')
            except Exception:
                self._latest_prediction = None
        else:
            self._latest_prediction = None
    except Exception:
        self._latest_prediction = None


def _compute_predicted_current_task_completion_impl(self):
    est = getattr(self, '_latest_prediction', None)
    if est is None:
        try:
            # try to build prediction on demand
            _repredict_on_setup_impl(self)
            est = getattr(self, '_latest_prediction', None)
        except Exception:
            est = None
    if not est:
        return 'N/A'
    try:
        mins = est.get('estimated_minutes_remaining')
        conf = est.get('confidence')
        if mins is None:
            return 'Completed'
        return f"{int(round(mins))}m remaining (conf {int(conf*100)}%)" if conf is not None else f"{int(round(mins))}m remaining"
    except Exception:
        return 'N/A'


try:
    BasicWindow.repredict_on_setup = _repredict_on_setup_impl
    BasicWindow.compute_predicted_current_task_completion = _compute_predicted_current_task_completion_impl
except Exception:
    pass

# Ensure the currently-bound BasicWindow (module-level) has a prediction helper
def _compute_wrapper_for_class(self):
    try:
        est = getattr(self, '_latest_prediction', None)
        if est is None:
            est = run_reprediction()
        if not est:
            return 'N/A'
        mins = est.get('estimated_minutes_remaining')
        conf = est.get('confidence')
        if mins is None:
            return 'Completed'
        return f"{int(round(mins))}m remaining (conf {int(conf*100)}%)" if conf is not None else f"{int(round(mins))}m remaining"
    except Exception:
        return 'N/A'

try:
    BasicWindow.compute_predicted_current_task_completion = _compute_wrapper_for_class
except Exception:
    pass


def run_reprediction():
    """Build behavioral model from DB and return the current completion_estimate (or None)."""
    try:
        from agent.analytics.behavioral_model import BehavioralModel
        from agent.analytics.prediction_hooks import PredictionHooks
        from agent.analytics.persistence import load_sessions
        from datetime import datetime, timezone

        try:
            sessions = load_sessions()
        except Exception:
            sessions = []

        model = BehavioralModel()
        for s in sessions:
            try:
                task_id = getattr(s, 'inferred_task_id', None)
                if not task_id:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        task_id = assign.get('task_id')
                model.update_from_session(s, task_id)
            except Exception:
                try:
                    model.update_from_session(s, None)
                except Exception:
                    pass

        hooks = PredictionHooks(behavioral_model=model)

        active_task = None
        active_duration = 0.0
        for s in sessions:
            try:
                if getattr(s, 'in_progress', False) or getattr(s, 'end', None) is None:
                    assign = getattr(s, 'current_task_assignment', None)
                    if isinstance(assign, dict):
                        active_task = assign.get('task_id')
                    elif getattr(s, 'inferred_task_id', None):
                        active_task = getattr(s, 'inferred_task_id')
                    if getattr(s, 'start', None):
                        active_duration = (datetime.now(timezone.utc) - s.start).total_seconds() / 60.0
                    break
            except Exception:
                continue

        if active_task:
            try:
                res = hooks.on_periodic_update(active_task, active_duration)
                return res.get('completion_estimate')
            except Exception:
                return None

        return None
    except Exception:
        return None
import sys
import threading
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSystemTrayIcon,
    QMenu,
    QTabBar,
    QHBoxLayout,
    QFrame,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QHeaderView,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QAction, QCloseEvent


class BasicWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ecliptous AI")
        self.setGeometry(100, 100, 600, 400)

        # Agent thread
        self.agent_thread = None
        self.agent_stop_flag = threading.Event()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Top tab bar
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Home")
        self.tab_bar.addTab("Tasks")
        self.tab_bar.addTab("Tab B")
        self.tab_bar.addTab("Tab C")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setMovable(False)
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tab_bar)

        # Content area (changes per tab)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        layout.addWidget(self.content_widget)

        # Initialize with Home view
        try:
            self.build_home_view()
        except Exception:
            # Fallback to simple placeholder
            placeholder = QLabel("Home view unavailable")
            placeholder.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(placeholder)

        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))

        # Create tray menu
        tray_menu = QMenu()

        # Show action
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exit_application)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

        # Start the agent in a background thread
        self.start_agent_thread()

        # Periodically refresh the Home view when active (every 3s)
        try:
            self.home_refresh_timer = QTimer(self)
            self.home_refresh_timer.timeout.connect(self._maybe_refresh_home)
            self.home_refresh_timer.start(3000)
        except Exception:
            pass

    def start_agent_thread(self):
        """Start the agent in a background thread"""
        try:
            from main import main as agent_main

            # Create a stop flag for the agent thread
            self.agent_stop_flag = threading.Event()

            def run_agent():
                """Run agent in thread"""
                try:
                    print("[UI] Starting agent in background thread...", flush=True)
                    agent_main(stop_flag=self.agent_stop_flag)
                    print("[UI] Agent thread exited normally", flush=True)
                except Exception as e:
                    print(f"[UI] Agent thread error: {e}", flush=True)

            self.agent_thread = threading.Thread(target=run_agent, daemon=True, name="AgentThread")
            self.agent_thread.start()

            status_text = "✓ Agent Started\nRunning in background"
            if hasattr(self, 'status_label'):
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
            print("[UI] Agent thread started successfully", flush=True)

        except Exception as e:
            error_text = f"✗ Error Starting Agent\n{str(e)}"
            if hasattr(self, 'status_label'):
                self.status_label.setText(error_text)
                self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: red;")
            print(f"[UI] Error starting agent: {e}", flush=True)

    def exit_application(self):
        """Exit the application and stop the agent thread"""
        if hasattr(self, 'status_label'):
            self.status_label.setText("Ending session and stopping agent...")
            self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
        QApplication.processEvents()  # Force UI update

        try:
            # First, end any active session before stopping the agent
            if hasattr(self, 'status_label'):
                self.status_label.setText("Ending active session...")
            QApplication.processEvents()

            try:
                from datetime import datetime, timezone
                from agent.session.sessionizer import SessionManager

                # Initialize session manager to end active session
                session_manager = SessionManager(idle_threshold_seconds=300)

                if session_manager.current_session:
                    session_id = session_manager.current_session.id or session_manager.current_session.session_id
                    print(f"[EXIT] Ending active session: {session_id}", flush=True)

                    # End the session
                    ended_session = session_manager.end_session_if_active(
                        datetime.now(timezone.utc),
                        reason="ui_exit"
                    )

                    if ended_session:
                        print(f"[EXIT] Session {session_id} ended successfully", flush=True)
                        if hasattr(self, 'status_label'):
                            self.status_label.setText("Session ended. Stopping agent...")
                    else:
                        print(f"[EXIT] Failed to end session", flush=True)
                        if hasattr(self, 'status_label'):
                            self.status_label.setText("Failed to end session. Stopping agent...")
                else:
                    print(f"[EXIT] No active session to end", flush=True)
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("No active session. Stopping agent...")

                QApplication.processEvents()
                import time
                time.sleep(1)  # Give time for session to persist

            except Exception as session_err:
                print(f"[EXIT] Error ending session: {session_err}", flush=True)
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Error ending session. Stopping agent anyway...")
                QApplication.processEvents()

            # Now stop the agent thread
            if hasattr(self, 'status_label'):
                self.status_label.setText("Stopping agent thread...")
            QApplication.processEvents()

            # Signal the agent thread to stop
            if hasattr(self, 'agent_stop_flag'):
                self.agent_stop_flag.set()
                print("[UI] Stop signal sent to agent thread", flush=True)

            # The agent thread is a daemon thread, so it will stop when the app exits
            status_text = "✓ Agent Stopped\nExiting..."
            if hasattr(self, 'status_label'):
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
            print("[UI] Agent stopped, exiting application", flush=True)

        except Exception as e:
            error_text = f"✗ Error Stopping Agent\n{str(e)}"
            if hasattr(self, 'status_label'):
                self.status_label.setText(error_text)
                self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: red;")
            print(f"[UI] Error stopping agent: {e}", flush=True)
        finally:
            # Quit the application after a brief delay to show status
            QApplication.processEvents()
            import time
            time.sleep(0.5)
            QApplication.quit()

    def closeEvent(self, event: QCloseEvent):
        """Override close event to minimize to tray instead of exiting"""
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Ecliptous AI",
            "Application minimized to tray",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )

    def on_tray_icon_activated(self, reason):
        """Show window when tray icon is double-clicked"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()

    def on_tab_changed(self, index: int):
        """Handle tab selection changes (simple placeholder behavior)."""
        name = self.tab_bar.tabText(index) if index >= 0 else ""
        # Update title label to reflect selected tab
        if name:
            if hasattr(self, 'title_label'):
                self.title_label.setText(f"Ecliptous AI - {name}")
        else:
            if hasattr(self, 'title_label'):
                self.title_label.setText("Ecliptous AI - Productivity Agent")
        # Rebuild content area based on selected tab
        try:
            if name == "Home":
                self.build_home_view()
            elif name == "Tasks":
                self.build_tasks_view()
            else:
                self.build_placeholder_view(name)
        except Exception as e:
            # If building view fails, show error placeholder
            self.clear_content()
            err = QLabel(f"Error building view: {e}")
            err.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(err)

    def clear_content(self):
        """Remove all widgets from the content layout."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def build_home_view(self):
        """Build the Home tab UI: current state, key info, and three graph placeholders."""
        self.clear_content()

        # Current state / current task area
        top = QWidget()
        top_l = QHBoxLayout()
        top.setLayout(top_l)
        try:
            top.setMaximumHeight(160)
        except Exception:
            pass

        # Left: current state
        state_frame = QFrame()
        state_frame.setFrameShape(QFrame.StyledPanel)
        state_layout = QVBoxLayout()
        state_frame.setLayout(state_layout)
        self.current_state_label = QLabel("Current State: Unknown")
        self.current_state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        state_layout.addWidget(self.current_state_label)
        self.current_task_label = QLabel("Current Task: Unknown")
        state_layout.addWidget(self.current_task_label)
        state_layout.addStretch()

        # Right: key info list
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout()
        info_frame.setLayout(info_layout)
        info_title = QLabel("Key Info")
        info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        info_layout.addWidget(info_title)

        # Populate with simple computed values where possible
        peak_focus = self.compute_peak_focus_hours()
        most_prod = self.compute_most_productive_session_durations()
        predicted = self.compute_predicted_current_task_completion()

        info_layout.addWidget(QLabel(f"Peak Focus Hours: {peak_focus}"))
        info_layout.addWidget(QLabel(f"Most Productive Session Durations: {most_prod}"))
        info_layout.addWidget(QLabel(f"Predicted Current Task Completion Time: {predicted}"))
        info_layout.addStretch()

        top_l.addWidget(state_frame, 1)
        top_l.addWidget(info_frame, 1)

        self.content_layout.addWidget(top)

        # Graph area: single large Overview graph in the free space
        graphs = QWidget()
        graphs_l = QVBoxLayout()
        graphs.setLayout(graphs_l)
        try:
            graphs.setMinimumHeight(420)
        except Exception:
            pass
        try:
            graphs.setStyleSheet("background: #ffffff;")
        except Exception:
            pass

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QVBoxLayout()
        frame.setLayout(fl)

        # Try rendering analytics model to a pixmap for reliable display
        pixmap_widget = None
        try:
            from agent.analytics.graph import render_analytics_graph
            model = render_analytics_graph(
                graph_type="time_series",
                metric="focus",
                time_window=30,
                comparison_mode=None,
                options={"moving_average": 7},
            )
            rp = _render_model_to_pixmap(model, width=760, height=380)
            if rp:
                lbl = QLabel()
                lbl.setPixmap(rp)
                lbl.setScaledContents(True)
                try:
                    lbl.setMinimumSize(760, 380)
                except Exception:
                    pass
                pixmap_widget = lbl
        except Exception:
            pixmap_widget = None

        if not pixmap_widget:
            # fallback: deterministic pixmap so the area is always visible
            pix = _create_test_pixmap("Overview", width=760, height=380)
            if pix:
                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setScaledContents(True)
            else:
                lbl = QLabel("Overview Graph Placeholder")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background:#f5f7ff; border: 1px solid #ddd; font-size:16px; padding:24px;")
            pixmap_widget = lbl

        fl.addWidget(pixmap_widget)
        graphs_l.addWidget(frame)

        self.content_layout.addWidget(graphs, 2)
        try:
            self.content_widget.update()
            self.content_widget.repaint()
        except Exception:
            pass

        # Footer note directing to Prediction tab
        footer = QLabel("Go to Prediction Tab for full predictions")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-size: 12px; color: #555; padding: 8px;")
        self.content_layout.addWidget(footer)

        # Try to populate current state/task from DB if available
        try:
            from agent.storage import db

            # Prefer checking for an active session first
            try:
                from agent.session.sessionizer import SessionManager
                sm = SessionManager()
                active_session = sm.current_session
            except Exception:
                active_session = None

            display = None
            color = None

            if active_session:
                display = 'Collecting Data'
                color = 'green'
            else:
                # Fallback to recent state events
                try:
                    hist = db.get_state_history()
                    latest = hist[-1] if hist else None
                    state_val = latest[1] if latest else None
                except Exception:
                    state_val = None

                if state_val:
                    s = str(state_val).lower()
                    if 'active_aligned' in s or 'aligned' in s:
                        display = 'Collecting Data - Task Detected'
                        color = 'green'
                    elif 'active_unaligned' in s or 'unalign' in s or 'drift' in s:
                        display = 'Collecting Data - Task Not Detected'
                        color = 'orange'
                    elif 'active' in s:
                        display = 'Active'
                        color = 'green'
                    elif 'paused' in s or 'pause' in s:
                        display = 'Paused'
                        color = '#666'
                    elif 'idle' in s:
                        display = 'Idle'
                        color = '#888'
                    else:
                        display = state_val

            if not display:
                try:
                    sessions = db.load_sessions_from_db(days_back=1)
                    if sessions:
                        s = sessions[0]
                        display = 'In Progress' if getattr(s, 'in_progress', False) else 'Idle'
                        color = '#333'
                except Exception:
                    pass

            if display:
                try:
                    self.current_state_label.setText(f"Current State: {display}")
                    if color:
                        self.current_state_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
                except Exception:
                    pass

            # Update current task label
            try:
                sessions = db.load_sessions_from_db(days_back=1)
                if sessions:
                    s = sessions[0]
                    assign = getattr(s, 'current_task_assignment', None)
                    if assign and isinstance(assign, dict):
                        task = assign.get('task_id') or assign.get('task_name') or 'Unknown'
                        conf = assign.get('confidence')
                        self.current_task_label.setText(f"Current Task: {task} (conf={conf:.2f})" if conf is not None else f"Current Task: {task}")
            except Exception:
                pass

            # Read latest STATE_CHANGE event to show live intensity
            try:
                events = db.get_events(event_type='STATE_CHANGE', limit=1)
                if events:
                    ts, evt_type, payload = events[0]
                    intensity = None
                    if isinstance(payload, dict) and payload.get('intensity') is not None:
                        intensity = payload.get('intensity')
                else:
                    intensity = None

                if intensity is None:
                    self.current_intensity_label.setText("Intensity: N/A")
                else:
                    self.current_intensity_label.setText(f"Intensity: {float(intensity):.2f}")
            except Exception:
                pass

        except Exception:
            pass

    def _maybe_refresh_home(self):
        """Rebuild Home view only when Home tab is selected to fetch latest data."""
        try:
            # Current tab index 0 is Home (as initialized)
            if getattr(self, 'tab_bar', None) and self.tab_bar.currentIndex() == 0:
                # Rebuild the home view to pull fresh data from storage
                self.build_home_view()
        except Exception:
            pass

    def build_tasks_view(self):
        """Minimal Tasks tab placeholder for now."""
        self.clear_content()

        # Tree widget to list tasks (expand to see details)
        tree = QTreeWidget()
        tree.setColumnCount(6)
        tree.setHeaderLabels(["Task ID", "Session ID", "Start", "End", "Duration (m)", "Confidence"])
        tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        tree.header().setStretchLastSection(False)

        # Load sessions and their intra-session tasks
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=30)
        except Exception:
            sessions = []

        count = 0
        for s in sessions:
            sess_id = getattr(s, 'session_id', None) or getattr(s, 'id', None)
            segments = getattr(s, 'intra_session_tasks', []) or []
            for seg in segments:
                count += 1
                start = getattr(seg, 'start_time', None)
                end = getattr(seg, 'end_time', None)
                dur = None
                try:
                    if start and end:
                        dur = int((end - start).total_seconds() / 60)
                except Exception:
                    dur = None

                conf = getattr(seg, 'confidence', None)
                top = QTreeWidgetItem([
                    str(getattr(seg, 'task_id', 'unknown')),
                    str(sess_id),
                    start.isoformat() if start else '',
                    end.isoformat() if end else '',
                    str(dur) if dur is not None else '',
                    f"{conf:.2f}" if isinstance(conf, float) else str(conf or ''),
                ])

                # Child with detailed JSON/text
                try:
                    detail = {
                        'task_id': getattr(seg, 'task_id', None),
                        'base_category': getattr(seg, 'base_category', None),
                        'app': getattr(seg, 'app', None),
                        'window_title': getattr(seg, 'window_title', None),
                        'normalized_title': getattr(seg, 'normalized_title', None),
                        'start_time': start.isoformat() if start else None,
                        'end_time': end.isoformat() if end else None,
                        'duration_minutes': dur,
                        'confidence': conf,
                        'reason': getattr(seg, 'reason', None),
                        'distance_to_centroid': getattr(seg, 'distance_to_centroid', None),
                        'feature_vector': getattr(seg, 'feature_vector', None),
                        # Any extra metadata attached during inference/clustering
                        'metadata': getattr(seg, 'metadata', None),
                        # Normalized numeric view (if metadata provides it)
                        'normalized_metadata': None,
                    }

                    # Try to extract normalized view from metadata if available
                    md = getattr(seg, 'metadata', None)
                    if isinstance(md, dict):
                        detail['normalized_metadata'] = md.get('normalized') or md.get('normalized_metadata') or None
                except Exception:
                    detail = {'repr': repr(seg)}

                child = QTreeWidgetItem()
                top.addChild(child)
                tree.addTopLevelItem(top)

                # Use a QTextEdit as a widget for the child (spanning columns)
                widget = QTextEdit()
                widget.setReadOnly(True)
                import json as _json
                try:
                    widget.setText(_json.dumps(detail, indent=2, default=str))
                except Exception:
                    widget.setText(str(detail))

                tree.setItemWidget(child, 0, widget)
                try:
                    # Prefer setting on the item — some PySide6 builds don't expose
                    # QTreeWidget.setFirstItemColumnSpanned; use the item method.
                    child.setFirstColumnSpanned(True)
                except Exception:
                    # Fallback: ignore if not available
                    pass

        if count == 0:
            lbl = QLabel("No tasks found in the last 30 days.")
            lbl.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(lbl)
        else:
            self.content_layout.addWidget(tree)

    def build_placeholder_view(self, name: str):
        self.clear_content()
        lbl = QLabel(f"{name} content coming soon")
        lbl.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(lbl)

    # ------------ Simple data helpers (best-effort, fast computations) ------------
    def compute_peak_focus_hours(self):
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=14)
            hours = {}
            for s in sessions:
                if hasattr(s, 'start') and s.start:
                    h = s.start.hour
                    dur = ((s.end - s.start).total_seconds()/60.0) if s.end and s.start else 0
                    hours[h] = hours.get(h, 0) + dur
            if not hours:
                return 'N/A'
            top = sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3]
            return ", ".join(f"{h}:00({int(m)}m)" for h, m in top)
        except Exception:
            return 'N/A'

    def compute_most_productive_session_durations(self):
        try:
            from agent.storage import db
            sessions = db.load_sessions_from_db(days_back=30)
            durations = []
            for s in sessions:
                if hasattr(s, 'start') and hasattr(s, 'end') and s.start and s.end:
                    minutes = (s.end - s.start).total_seconds()/60.0
                    durations.append(minutes)
            if not durations:
                return 'N/A'
            top = sorted(durations, reverse=True)[:3]
            return ", ".join(f"{int(m)}m" for m in top)
        except Exception:
            return 'N/A'

    def compute_predicted_current_task_completion(self):
        # Use cached prediction from setup if available, otherwise attempt a quick re-predict
        est = getattr(self, '_latest_prediction', None)
        if est is None:
            try:
                from main_window import run_reprediction as _run_reprediction
                est = _run_reprediction()
            except Exception:
                est = None

        if not est:
            return 'N/A'

        try:
            mins = est.get('estimated_minutes_remaining')
            conf = est.get('confidence')
            if mins is None:
                return 'Completed'
            return f"{int(round(mins))}m remaining (conf {int(conf*100)}%)" if conf is not None else f"{int(round(mins))}m remaining"
        except Exception:
            return 'N/A'


def main():
    app = QApplication(sys.argv)
    window = BasicWindow()
    # Show the window on start so users see the UI immediately
    window.show()
    print('[UI] main_window started', flush=True)
    window.tray_icon.showMessage(
        "Ecliptous AI",
        "Application started in system tray",
        QSystemTrayIcon.MessageIcon.Information,
        2000
    )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
import sys
import threading
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSystemTrayIcon,
    QMenu,
    QTabBar,
    QHBoxLayout,
    QFrame,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QHeaderView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QCloseEvent


class BasicWindow(QMainWindow):
    def __init__(self):
        import sys
        import threading
        from PySide6.QtWidgets import (
            QApplication,
            QMainWindow,
            QLabel,
            QVBoxLayout,
            QWidget,
            QSystemTrayIcon,
            QMenu,
            QTabBar,
            QHBoxLayout,
            QFrame,
            QTreeWidget,
            QTreeWidgetItem,
            QTextEdit,
            QHeaderView,
        )
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QIcon, QAction, QCloseEvent


        class BasicWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Ecliptous AI")
                self.setGeometry(100, 100, 600, 400)
        
                # Agent thread
                self.agent_thread = None
                self.agent_stop_flag = threading.Event()
        
                # Create central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
        
                # Create layout
                layout = QVBoxLayout()
                central_widget.setLayout(layout)
        
                # Top tab bar
                self.tab_bar = QTabBar()
                self.tab_bar.addTab("Home")
                self.tab_bar.addTab("Tasks")
                self.tab_bar.addTab("Tab B")
                self.tab_bar.addTab("Tab C")
                self.tab_bar.setExpanding(True)
                self.tab_bar.setMovable(False)
                self.tab_bar.currentChanged.connect(self.on_tab_changed)
                layout.addWidget(self.tab_bar)

     

                # Content area (changes per tab)
                self.content_widget = QWidget()
                self.content_layout = QVBoxLayout()
                self.content_widget.setLayout(self.content_layout)
                layout.addWidget(self.content_widget)

                # Initialize with Home view
                try:
                    self.build_home_view()
                except Exception:
                    # Fallback to simple placeholder
                    placeholder = QLabel("Home view unavailable")
                    placeholder.setAlignment(Qt.AlignCenter)
                    self.content_layout.addWidget(placeholder)
        
                # Create system tray icon
                self.tray_icon = QSystemTrayIcon(self)
                self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        
                # Create tray menu
                tray_menu = QMenu()
        
                # Show action
                show_action = QAction("Show", self)
                show_action.triggered.connect(self.show)
                tray_menu.addAction(show_action)
        
                # Exit action
                exit_action = QAction("Exit", self)
                exit_action.triggered.connect(self.exit_application)
                tray_menu.addAction(exit_action)
        
                self.tray_icon.setContextMenu(tray_menu)
                self.tray_icon.activated.connect(self.on_tray_icon_activated)
                self.tray_icon.show()
        
                # Start the agent in a background thread
                self.start_agent_thread()
    
            def start_agent_thread(self):
                """Start the agent in a background thread"""
                try:
                    from main import main as agent_main
            
                    # Create a stop flag for the agent thread
                    self.agent_stop_flag = threading.Event()
            
                    def run_agent():
                        """Run agent in thread"""
                        try:
                            print("[UI] Starting agent in background thread...", flush=True)
                            agent_main(stop_flag=self.agent_stop_flag)
                            print("[UI] Agent thread exited normally", flush=True)
                        except Exception as e:
                            print(f"[UI] Agent thread error: {e}", flush=True)
            
                    self.agent_thread = threading.Thread(target=run_agent, daemon=True, name="AgentThread")
                    self.agent_thread.start()
            
                    status_text = "✓ Agent Started\nRunning in background"
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(status_text)
                        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
                    print("[UI] Agent thread started successfully", flush=True)
            
                except Exception as e:
                    error_text = f"✗ Error Starting Agent\n{str(e)}"
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(error_text)
                        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: red;")
                    print(f"[UI] Error starting agent: {e}", flush=True)
    
            def exit_application(self):
                """Exit the application and stop the agent thread"""
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Ending session and stopping agent...")
                    self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
                QApplication.processEvents()  # Force UI update
        
                try:
                    # First, end any active session before stopping the agent
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("Ending active session...")
                    QApplication.processEvents()
            
                    try:
                        from datetime import datetime, timezone
                        from agent.session.sessionizer import SessionManager
                
                        # Initialize session manager to end active session
                        session_manager = SessionManager(idle_threshold_seconds=300)
                
                        if session_manager.current_session:
                            session_id = session_manager.current_session.id or session_manager.current_session.session_id
                            print(f"[EXIT] Ending active session: {session_id}", flush=True)
                    
                            # End the session
                            ended_session = session_manager.end_session_if_active(
                                datetime.now(timezone.utc), 
                                reason="ui_exit"
                            )
                    
                            if ended_session:
                                print(f"[EXIT] Session {session_id} ended successfully", flush=True)
                                if hasattr(self, 'status_label'):
                                    self.status_label.setText("Session ended. Stopping agent...")
                            else:
                                print(f"[EXIT] Failed to end session", flush=True)
                                if hasattr(self, 'status_label'):
                                    self.status_label.setText("Failed to end session. Stopping agent...")
                        else:
                            print(f"[EXIT] No active session to end", flush=True)
                            if hasattr(self, 'status_label'):
                                self.status_label.setText("No active session. Stopping agent...")
                
                        QApplication.processEvents()
                        import time
                        time.sleep(1)  # Give time for session to persist
                
                    except Exception as session_err:
                        print(f"[EXIT] Error ending session: {session_err}", flush=True)
                        if hasattr(self, 'status_label'):
                            self.status_label.setText("Error ending session. Stopping agent anyway...")
                        QApplication.processEvents()
            
                    # Now stop the agent thread
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("Stopping agent thread...")
                    QApplication.processEvents()
            
                    # Signal the agent thread to stop
                    if hasattr(self, 'agent_stop_flag'):
                        self.agent_stop_flag.set()
                        print("[UI] Stop signal sent to agent thread", flush=True)
            
                    # The agent thread is a daemon thread, so it will stop when the app exits
                    status_text = "✓ Agent Stopped\nExiting..."
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(status_text)
                        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
                    print("[UI] Agent stopped, exiting application", flush=True)
            
                except Exception as e:
                    error_text = f"✗ Error Stopping Agent\n{str(e)}"
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(error_text)
                        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; color: red;")
                    print(f"[UI] Error stopping agent: {e}", flush=True)
                finally:
                    # Quit the application after a brief delay to show status
                    QApplication.processEvents()
                    import time
                    time.sleep(0.5)
                    QApplication.quit()
    
            def closeEvent(self, event: QCloseEvent):
                """Override close event to minimize to tray instead of exiting"""
                event.ignore()
                self.hide()
                self.tray_icon.showMessage(
                    "Ecliptous AI",
                    "Application minimized to tray",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
    
            def on_tray_icon_activated(self, reason):
                """Show window when tray icon is double-clicked"""
                if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
                    self.show()
                    self.activateWindow()
        


            def on_tab_changed(self, index: int):
                """Handle tab selection changes (simple placeholder behavior)."""
                name = self.tab_bar.tabText(index) if index >= 0 else ""
                # Update title label to reflect selected tab
                if name:
                    if hasattr(self, 'title_label'):
                        self.title_label.setText(f"Ecliptous AI - {name}")
                else:
                    if hasattr(self, 'title_label'):
                        self.title_label.setText("Ecliptous AI - Productivity Agent")
                # Rebuild content area based on selected tab
                try:
                    if name == "Home":
                        self.build_home_view()
                    elif name == "Tasks":
                        self.build_tasks_view()
                    else:
                        self.build_placeholder_view(name)
                except Exception as e:
                    # If building view fails, show error placeholder
                    self.clear_content()
                    err = QLabel(f"Error building view: {e}")
                    err.setAlignment(Qt.AlignCenter)
                    self.content_layout.addWidget(err)

            def clear_content(self):
                """Remove all widgets from the content layout."""
                while self.content_layout.count():
                    item = self.content_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.setParent(None)

            def build_home_view(self):
                """Build the Home tab UI: current state, key info, and three graph placeholders."""
                self.clear_content()

                # Current state / current task area
                top = QWidget()
                top_l = QHBoxLayout()
                top.setLayout(top_l)
                try:
                    top.setMaximumHeight(160)
                except Exception:
                    pass

                # Left: current state
                state_frame = QFrame()
                state_frame.setFrameShape(QFrame.StyledPanel)
                state_layout = QVBoxLayout()
                state_frame.setLayout(state_layout)
                self.current_state_label = QLabel("Current State: Unknown")
                self.current_state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
                state_layout.addWidget(self.current_state_label)
                self.current_task_label = QLabel("Current Task: Unknown")
                state_layout.addWidget(self.current_task_label)
                # Live intensity score under the task name
                self.current_intensity_label = QLabel("Intensity: N/A")
                self.current_intensity_label.setStyleSheet("color: #555;")
                state_layout.addWidget(self.current_intensity_label)
                state_layout.addStretch()

                # Right: key info list
                info_frame = QFrame()
                info_frame.setFrameShape(QFrame.StyledPanel)
                info_layout = QVBoxLayout()
                info_frame.setLayout(info_layout)
                info_title = QLabel("Key Info")
                info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
                info_layout.addWidget(info_title)

                # Populate with simple computed values where possible
                peak_focus = self.compute_peak_focus_hours()
                most_prod = self.compute_most_productive_session_durations()
                predicted = self.compute_predicted_current_task_completion()

                info_layout.addWidget(QLabel(f"Peak Focus Hours: {peak_focus}"))
                info_layout.addWidget(QLabel(f"Most Productive Session Durations: {most_prod}"))
              
                info_layout.addStretch()

                top_l.addWidget(state_frame, 1)
                top_l.addWidget(info_frame, 1)

                self.content_layout.addWidget(top)

                # Graph area: single large Overview graph in the free space
                graphs = QWidget()
                graphs_l = QVBoxLayout()
                graphs.setLayout(graphs_l)
                try:
                    graphs.setMinimumHeight(420)
                except Exception:
                    pass

                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                fl = QVBoxLayout()
                frame.setLayout(fl)

                pixmap_widget = None
                try:
                    from agent.analytics.graph import render_analytics_graph
                    model = render_analytics_graph(
                        graph_type="time_series",
                        metric="focus",
                        time_window=30,
                        comparison_mode=None,
                        options={"moving_average": 7},
                    )
                    rp = _render_model_to_pixmap(model, width=760, height=380)
                    if rp:
                        lbl = QLabel()
                        lbl.setPixmap(rp)
                        lbl.setScaledContents(True)
                        try:
                            lbl.setMinimumSize(760, 380)
                        except Exception:
                            pass
                        pixmap_widget = lbl
                except Exception:
                    pixmap_widget = None

                if not pixmap_widget:
                    pix = _create_test_pixmap("Overview", width=760, height=380)
                    if pix:
                        lbl = QLabel()
                        lbl.setPixmap(pix)
                        lbl.setScaledContents(True)
                    else:
                        lbl = QLabel("Overview Graph Placeholder")
                        lbl.setAlignment(Qt.AlignCenter)
                        lbl.setStyleSheet("background:#f5f7ff; border: 1px solid #ddd; font-size:16px; padding:24px;")
                    pixmap_widget = lbl

                fl.addWidget(pixmap_widget)
                graphs_l.addWidget(frame)
                self.content_layout.addWidget(graphs, 2)

                # Footer note directing to Prediction tab
                footer = QLabel("Go to Prediction Tab for full predictions")
                footer.setAlignment(Qt.AlignCenter)
                footer.setStyleSheet("font-size: 12px; color: #555; padding: 8px;")
                self.content_layout.addWidget(footer)

                # Try to populate current state/task from DB if available
                try:
                    from agent.storage import db
                    sessions = db.load_sessions_from_db(days_back=1)
                    if sessions:
                        s = sessions[0]
                        # Update labels
                        self.current_state_label.setText(f"Current State: {'In Progress' if getattr(s, 'in_progress', False) else 'Idle'}")
                        assign = getattr(s, 'current_task_assignment', None)
                        if assign and isinstance(assign, dict):
                            task = assign.get('task_id') or assign.get('task_name') or 'Unknown'
                            conf = assign.get('confidence')
                            self.current_task_label.setText(f"Current Task: {task} (conf={conf:.2f})" if conf is not None else f"Current Task: {task}")
                            # Try to show intensity from assignment.features if available
                            intensity_val = None
                            try:
                                features = assign.get('features', {}) or {}
                                if isinstance(features, dict) and features.get('intensity') is not None:
                                    intensity_val = float(features.get('intensity'))
                            except Exception:
                                intensity_val = None

                            if intensity_val is None:
                                try:
                                    timeline = getattr(s, 'timeline', {}) or {}
                                    from datetime import datetime, timezone, timedelta
                                    now = datetime.now(timezone.utc)
                                    cutoff = now - timedelta(seconds=60)
                                    recent = []
                                    for k, v in (timeline.items() if isinstance(timeline, dict) else []):
                                        try:
                                            entry_time = k if isinstance(k, datetime) else datetime.fromisoformat(k)
                                            if entry_time.tzinfo is None:
                                                entry_time = entry_time.replace(tzinfo=timezone.utc)
                                            if entry_time >= cutoff:
                                                recent.append(v)
                                        except Exception:
                                            continue
                                    if recent:
                                        total_input = sum((it.get('keys', 0) + it.get('clicks', 0)) for it in recent if isinstance(it, dict))
                                        buckets = len(recent)
                                        intensity_val = min(total_input / (buckets * 60) if buckets else 0, 1.0)
                                except Exception:
                                    intensity_val = None

                            if intensity_val is None:
                                self.current_intensity_label.setText("Intensity: N/A")
                            else:
                                self.current_intensity_label.setText(f"Intensity: {intensity_val:.2f}")
                except Exception:
                    pass

            def build_tasks_view(self):
                """Minimal Tasks tab placeholder for now."""
                self.clear_content()

                # Tree widget to list tasks (expand to see details)
                tree = QTreeWidget()
                tree.setColumnCount(6)
                tree.setHeaderLabels(["Task ID", "Session ID", "Start", "End", "Duration (m)", "Confidence"])
                tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
                tree.header().setStretchLastSection(False)

                # Load sessions and their intra-session tasks
                try:
                    from agent.storage import db
                    sessions = db.load_sessions_from_db(days_back=30)
                except Exception:
                    sessions = []

                count = 0
                for s in sessions:
                    sess_id = getattr(s, 'session_id', None) or getattr(s, 'id', None)
                    segments = getattr(s, 'intra_session_tasks', []) or []
                    for seg in segments:
                        count += 1
                        start = getattr(seg, 'start_time', None)
                        end = getattr(seg, 'end_time', None)
                        dur = None
                        try:
                            if start and end:
                                dur = int((end - start).total_seconds() / 60)
                        except Exception:
                            dur = None

                        conf = getattr(seg, 'confidence', None)
                        top = QTreeWidgetItem([
                            str(getattr(seg, 'task_id', 'unknown')),
                            str(sess_id),
                            start.isoformat() if start else '',
                            end.isoformat() if end else '',
                            str(dur) if dur is not None else '',
                            f"{conf:.2f}" if isinstance(conf, float) else str(conf or ''),
                        ])

                        # Child with detailed JSON/text
                        try:
                            detail = {
                                'task_id': getattr(seg, 'task_id', None),
                                'base_category': getattr(seg, 'base_category', None),
                                'app': getattr(seg, 'app', None),
                                'window_title': getattr(seg, 'window_title', None),
                                'normalized_title': getattr(seg, 'normalized_title', None),
                                'start_time': start.isoformat() if start else None,
                                'end_time': end.isoformat() if end else None,
                                'duration_minutes': dur,
                                'confidence': conf,
                                'reason': getattr(seg, 'reason', None),
                                'distance_to_centroid': getattr(seg, 'distance_to_centroid', None),
                                'feature_vector': getattr(seg, 'feature_vector', None),
                                # Any extra metadata attached during inference/clustering
                                'metadata': getattr(seg, 'metadata', None),
                                # Normalized numeric view (if metadata provides it)
                                'normalized_metadata': None,
                            }

                            # Try to extract normalized view from metadata if available
                            md = getattr(seg, 'metadata', None)
                            if isinstance(md, dict):
                                detail['normalized_metadata'] = md.get('normalized') or md.get('normalized_metadata') or None
                        except Exception:
                            detail = {'repr': repr(seg)}

                        child = QTreeWidgetItem()
                        top.addChild(child)
                        tree.addTopLevelItem(top)

                        # Use a QTextEdit as a widget for the child (spanning columns)
                        widget = QTextEdit()
                        widget.setReadOnly(True)
                        import json as _json
                        try:
                            widget.setText(_json.dumps(detail, indent=2, default=str))
                        except Exception:
                            widget.setText(str(detail))

                        tree.setItemWidget(child, 0, widget)
                        try:
                            # Prefer setting on the item — some PySide6 builds don't expose
                            # QTreeWidget.setFirstItemColumnSpanned; use the item method.
                            child.setFirstColumnSpanned(True)
                        except Exception:
                            # Fallback: ignore if not available
                            pass

                if count == 0:
                    lbl = QLabel("No tasks found in the last 30 days.")
                    lbl.setAlignment(Qt.AlignCenter)
                    self.content_layout.addWidget(lbl)
                else:
                    self.content_layout.addWidget(tree)

            def build_placeholder_view(self, name: str):
                self.clear_content()
                lbl = QLabel(f"{name} content coming soon")
                lbl.setAlignment(Qt.AlignCenter)
                self.content_layout.addWidget(lbl)

            # ------------ Simple data helpers (best-effort, fast computations) ------------
            def compute_peak_focus_hours(self):
                try:
                    from agent.storage import db
                    sessions = db.load_sessions_from_db(days_back=14)
                    hours = {}
                    for s in sessions:
                        if hasattr(s, 'start') and s.start:
                            h = s.start.hour
                            dur = ((s.end - s.start).total_seconds()/60.0) if s.end and s.start else 0
                            hours[h] = hours.get(h, 0) + dur
                    if not hours:
                        return 'N/A'
                    top = sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3]
                    return ", ".join(f"{h}:00({int(m)}m)" for h, m in top)
                except Exception:
                    return 'N/A'

            def compute_most_productive_session_durations(self):
                try:
                    from agent.storage import db
                    sessions = db.load_sessions_from_db(days_back=30)
                    durations = []
                    for s in sessions:
                        if hasattr(s, 'start') and hasattr(s, 'end') and s.start and s.end:
                            minutes = (s.end - s.start).total_seconds()/60.0
                            durations.append(minutes)
                    if not durations:
                        return 'N/A'
                    top = sorted(durations, reverse=True)[:3]
                    return ", ".join(f"{int(m)}m" for m in top)
                except Exception:
                    return 'N/A'

            def compute_predicted_current_task_completion(self):
                # Use cached prediction from setup if available, otherwise attempt a quick re-predict
                est = getattr(self, '_latest_prediction', None)
                if est is None:
                    try:
                        from main_window import run_reprediction as _run_reprediction
                        est = _run_reprediction()
                    except Exception:
                        est = None

                if not est:
                    return 'N/A'

                try:
                    mins = est.get('estimated_minutes_remaining')
                    conf = est.get('confidence')
                    if mins is None:
                        return 'Completed'
                    return f"{int(round(mins))}m remaining (conf {int(conf*100)}%)" if conf is not None else f"{int(round(mins))}m remaining"
                except Exception:
                    return 'N/A'


        def main():
            app = QApplication(sys.argv)
            window = BasicWindow()
            # Show the window on start so users see the UI immediately
            window.show()
            print('[UI] main_window started', flush=True)
            window.tray_icon.showMessage(
                "Ecliptous AI",
                "Application started in system tray",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
            sys.exit(app.exec())


        if __name__ == "__main__":
            main()

# Final attach: ensure whichever BasicWindow is last gets graph helpers
try:
    BasicWindow._ensure_graph_canvases = _module_ensure_graph_canvases
    BasicWindow.update_graphs = _module_update_graphs
except Exception:
    pass

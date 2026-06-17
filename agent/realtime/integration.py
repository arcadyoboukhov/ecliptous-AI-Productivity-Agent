"""
Real-Time Integration Pipeline

Connects:
- Collector → intervals (already persisted)
- Normalizer → interval aggregation
- Clustering → task assignment in near-real-time
- Analytics → realtime insights snapshots
- UI → latest analytics snapshot for dashboard
"""

from datetime import datetime, timezone, timedelta
import threading
import time
from typing import Optional

from agent.analytics.insights import generate_insights_report


class RealTimeIntegrationRunner(threading.Thread):
    """
    Background runner to produce realtime analytics and task snapshots.
    """
    def __init__(self, interval_seconds: int = 60, lookback_minutes: int = 120):
        super().__init__(daemon=True)
        self.interval_seconds = interval_seconds
        self.lookback_minutes = lookback_minutes
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                now = datetime.now(timezone.utc)
                start_time = now - timedelta(minutes=self.lookback_minutes)

                # Generate realtime insights and persist snapshot
                generate_insights_report(
                    start_time=start_time,
                    end_time=now,
                    limit=5000,
                    clustering_method='dbscan',
                    focus_threshold=0.6,
                    audio_engagement_threshold=0.5,
                    persist_enriched=True,
                    persist_snapshot=True,
                )
            except Exception:
                pass

            self._stop.wait(self.interval_seconds)

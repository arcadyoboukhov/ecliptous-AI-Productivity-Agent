"""
Data Maintenance & Integrity Layer

Implements:
1. Session/task versioning with schema version tracking
2. Backward compatibility for older sessions/tasks
3. Automated cleanup of stale sessions
4. Session aggregation into behavioral baselines
5. Data validation and repair

This module handles:
- Versioning schema in sessions.json and tasks.json
- Migrating data between schema versions
- Removing sessions older than retention period
- Summarizing old sessions into aggregate stats
- Integrity checks and repair
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Current schema versions
SESSIONS_SCHEMA_VERSION = "2.0"
TASKS_SCHEMA_VERSION = "1.0"

# Data retention policy
DEFAULT_RETENTION_DAYS = 365  # Keep sessions for 1 year
DEFAULT_ARCHIVE_THRESHOLD_DAYS = 180  # Archive sessions older than 6 months

# Files
SESSIONS_FILE = "sessions.json"
TASKS_FILE = "tasks.json"
SESSION_ARCHIVES_DIR = "session_archives"

# ============================================================================
# Schema Definitions
# ============================================================================

class SchemaVersion:
    """Represents a schema version with migration path."""
    
    def __init__(self, version: str, fields: List[str], deprecated_fields: List[str] = None):
        """
        Args:
            version: Version string (e.g., "1.0", "2.0")
            fields: Current fields in this version
            deprecated_fields: Fields removed in this version
        """
        self.version = version
        self.fields = set(fields)
        self.deprecated_fields = set(deprecated_fields or [])
    
    def is_compatible(self, incoming_version: str) -> bool:
        """Check if incoming_version is compatible with this schema."""
        # 1.0 can be read by 1.x, 2.0
        # 2.0 requires explicit migration from 1.0
        major_current = self.version.split(".")[0]
        major_incoming = incoming_version.split(".")[0]
        return major_incoming <= major_current


# Session schema versions
SESSIONS_SCHEMA_V1 = SchemaVersion(
    "1.0",
    fields=[
        "id", "start", "end", "device_id", "in_progress",
        "apps", "event_count", "input_events", "timeline", 
        "intent_breakdown", "intent_segments"
    ]
)

SESSIONS_SCHEMA_V2 = SchemaVersion(
    "2.0",
    fields=[
        "id", "start", "end", "device_id", "in_progress",
        "schema_version", "migrated_at"
    ],
    deprecated_fields=["apps", "event_count", "input_events", "timeline", "intent_breakdown"]
)

# Task schema versions
TASKS_SCHEMA_V1 = SchemaVersion(
    "1.0",
    fields=[
        "id", "name", "created_at", "updated_at", "is_unstable",
        "schema_version", "migrated_at"
    ]
)


# ============================================================================
# Data Maintenance Class
# ============================================================================

class DataMaintenance:
    """Handles data versioning, backward compatibility, and cleanup."""
    
    def __init__(self, sessions_file: str = SESSIONS_FILE, tasks_file: str = TASKS_FILE):
        """Initialize maintenance layer.
        
        Args:
            sessions_file: Path to sessions.json
            tasks_file: Path to tasks.json
        """
        self.sessions_file = sessions_file
        self.tasks_file = tasks_file
        self.session_archives_dir = SESSION_ARCHIVES_DIR
        
        # Ensure archive directory exists
        os.makedirs(self.session_archives_dir, exist_ok=True)
    
    # ========================================================================
    # Schema Versioning
    # ========================================================================
    
    def get_sessions_schema_version(self) -> str:
        """Get current schema version from sessions.json."""
        if not os.path.exists(self.sessions_file):
            return SESSIONS_SCHEMA_VERSION
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            return data.get("schema_version", "1.0")
        except Exception as e:
            logger.warning(f"Could not read schema version: {e}")
            return "1.0"
    
    def get_tasks_schema_version(self) -> str:
        """Get current schema version from tasks.json."""
        if not os.path.exists(self.tasks_file):
            return TASKS_SCHEMA_VERSION
        
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
            return data.get("schema_version", "1.0")
        except Exception as e:
            logger.warning(f"Could not read schema version: {e}")
            return "1.0"
    
    # ========================================================================
    # Schema Migration
    # ========================================================================
    
    def migrate_sessions_schema(self) -> bool:
        """Migrate sessions from old schema to current version.
        
        Handles:
        - v1.0 → v2.0: Remove activity fields (now in Task/SignalWindow)
        
        Returns:
            True if migration successful, False otherwise
        """
        current_version = self.get_sessions_schema_version()
        
        # Already at current version
        if current_version == SESSIONS_SCHEMA_VERSION:
            logger.info(f"Sessions already at schema version {SESSIONS_SCHEMA_VERSION}")
            return True
        
        # Parse versions
        try:
            current_major = int(current_version.split(".")[0])
            target_major = int(SESSIONS_SCHEMA_VERSION.split(".")[0])
        except Exception:
            logger.error(f"Invalid version format: {current_version}")
            return False
        
        if current_major > target_major:
            logger.warning(f"Cannot downgrade from {current_version} to {SESSIONS_SCHEMA_VERSION}")
            return False
        
        # Perform migration
        try:
            if current_major == 1 and target_major >= 2:
                return self._migrate_sessions_v1_to_v2()
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        
        return True
    
    def _migrate_sessions_v1_to_v2(self) -> bool:
        """Migrate sessions from v1.0 to v2.0.
        
        Changes:
        - Add schema_version and migrated_at fields
        - Remove deprecated activity fields (apps, event_count, input_events, timeline, intent_breakdown)
        - Keep intent_segments for task assignment recovery
        """
        if not os.path.exists(self.sessions_file):
            return True
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            sessions = data.get("sessions", [])
            now = datetime.now(timezone.utc).isoformat()
            migrated_count = 0
            
            for session in sessions:
                # Remove deprecated fields
                deprecated_keys = ["apps", "event_count", "input_events", "timeline", "intent_breakdown"]
                for key in deprecated_keys:
                    session.pop(key, None)
                
                # Add versioning metadata
                session["schema_version"] = SESSIONS_SCHEMA_VERSION
                session["migrated_at"] = now
                
                migrated_count += 1
            
            # Update top-level metadata
            data["schema_version"] = SESSIONS_SCHEMA_VERSION
            data["migrated_at"] = now
            
            # Write back
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Migrated {migrated_count} sessions to schema v{SESSIONS_SCHEMA_VERSION}")
            return True
            
        except Exception as e:
            logger.error(f"v1→v2 migration failed: {e}")
            return False
    
    def migrate_tasks_schema(self) -> bool:
        """Migrate tasks from old schema to current version.
        
        Returns:
            True if migration successful, False otherwise
        """
        current_version = self.get_tasks_schema_version()
        
        # Already at current version
        if current_version == TASKS_SCHEMA_VERSION:
            logger.info(f"Tasks already at schema version {TASKS_SCHEMA_VERSION}")
            return True
        
        try:
            if current_version == "1.0":
                # v1.0 is current, ensure metadata present
                if not os.path.exists(self.tasks_file):
                    return True
                
                with open(self.tasks_file, 'r') as f:
                    data = json.load(f)
                
                # Add schema version if missing
                if "schema_version" not in data:
                    data["schema_version"] = TASKS_SCHEMA_VERSION
                    data["migrated_at"] = datetime.now(timezone.utc).isoformat()
                    
                    with open(self.tasks_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    logger.info(f"Added schema metadata to tasks")
                
                return True
        except Exception as e:
            logger.error(f"Tasks migration failed: {e}")
            return False
        
        return True
    
    # ========================================================================
    # Backward Compatibility
    # ========================================================================
    
    def load_sessions_compatible(self) -> List[Dict[str, Any]]:
        """Load sessions with backward compatibility.
        
        Handles:
        - Old v1.0 sessions
        - New v2.0 sessions
        - Partial/corrupted data
        
        Returns:
            List of session dicts (migrated if needed)
        """
        if not os.path.exists(self.sessions_file):
            return []
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            sessions = data.get("sessions", [])
            version = data.get("schema_version", "1.0")
            
            # If old version, try to migrate
            if version == "1.0":
                self.migrate_sessions_schema()
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            return []
    
    def load_tasks_compatible(self) -> List[Dict[str, Any]]:
        """Load tasks with backward compatibility.
        
        Returns:
            List of task dicts (migrated if needed)
        """
        if not os.path.exists(self.tasks_file):
            return []
        
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
            
            tasks = data.get("tasks", [])
            version = data.get("schema_version", "1.0")
            
            # If old version, try to migrate
            if version != TASKS_SCHEMA_VERSION:
                self.migrate_tasks_schema()
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            return []
    
    # ========================================================================
    # Data Cleanup
    # ========================================================================
    
    def cleanup_stale_sessions(
        self,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        dry_run: bool = False,
        cutoff_date: datetime = None  # For testing with fixed dates
    ) -> Dict[str, Any]:
        """Remove sessions older than retention period.
        
        Args:
            retention_days: Keep sessions newer than this many days
            dry_run: If True, only report what would be deleted
            cutoff_date: Override cutoff date (for testing)
        
        Returns:
            Dict with cleanup stats:
            {
                "deleted_count": int,
                "archived_count": int,
                "kept_count": int,
                "oldest_kept": datetime_str,
                "newest_deleted": datetime_str
            }
        """
        if not os.path.exists(self.sessions_file):
            return {"deleted_count": 0, "archived_count": 0, "kept_count": 0}
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            sessions = data.get("sessions", [])
            if cutoff_date is None:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            kept = []
            deleted = []
            archived = []
            
            for session in sessions:
                try:
                    end_str = session.get("end")
                    if not end_str:
                        # No end time, keep it (likely in-progress)
                        kept.append(session)
                        continue
                    
                    end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                    
                    if end_date < cutoff_date:
                        # Old session
                        if session.get("in_progress") is False:
                            # Completed session - can delete
                            deleted.append(session)
                        else:
                            # Keep in-progress sessions
                            kept.append(session)
                    else:
                        # Recent session
                        kept.append(session)
                except Exception as e:
                    logger.warning(f"Error processing session: {e}")
                    kept.append(session)
            
            # If not dry-run, write changes
            if not dry_run and deleted:
                data["sessions"] = kept
                data["last_cleanup"] = datetime.now(timezone.utc).isoformat()
                
                with open(self.sessions_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Archive deleted sessions
                if deleted:
                    self._archive_sessions(deleted)
            
            # Prepare response
            oldest_kept_str = None
            newest_deleted_str = None
            
            if kept:
                try:
                    oldest_kept = min([s for s in kept if s.get("end")], 
                                    key=lambda s: s.get("end", ""))
                    oldest_kept_str = oldest_kept.get("end")
                except Exception:
                    pass
            
            if deleted:
                try:
                    newest_deleted = max([s for s in deleted if s.get("end")], 
                                        key=lambda s: s.get("end", ""))
                    newest_deleted_str = newest_deleted.get("end")
                except Exception:
                    pass
            
            return {
                "deleted_count": len(deleted),
                "archived_count": len(deleted),  # We archive all deleted
                "kept_count": len(kept),
                "oldest_kept": oldest_kept_str,
                "newest_deleted": newest_deleted_str,
                "dry_run": dry_run
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"deleted_count": 0, "archived_count": 0, "kept_count": 0, "error": str(e)}
    
    def _archive_sessions(self, sessions: List[Dict[str, Any]]) -> bool:
        """Archive sessions to separate file.
        
        Args:
            sessions: Sessions to archive
        
        Returns:
            True if successful
        """
        try:
            # Create timestamped archive file
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_file = os.path.join(
                self.session_archives_dir,
                f"sessions_archive_{timestamp}.json"
            )
            
            # Read existing archive or create new
            archive_data = {
                "schema_version": SESSIONS_SCHEMA_VERSION,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "archived_count": len(sessions),
                "sessions": sessions
            }
            
            with open(archive_file, 'w') as f:
                json.dump(archive_data, f, indent=2)
            
            logger.info(f"Archived {len(sessions)} sessions to {archive_file}")
            return True
            
        except Exception as e:
            logger.error(f"Archive failed: {e}")
            return False
    
    # ========================================================================
    # Session Aggregation into Baselines
    # ========================================================================
    
    def aggregate_sessions_to_baseline(self) -> Dict[str, Any]:
        """Aggregate old sessions into behavioral baseline statistics.
        
        Extracts:
        - Average session duration per task
        - Session frequency per task
        - Time-of-day activity patterns
        - Device breakdown
        
        Returns:
            Dict with aggregated baseline statistics
        """
        sessions = self.load_sessions_compatible()
        
        if not sessions:
            return {
                "baseline_version": "1.0",
                "aggregated_at": datetime.now(timezone.utc).isoformat(),
                "session_count": 0,
                "stats": {}
            }
        
        stats = {
            "total_sessions": 0,
            "duration_by_hour": {},  # hour -> [durations]
            "device_breakdown": {},   # device_id -> count
            "session_count_by_date": {},  # date -> count
            "avg_session_duration_minutes": 0.0,
        }
        
        total_duration = 0.0
        
        for session in sessions:
            try:
                # Parse dates
                start_str = session.get("start")
                end_str = session.get("end")
                
                if not start_str or not end_str:
                    continue
                
                start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                
                duration = (end - start).total_seconds() / 60.0
                
                # Track duration
                total_duration += duration
                stats["total_sessions"] += 1
                
                # Track by hour
                hour = start.hour
                if hour not in stats["duration_by_hour"]:
                    stats["duration_by_hour"][hour] = []
                stats["duration_by_hour"][hour].append(duration)
                
                # Track by device
                device = session.get("device_id", "unknown")
                stats["device_breakdown"][device] = stats["device_breakdown"].get(device, 0) + 1
                
                # Track by date
                date_key = start.date().isoformat()
                stats["session_count_by_date"][date_key] = stats["session_count_by_date"].get(date_key, 0) + 1
                
            except Exception as e:
                logger.warning(f"Error aggregating session: {e}")
                continue
        
        # Compute averages
        if stats["total_sessions"] > 0:
            stats["avg_session_duration_minutes"] = total_duration / stats["total_sessions"]
        
        # Compute hourly averages
        for hour in stats["duration_by_hour"]:
            durations = stats["duration_by_hour"][hour]
            stats["duration_by_hour"][hour] = {
                "avg": sum(durations) / len(durations),
                "count": len(durations)
            }
        
        return {
            "baseline_version": "1.0",
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "source_session_count": len(sessions),
            "stats": stats
        }
    
    # ========================================================================
    # Data Validation & Repair
    # ========================================================================
    
    def validate_sessions(self) -> Dict[str, Any]:
        """Validate session data integrity.
        
        Checks:
        - Required fields present
        - Valid datetime formats
        - Logical constraints (start < end)
        - Schema compliance
        
        Returns:
            Dict with validation results:
            {
                "valid_count": int,
                "invalid_count": int,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        sessions = self.load_sessions_compatible()
        
        valid_count = 0
        invalid_count = 0
        errors = []
        warnings = []
        
        for i, session in enumerate(sessions):
            try:
                # Check required fields
                required = ["id", "start", "end", "device_id"]
                missing = [f for f in required if not session.get(f)]
                
                if missing:
                    errors.append(f"Session {i}: Missing fields {missing}")
                    invalid_count += 1
                    continue
                
                # Validate datetimes
                try:
                    start = datetime.fromisoformat(session["start"].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(session["end"].replace('Z', '+00:00'))
                except Exception as e:
                    errors.append(f"Session {i}: Invalid datetime format - {e}")
                    invalid_count += 1
                    continue
                
                # Check logical constraints
                if start > end:
                    errors.append(f"Session {i}: start > end")
                    invalid_count += 1
                    continue
                
                # Check for reasonable duration (not > 24 hours)
                duration_hours = (end - start).total_seconds() / 3600
                if duration_hours > 24:
                    warnings.append(f"Session {i}: Unusually long duration ({duration_hours:.1f}h)")
                
                valid_count += 1
                
            except Exception as e:
                errors.append(f"Session {i}: Unexpected error - {e}")
                invalid_count += 1
        
        return {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "total_count": len(sessions),
            "errors": errors,
            "warnings": warnings
        }
    
    def repair_sessions(self) -> Dict[str, Any]:
        """Attempt to repair invalid sessions.
        
        Repairs:
        - Missing device_id (set to "unknown")
        - Invalid datetime formats
        - Sessions with start > end (swap them)
        
        Returns:
            Dict with repair stats
        """
        if not os.path.exists(self.sessions_file):
            return {"repaired_count": 0, "removed_count": 0}
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            sessions = data.get("sessions", [])
            repaired_count = 0
            removed_count = 0
            repaired_sessions = []
            
            for session in sessions:
                try:
                    # Fix missing device_id
                    if not session.get("device_id"):
                        session["device_id"] = "unknown"
                        repaired_count += 1
                    
                    # Parse and validate dates
                    start_str = session.get("start")
                    end_str = session.get("end")
                    
                    if not start_str or not end_str:
                        # Cannot repair - remove
                        logger.warning(f"Removing session with missing dates")
                        removed_count += 1
                        continue
                    
                    start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    
                    # Fix if reversed
                    if start > end:
                        start, end = end, start
                        session["start"] = start.isoformat()
                        session["end"] = end.isoformat()
                        repaired_count += 1
                        logger.warning(f"Swapped reversed start/end times")
                    
                    repaired_sessions.append(session)
                    
                except Exception as e:
                    logger.warning(f"Cannot repair session: {e}")
                    removed_count += 1
            
            # Write repaired data
            if repaired_count > 0 or removed_count > 0:
                data["sessions"] = repaired_sessions
                data["last_repair"] = datetime.now(timezone.utc).isoformat()
                
                with open(self.sessions_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Repaired {repaired_count}, removed {removed_count} sessions")
            
            return {
                "repaired_count": repaired_count,
                "removed_count": removed_count,
                "remaining_count": len(repaired_sessions)
            }
            
        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return {"repaired_count": 0, "removed_count": 0, "error": str(e)}


# ============================================================================
# Convenience Functions
# ============================================================================

def get_maintenance() -> DataMaintenance:
    """Get singleton DataMaintenance instance."""
    return DataMaintenance()


def run_full_maintenance(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run all maintenance tasks.
    
    Args:
        retention_days: Days to retain sessions
        dry_run: If True, only report without making changes
    
    Returns:
        Dict with results of all maintenance operations
    """
    maintenance = get_maintenance()
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "operations": {}
    }
    
    # Migrate schemas
    results["operations"]["migrate_sessions"] = maintenance.migrate_sessions_schema()
    results["operations"]["migrate_tasks"] = maintenance.migrate_tasks_schema()
    
    # Validate data
    results["operations"]["validate"] = maintenance.validate_sessions()
    
    # Repair if needed
    if results["operations"]["validate"]["invalid_count"] > 0:
        results["operations"]["repair"] = maintenance.repair_sessions()
    
    # Cleanup stale sessions
    results["operations"]["cleanup"] = maintenance.cleanup_stale_sessions(
        retention_days=retention_days,
        dry_run=dry_run
    )
    
    # Aggregate to baseline
    results["operations"]["aggregate_baseline"] = maintenance.aggregate_sessions_to_baseline()
    
    return results

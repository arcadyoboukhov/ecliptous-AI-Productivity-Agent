"""
Core Task Recognition - v2 Task Labels

Defines 23 core task types with behavioral signatures,
feature patterns, and multi-layer classification logic (app, window, behavioral).
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math


@dataclass
class TaskSignature:
    """Behavioral signature for a task type."""
    task_id: str
    display_name: str
    description: str
    
    # Feature expectations (0.0 to 1.0 normalized)
    expected_intensity: float  # Input events per minute
    expected_continuity: float  # Focus without interruption
    expected_entropy: float  # Context switching frequency
    expected_app_diversity: float  # Number of different apps
    expected_duration: float  # Typical session length (hours)
    
    # Classification thresholds
    min_confidence: float = 0.6
    sticky: bool = False  # Resist reclassification
    
    # Behavioral hints
    dominant_apps: List[str] = None  # Key indicator apps
    interaction_pattern: str = ""  # Description of interaction style


# Core v1 Task Definitions
CORE_TASKS = {
    "deep_development": TaskSignature(
        task_id="deep_development",
        display_name="Deep Development",
        description="Sustained, high-focus execution work with minimal interruptions",
        expected_intensity=0.8,  # High keystroke/mouse activity
        expected_continuity=0.85,  # Very focused
        expected_entropy=0.2,  # Low switching
        expected_app_diversity=0.2,  # Few tools
        expected_duration=2.0,  # Long sessions
        min_confidence=0.65,
        sticky=True,  # Don't reclassify easily
        dominant_apps=["code", "visual studio", "pycharm", "terminal", "powershell"],
        interaction_pattern="High intensity, low switching, sustained focus"
    ),
    
    "technical_research": TaskSignature(
        task_id="technical_research",
        display_name="Technical Research",
        description="Information gathering, reading, learning, exploratory investigation",
        expected_intensity=0.4,  # Moderate input
        expected_continuity=0.6,  # Moderate focus
        expected_entropy=0.5,  # Medium switching
        expected_app_diversity=0.5,  # Multiple sources
        expected_duration=1.0,  # Medium sessions
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["browser", "chrome", "firefox", "edge", "documentation"],
        interaction_pattern="Reading-heavy, moderate switching, exploratory"
    ),
    
    "context_switching": TaskSignature(
        task_id="context_switching",
        display_name="Context Switching",
        description="Fragmented, interrupt-driven work with frequent task changes",
        expected_intensity=0.5,  # Variable input
        expected_continuity=0.2,  # Low focus
        expected_entropy=0.85,  # High switching
        expected_app_diversity=0.8,  # Many different apps
        expected_duration=0.3,  # Short bursts
        min_confidence=0.5,
        sticky=False,
        dominant_apps=[],  # No dominant app
        interaction_pattern="Rapid switching, short dwell times, fragmented"
    ),
    
    "team_meeting": TaskSignature(
        task_id="team_meeting",
        display_name="Team Meeting",
        description="Synchronous collaboration via calls or structured communication",
        expected_intensity=0.2,  # Low activity
        expected_continuity=0.7,  # Sustained but passive
        expected_entropy=0.1,  # Minimal switching
        expected_app_diversity=0.1,  # Single meeting app
        expected_duration=0.75,  # Typical meeting length
        min_confidence=0.75,  # High confidence when detected
        sticky=True,
        dominant_apps=["zoom", "teams", "meet", "slack", "discord"],
        interaction_pattern="Low input, fixed duration, passive engagement"
    ),
    
    "administrative_work": TaskSignature(
        task_id="administrative_work",
        display_name="Administrative Work",
        description="Operational, repetitive, or maintenance-oriented tasks",
        expected_intensity=0.45,  # Moderate bursts
        expected_continuity=0.4,  # Interrupted focus
        expected_entropy=0.55,  # Medium switching
        expected_app_diversity=0.5,  # Few core apps
        expected_duration=0.5,  # Short to medium
        min_confidence=0.5,
        sticky=False,
        dominant_apps=["outlook", "mail", "calendar", "excel", "word"],
        interaction_pattern="Bursty activity, operational tasks, moderate switching"
    ),
    
    "strategic_planning": TaskSignature(
        task_id="strategic_planning",
        display_name="Strategic Planning",
        description="High-level thinking, planning, and decision-making work",
        expected_intensity=0.3,  # Lower intensity
        expected_continuity=0.75,  # Sustained thinking
        expected_entropy=0.3,  # Low switching
        expected_app_diversity=0.3,  # Mixed but focused tools
        expected_duration=1.5,  # Longer sessions
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["notion", "onenote", "miro", "figma", "whiteboard"],
        interaction_pattern="Thoughtful pauses, mixed tools, sustained sessions"
    ),
    
    # Additional task categories
    "content_creation": TaskSignature(
        task_id="content_creation",
        display_name="Content Creation",
        description="Writing, designing, or producing creative content",
        expected_intensity=0.6,
        expected_continuity=0.7,
        expected_entropy=0.3,
        expected_app_diversity=0.3,
        expected_duration=1.5,
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["word", "photoshop", "figma", "canva", "premiere"],
        interaction_pattern="Sustained creative work with moderate intensity"
    ),
    
    "code_review": TaskSignature(
        task_id="code_review",
        display_name="Code Review",
        description="Reviewing code, pull requests, or technical documentation",
        expected_intensity=0.4,
        expected_continuity=0.65,
        expected_entropy=0.4,
        expected_app_diversity=0.3,
        expected_duration=0.5,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["browser", "github", "gitlab", "code"],
        interaction_pattern="Reading-focused with occasional comments"
    ),
    
    "debugging": TaskSignature(
        task_id="debugging",
        display_name="Debugging",
        description="Troubleshooting, investigating errors, and fixing bugs",
        expected_intensity=0.7,
        expected_continuity=0.6,
        expected_entropy=0.6,
        expected_app_diversity=0.4,
        expected_duration=1.0,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["code", "terminal", "browser", "debugger"],
        interaction_pattern="High intensity with frequent context switches"
    ),
    
    "learning": TaskSignature(
        task_id="learning",
        display_name="Learning",
        description="Educational content, tutorials, courses, or skill development",
        expected_intensity=0.3,
        expected_continuity=0.7,
        expected_entropy=0.4,
        expected_app_diversity=0.4,
        expected_duration=1.0,
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["browser", "youtube", "udemy", "coursera", "notes"],
        interaction_pattern="Passive consumption with note-taking"
    ),
    
    "email_communication": TaskSignature(
        task_id="email_communication",
        display_name="Email Communication",
        description="Reading and responding to emails",
        expected_intensity=0.5,
        expected_continuity=0.5,
        expected_entropy=0.4,
        expected_app_diversity=0.2,
        expected_duration=0.4,
        min_confidence=0.65,
        sticky=False,
        dominant_apps=["outlook", "gmail", "mail", "thunderbird"],
        interaction_pattern="Moderate bursts of reading and typing"
    ),
    
    "chat_messaging": TaskSignature(
        task_id="chat_messaging",
        display_name="Chat & Messaging",
        description="Quick messages, async communication, group chats",
        expected_intensity=0.6,
        expected_continuity=0.4,
        expected_entropy=0.5,
        expected_app_diversity=0.3,
        expected_duration=0.3,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["slack", "teams", "discord", "telegram", "whatsapp"],
        interaction_pattern="Short bursts with quick responses"
    ),
    
    "documentation": TaskSignature(
        task_id="documentation",
        display_name="Documentation",
        description="Writing docs, READMEs, technical specifications",
        expected_intensity=0.5,
        expected_continuity=0.7,
        expected_entropy=0.3,
        expected_app_diversity=0.2,
        expected_duration=0.8,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["word", "notion", "confluence", "docs", "markdown"],
        interaction_pattern="Sustained writing with occasional references"
    ),
    
    "data_analysis": TaskSignature(
        task_id="data_analysis",
        display_name="Data Analysis",
        description="Working with data, spreadsheets, analytics, or reports",
        expected_intensity=0.6,
        expected_continuity=0.65,
        expected_entropy=0.35,
        expected_app_diversity=0.3,
        expected_duration=1.0,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["excel", "powerbi", "tableau", "jupyter", "sql"],
        interaction_pattern="Moderate intensity with analytical focus"
    ),
    
    "system_maintenance": TaskSignature(
        task_id="system_maintenance",
        display_name="System Maintenance",
        description="System updates, configuration, DevOps, infrastructure work",
        expected_intensity=0.5,
        expected_continuity=0.5,
        expected_entropy=0.5,
        expected_app_diversity=0.5,
        expected_duration=0.6,
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["terminal", "powershell", "docker", "kubernetes", "aws"],
        interaction_pattern="Command-line focused with monitoring"
    ),
    
    "browsing_research": TaskSignature(
        task_id="browsing_research",
        display_name="Web Browsing & Research",
        description="General web browsing, exploration, or information gathering",
        expected_intensity=0.4,
        expected_continuity=0.5,
        expected_entropy=0.6,
        expected_app_diversity=0.4,
        expected_duration=0.5,
        min_confidence=0.5,
        sticky=False,
        dominant_apps=["browser", "chrome", "firefox", "edge"],
        interaction_pattern="Moderate switching between tabs and sites"
    ),
    
    "video_conferencing": TaskSignature(
        task_id="video_conferencing",
        display_name="Video Conferencing",
        description="Video calls, webinars, or virtual presentations",
        expected_intensity=0.2,
        expected_continuity=0.8,
        expected_entropy=0.1,
        expected_app_diversity=0.1,
        expected_duration=0.75,
        min_confidence=0.75,
        sticky=True,
        dominant_apps=["zoom", "teams", "meet", "webex", "skype"],
        interaction_pattern="Low activity during call duration"
    ),
    
    "content_consumption": TaskSignature(
        task_id="content_consumption",
        display_name="Content Consumption",
        description="Watching videos, reading articles, consuming media",
        expected_intensity=0.2,
        expected_continuity=0.7,
        expected_entropy=0.3,
        expected_app_diversity=0.3,
        expected_duration=0.6,
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["youtube", "netflix", "spotify", "browser"],
        interaction_pattern="Passive consumption with minimal interaction"
    ),
    
    "task_management": TaskSignature(
        task_id="task_management",
        display_name="Task Management",
        description="Planning, organizing tasks, updating project boards",
        expected_intensity=0.5,
        expected_continuity=0.6,
        expected_entropy=0.4,
        expected_app_diversity=0.3,
        expected_duration=0.4,
        min_confidence=0.6,
        sticky=False,
        dominant_apps=["jira", "trello", "asana", "notion", "todoist"],
        interaction_pattern="Moderate activity organizing and updating"
    ),
    
    "file_management": TaskSignature(
        task_id="file_management",
        display_name="File Management",
        description="Organizing files, cleaning up, file operations",
        expected_intensity=0.6,
        expected_continuity=0.4,
        expected_entropy=0.5,
        expected_app_diversity=0.3,
        expected_duration=0.3,
        min_confidence=0.55,
        sticky=False,
        dominant_apps=["explorer", "finder", "dropbox", "drive"],
        interaction_pattern="Repetitive operations with frequent selections"
    ),
    
    "general_productivity": TaskSignature(
        task_id="general_productivity",
        display_name="General Productivity",
        description="Mixed productive work that doesn't fit specific categories",
        expected_intensity=0.5,
        expected_continuity=0.5,
        expected_entropy=0.5,
        expected_app_diversity=0.5,
        expected_duration=0.5,
        min_confidence=0.4,
        sticky=False,
        dominant_apps=[],
        interaction_pattern="Balanced mix of activities"
    ),
    
    "idle_break": TaskSignature(
        task_id="idle_break",
        display_name="Idle / Break",
        description="Minimal activity, breaks, or idle time",
        expected_intensity=0.1,
        expected_continuity=0.3,
        expected_entropy=0.2,
        expected_app_diversity=0.2,
        expected_duration=0.2,
        min_confidence=0.5,
        sticky=False,
        dominant_apps=[],
        interaction_pattern="Very low activity or idle"
    ),
}


def normalize_feature(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a feature value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def extract_task_features(rolling_features: Dict) -> Dict[str, float]:
    """
    Extract normalized task recognition features from rolling features.
    
    Maps raw metrics to normalized 0-1 scale suitable for distance calculation.
    """
    # Normalize intensity (input events per minute)
    intensity_raw = rolling_features.get('intensity', 0)
    intensity = normalize_feature(intensity_raw, 0, 100)  # 0-100 events/min
    
    # Continuity (already 0-1)
    continuity = rolling_features.get('continuity', 0.5)
    
    # Entropy from app diversity
    app_diversity_raw = rolling_features.get('app_diversity', 0)
    app_diversity = normalize_feature(app_diversity_raw, 0, 1.0)
    
    # Context switching entropy
    num_windows = rolling_features.get('num_windows', 1)
    entropy = normalize_feature(num_windows, 0, 20)  # 0-20 windows
    
    # Duration (hours)
    duration_raw = rolling_features.get('duration', 0)
    duration = normalize_feature(duration_raw, 0, 4.0)  # 0-4 hours
    
    return {
        'intensity': intensity,
        'continuity': continuity,
        'entropy': entropy,
        'app_diversity': app_diversity,
        'duration': duration,
    }


def compute_task_distance(features: Dict[str, float], signature: TaskSignature) -> float:
    """
    Compute Euclidean distance between observed features and task signature.
    
    Lower distance = better match.
    """
    feature_weights = {
        'intensity': 1.5,  # High weight - key discriminator
        'continuity': 1.2,
        'entropy': 1.5,  # High weight - key discriminator
        'app_diversity': 0.8,
        'duration': 0.6,  # Lower weight - less reliable
    }
    
    squared_diffs = []
    for key, weight in feature_weights.items():
        observed = features.get(key, 0.5)
        expected = getattr(signature, f'expected_{key}')
        diff = (observed - expected) ** 2
        squared_diffs.append(weight * diff)
    
    distance = math.sqrt(sum(squared_diffs))
    
    # Normalize to 0-1 range (max possible distance ~3.5)
    normalized_distance = min(1.0, distance / 3.5)
    
    return normalized_distance


def compute_task_confidence(distance: float, signature: TaskSignature) -> float:
    """
    Convert distance to confidence score (0-1).
    
    Closer match = higher confidence.
    """
    # Base confidence from distance
    base_confidence = 1.0 - distance
    
    # Apply minimum threshold
    if base_confidence < signature.min_confidence:
        return base_confidence * 0.7  # Reduced confidence below threshold
    
    return base_confidence


def _build_contextual_task_id(base_task_id: str, features: Dict[str, float]) -> str:
    """Build a contextual task id using smart naming engine."""
    from agent.task.smart_naming import generate_smart_task_name
    
    active_app = (features.get("active_app") or "").strip()
    active_window = (features.get("active_window_title") or "").strip()
    
    # Get behavioral confidence from features
    behavioral_confidence = features.get("continuity", 0.5)
    
    # Generate smart task name
    smart_name = generate_smart_task_name(
        base_category=base_task_id,
        app=active_app,
        window_title=active_window,
        behavioral_confidence=behavioral_confidence
    )
    
    return smart_name


def get_task_recommendation(features: Dict[str, float]) -> tuple[str, float, str]:
    """
    Recommend best task label based on feature vector.
    
    Uses multi-layer classification:
    1. App-based direct classification (high confidence)
    2. Window title keywords (medium confidence)
    3. Behavioral pattern matching (fallback)
    
    Returns: (task_id, confidence, reason)
    """
    # Layer 1: App-based classification (most reliable)
    app = (features.get("app_name") or features.get("active_app") or "").lower()
    window_title = (features.get("window_title") or features.get("active_window_title") or "").lower()

    # Ignore noisy system UI windows
    if "system tray overflow" in window_title or "systemtrayoverflow" in window_title:
        return ("general_productivity", 0.2, "ignored_system_ui")
    
    app_task, app_confidence = classify_by_app(app, window_title)
    if app_task and app_confidence > 0.7:
        return (app_task, app_confidence, "app_match")
    
    # Layer 2: Window title keywords
    title_task, title_confidence = classify_by_window_title(window_title)
    if title_task and title_confidence > 0.6:
        # Blend with app classification if available
        if app_task and app_confidence > 0.5:
            confidence = (title_confidence + app_confidence) / 2
            return (app_task if app_confidence > title_confidence else title_task, 
                   confidence, "title_and_app_match")
        return (title_task, title_confidence, "title_match")
    
    # Layer 3: Behavioral pattern matching
    task_features = extract_task_features(features)
    
    best_task = None
    best_distance = float('inf')
    
    for task_id, signature in CORE_TASKS.items():
        distance = compute_task_distance(task_features, signature)
        
        if distance < best_distance:
            best_distance = distance
            best_task = signature
    
    if not best_task:
        # Ultimate fallback: general_productivity
        return ("general_productivity", 0.4, "default_fallback")
    
    confidence = compute_task_confidence(best_distance, best_task)

    # Guard: avoid mislabeling passive browsing as video conferencing
    if best_task.task_id == "video_conferencing":
        app_l = app
        title_l = window_title
        has_conf_app = any(k in app_l for k in ["zoom", "teams", "meet", "webex", "skype"])
        has_conf_title = any(k in title_l for k in ["zoom", "meeting", "call", "webinar", "teams", "meet"])
        mic_ratio = features.get("mic_active_ratio", 0.0) or features.get("mic_ratio", 0.0)
        cam_ratio = features.get("camera_active_ratio", 0.0) or features.get("camera_ratio", 0.0)

        if not (has_conf_app or has_conf_title or mic_ratio >= 0.3 or cam_ratio >= 0.2):
            if any(b in app_l for b in ["browser", "firefox", "chrome", "edge", "safari", "brave"]):
                return ("browsing_research", max(0.55, confidence), "video_conf_blocked_browser_fallback")
            return ("content_consumption", max(0.5, confidence), "video_conf_blocked_fallback")
    
    # Boost confidence if app/title partially matched
    if app_task and app_task == best_task.task_id:
        confidence = min(0.95, confidence + app_confidence * 0.2)
        return (best_task.task_id, confidence, "behavioral_with_app_boost")
    
    # If confidence too low, fall back to general_productivity or browsing_research
    if confidence < 0.4:
        if "browser" in app or "firefox" in app or "chrome" in app:
            return ("browsing_research", 0.5, "browser_fallback")
        return ("general_productivity", 0.45, "low_confidence_fallback")
    
    return (best_task.task_id, confidence, "behavioral_match")


def classify_by_app(app: str, window_title: str = "") -> tuple[Optional[str], float]:
    """
    Classify task based on application name.
    Returns (task_id, confidence) or (None, 0.0) if no match.
    """
    if not app:
        return (None, 0.0)
    
    # High confidence app mappings
    high_confidence_apps = {
        # Development
        "code.exe": "deep_development",
        "vscode": "deep_development",
        "visual studio": "deep_development",
        "pycharm": "deep_development",
        "intellij": "deep_development",
        "sublime": "deep_development",
        "atom": "deep_development",
        "vim": "deep_development",
        
        # Email
        "outlook": "email_communication",
        "thunderbird": "email_communication",
        "mail": "email_communication",
        
        # Chat/Messaging
        "slack": "chat_messaging",
        "discord": "chat_messaging",
        "telegram": "chat_messaging",
        "whatsapp": "chat_messaging",
        "signal": "chat_messaging",
        
        # Video conferencing
        "zoom": "video_conferencing",
        "meet": "video_conferencing",
        "webex": "video_conferencing",
        "skype": "video_conferencing",
        
        # Data analysis
        "excel": "data_analysis",
        "powerbi": "data_analysis",
        "tableau": "data_analysis",
        
        # Documentation/Writing
        "word": "documentation",
        "notion": "documentation",
        "onenote": "documentation",
        "confluence": "documentation",
        
        # Task management
        "jira": "task_management",
        "trello": "task_management",
        "asana": "task_management",
        "todoist": "task_management",
        
        # Terminal/System
        "terminal": "system_maintenance",
        "powershell": "system_maintenance",
        "cmd": "system_maintenance",
        "bash": "system_maintenance",
        
        # File management
        "explorer": "file_management",
        "finder": "file_management",
    }
    
    # Check direct matches
    for app_key, task_id in high_confidence_apps.items():
        if app_key in app:
            return (task_id, 0.85)
    
    # Browser with context
    if any(browser in app for browser in ["firefox", "chrome", "edge", "safari", "brave"]):
        # Disambiguate by window title
        if any(keyword in window_title for keyword in ["github", "gitlab", "stackoverflow", "docs"]):
            return ("technical_research", 0.75)
        elif any(keyword in window_title for keyword in ["youtube", "netflix", "spotify"]):
            return ("content_consumption", 0.75)
        elif any(keyword in window_title for keyword in ["gmail", "mail", "inbox"]):
            return ("email_communication", 0.75)
        elif any(keyword in window_title for keyword in ["teams", "slack", "discord"]):
            return ("chat_messaging", 0.75)
        else:
            return ("browsing_research", 0.65)
    
    # Communication apps context
    if "teams" in app:
        if any(keyword in window_title for keyword in ["call", "meeting", "video"]):
            return ("video_conferencing", 0.85)
        else:
            return ("chat_messaging", 0.75)
    
    return (None, 0.0)


def classify_by_window_title(window_title: str) -> tuple[Optional[str], float]:
    """
    Classify task based on window title keywords.
    Returns (task_id, confidence) or (None, 0.0) if no match.
    """
    if not window_title:
        return (None, 0.0)
    
    # Keyword mappings with confidence
    keyword_mappings = [
        # Development indicators
        (["github", "gitlab", "pull request", "merge request", "code review"], "code_review", 0.75),
        (["debug", "debugger", "breakpoint", "stack trace", "error"], "debugging", 0.7),
        (["terminal", "command", "bash", "powershell", "cmd"], "system_maintenance", 0.7),
        
        # Learning indicators
        (["tutorial", "course", "learn", "udemy", "coursera", "lesson"], "learning", 0.75),
        (["documentation", "docs", "api reference", "manual"], "technical_research", 0.7),
        
        # Communication indicators
        (["email", "inbox", "gmail", "outlook"], "email_communication", 0.75),
        (["chat", "message", "slack", "discord", "teams"], "chat_messaging", 0.75),
        (["zoom", "meeting", "call", "webinar"], "video_conferencing", 0.8),
        
        # Content indicators
        (["youtube", "video", "watch"], "content_consumption", 0.75),
        (["netflix", "hulu", "prime video", "spotify"], "content_consumption", 0.8),
        
        # Research indicators
        (["google", "search", "stackoverflow", "reddit"], "browsing_research", 0.65),
        (["article", "blog", "news", "medium"], "browsing_research", 0.65),
        
        # Productivity indicators
        (["todo", "task", "jira", "trello", "project"], "task_management", 0.7),
        (["calendar", "schedule", "meeting", "appointment"], "task_management", 0.7),
        (["spreadsheet", "excel", "data", "analytics"], "data_analysis", 0.7),
        (["word", "document", "write", "draft"], "documentation", 0.65),
        
        # Creative indicators
        (["design", "figma", "photoshop", "illustrator"], "content_creation", 0.75),
        (["edit", "editing", "premiere", "davinci"], "content_creation", 0.75),
    ]
    
    # Check each keyword set
    for keywords, task_id, confidence in keyword_mappings:
        for keyword in keywords:
            if keyword in window_title:
                return (task_id, confidence)
    
    return (None, 0.0)
    
    # Determine reason
    if confidence >= best_task.min_confidence + 0.2:
        reason = f"strong_match_{best_task.task_id}"
    elif confidence >= best_task.min_confidence:
        reason = f"confident_{best_task.task_id}"
    else:
        reason = f"tentative_{best_task.task_id}"
    
    # Return base category (not contextualized yet)
    return (best_task.task_id, confidence, reason)


def should_transition(current_task_id: str, new_task_id: str, 
                     current_confidence: float, new_confidence: float,
                     current_duration_minutes: float) -> bool:
    """
    Determine if task should transition based on confidence and stickiness.
    
    Implements hysteresis to prevent label thrashing while allowing
    legitimate task changes.
    """
    if current_task_id == new_task_id:
        return False  # No transition needed
    
    # Get task signatures
    current_sig = CORE_TASKS.get(current_task_id)
    new_sig = CORE_TASKS.get(new_task_id)
    
    # Check if current task is sticky (Deep Development, Team Meeting)
    if current_sig and current_sig.sticky:
        # Sticky tasks resist transitions more strongly
        confidence_gap = new_confidence - current_confidence
        
        # Require higher confidence OR sufficient duration
        # Don't let sticky tasks lock in forever just because they started with high confidence
        has_confidence_improvement = confidence_gap >= 0.10  # 10% improvement
        has_sufficient_duration = current_duration_minutes >= 10  # 10 minutes
        confidence_is_reasonable = confidence_gap >= -0.05  # New task not much worse (within 5%)
        
        # Allow transition if:
        # 1. Clear confidence improvement (10%+), OR
        # 2. Sufficient time has passed AND new confidence is reasonable
        if has_confidence_improvement:
            return True
        elif has_sufficient_duration and confidence_is_reasonable:
            return True
        else:
            return False  # Too early and/or new task has much lower confidence
    
    # For non-sticky tasks, allow transition if:
    # 1. New confidence is meaningfully higher (>3% gap), OR
    # 2. Task types are different AND confidence is comparable (within 15%)
    # 3. Sufficient time has passed (at least 3 minutes)
    confidence_gap = new_confidence - current_confidence
    
    if confidence_gap > 0.03:
        # Clear confidence improvement - allow transition
        return True
    
    # Check if tasks are significantly different even with similar confidence
    if abs(confidence_gap) <= 0.15:  # Within 15% confidence
        # Allow transition if enough time has passed (at least 3 minutes)
        # This enables detecting genuine task changes
        if current_duration_minutes >= 3:
            return True
    
    # Even with lower confidence, allow transition if enough time passed
    # and the new confidence is still reasonable (>0.5)
    if current_duration_minutes >= 5 and new_confidence > 0.5:
        return True
    
    return False


def format_task_label(task_id: str) -> str:
    """Get display name for task ID."""
    signature = CORE_TASKS.get(task_id)
    return signature.display_name if signature else task_id.replace('_', ' ').title()

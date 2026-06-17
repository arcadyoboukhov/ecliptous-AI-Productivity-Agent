"""
Smart Task Naming Engine

Combines behavioral patterns, app context, and NLP-based window title analysis
to produce accurate, human-readable task names.

Architecture:
1. Clean and normalize raw activity data (remove browser cruft, punctuation)
2. Extract keywords from window titles
3. Match keywords to domain patterns (coding, research, communication, etc.)
4. Validate with behavioral signature
5. Generate contextual task name
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WindowContext:
    """Parsed context from window title and app."""
    app: str
    window_title: str
    normalized_title: str
    domain_keywords: List[str]
    entities: List[str]
    category_hint: Optional[str]


# Browser and app suffixes to remove
BROWSER_SUFFIXES = [
    "Mozilla Firefox",
    "Google Chrome",
    "Microsoft Edge",
    "Safari",
    "Opera",
    "Brave Browser",
    "Firefox",
    "Chrome",
    "Edge",
]


def normalize_window_title(window_title: str) -> str:
    """
    Clean and normalize window title by removing browser cruft and noise.
    
    Examples:
        "ChatGPT – Mozilla Firefox" → "ChatGPT"
        "quantova - Google Search – Mozilla Firefox" → "quantova | Google Search"
        "main.py - Visual Studio Code" → "main.py | Visual Studio Code"
    """
    if not window_title:
        return ""

    # Ignore noisy system UI windows
    if "system tray overflow" in window_title.lower():
        return ""
    
    # Remove unicode dashes and normalize
    normalized = window_title.replace("–", "-").replace("—", "-").replace("•", "-")
    
    # Remove browser suffixes (case insensitive)
    for suffix in BROWSER_SUFFIXES:
        # Try with various separators
        patterns = [
            f" - {suffix}$",
            f" – {suffix}$",
            f" — {suffix}$",
            f" | {suffix}$",
        ]
        for pattern in patterns:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    
    # Clean up multiple separators
    normalized = re.sub(r'\s*[-–—|]\s*[-–—|]+\s*', ' | ', normalized)
    
    # Normalize separators to consistent format
    normalized = re.sub(r'\s*[-–—]\s*', ' | ', normalized)
    
    # Strip leading/trailing whitespace and separators
    normalized = normalized.strip(' |-–—')
    
    return normalized


def normalize_app_name(app: str) -> str:
    """
    Normalize app/process name for cleaner display.
    
    Examples:
        "firefox.exe" → "Firefox"
        "Code.exe" → "VSCode"
        "python.exe" → "Python"
    """
    if not app:
        return ""
    
    # Remove .exe extension
    app_clean = re.sub(r'\.exe$', '', app, flags=re.IGNORECASE)
    
    # Special case mappings for common apps
    app_mappings = {
        "code": "VSCode",
        "chrome": "Chrome",
        "firefox": "Firefox",
        "msedge": "Edge",
        "notepad++": "Notepad++",
        "sublime_text": "Sublime",
        "pycharm64": "PyCharm",
        "idea64": "IntelliJ",
        "slack": "Slack",
        "discord": "Discord",
        "teams": "Teams",
        "zoom": "Zoom",
    }
    
    app_lower = app_clean.lower()
    if app_lower in app_mappings:
        return app_mappings[app_lower]
    
    # Title case for others
    return app_clean.replace("_", " ").title()


def is_relevant_activity(app: str, window_title: str, duration_seconds: float) -> bool:
    """
    Filter out irrelevant activity based on heuristics.
    
    Args:
        app: Application name
        window_title: Window title
        duration_seconds: Duration of activity
    
    Returns:
        True if activity is relevant for task classification
    """
    # Ignore very short activities (< 30 seconds)
    if duration_seconds < 30:
        return False
    
    # Ignore empty/null windows
    if not window_title or not window_title.strip():
        return False
    
    # Ignore system/background apps
    irrelevant_apps = [
        "dwm.exe",
        "explorer.exe",
        "taskmgr.exe",
        "SystemSettings.exe",
        "ApplicationFrameHost.exe",
    ]
    
    app_lower = (app or "").lower()
    if any(irrelevant in app_lower for irrelevant in irrelevant_apps):
        return False
    
    # Ignore generic window titles
    generic_titles = [
        "desktop",
        "task manager",
        "settings",
        "start menu",
        "notification",
    ]
    
    title_lower = window_title.lower()
    if any(generic in title_lower for generic in generic_titles):
        return False
    
    return True


# Domain keyword patterns for context detection
DOMAIN_PATTERNS = {
    "coding": {
        "keywords": ["code", "python", "javascript", "typescript", "html", "css", "github", "git", 
                    "vscode", "visual studio", "pycharm", "intellij", "sublime", "editor",
                    "function", "class", "debug", "terminal", "console", "repository"],
        "file_extensions": [".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".md"],
        "apps": ["code.exe", "pycharm64.exe", "sublime_text.exe", "notepad++.exe"],
        "category": "deep_development"
    },
    "documentation": {
        "keywords": ["docs", "documentation", "api", "reference", "guide", "tutorial", 
                    "readme", "wiki", "manual", "help"],
        "domains": ["docs.", "documentation.", "readthedocs", "github.io"],
        "category": "technical_research"
    },
    "communication": {
        "keywords": ["slack", "discord", "teams", "zoom", "meet", "chat", "message", 
                    "email", "gmail", "outlook", "mail", "conversation"],
        "apps": ["slack.exe", "discord.exe", "teams.exe", "zoom.exe"],
        "category": "team_meeting"
    },
    "research": {
        "keywords": ["google", "search", "stackoverflow", "reddit", "wikipedia", 
                    "article", "blog", "tutorial", "how to", "learn"],
        "domains": ["google.com", "stackoverflow.com", "reddit.com", "wikipedia.org"],
        "category": "technical_research"
    },
    "productivity": {
        "keywords": ["notion", "trello", "asana", "jira", "ticket", "task", "project",
                    "calendar", "schedule", "meeting", "todo", "checklist"],
        "apps": ["notion.exe", "trello.exe", "asana.exe"],
        "category": "administrative_work"
    },
    "content": {
        "keywords": ["youtube", "video", "watch", "stream", "netflix", "spotify",
                    "music", "podcast", "article", "news"],
        "domains": ["youtube.com", "netflix.com", "spotify.com"],
        "category": "context_switching"
    }
}


def extract_keywords(window_title: str) -> List[str]:
    """Extract meaningful keywords from normalized window title."""
    if not window_title:
        return []
    
    # Normalize first
    normalized = normalize_window_title(window_title)
    
    # Remove common noise words
    noise_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "page", "tab", "untitled", "new"
    }
    
    # Split on common separators
    title_lower = normalized.lower()
    
    # Split by separators
    tokens = re.split(r'[-_|•·:;\s]+', title_lower)
    
    # Filter out noise and short tokens
    keywords = [
        token.strip() 
        for token in tokens 
        if token.strip() and len(token.strip()) > 2 and token.strip() not in noise_words
    ]
    
    return keywords[:10]  # Limit to top 10 keywords


def extract_entities(window_title: str) -> List[str]:
    """Extract named entities (project names, tools, technologies)."""
    entities = []
    
    # Common tech/tool patterns
    tech_patterns = [
        r'\b(react|vue|angular|node|python|javascript|typescript|java|c\+\+|rust|go)\b',
        r'\b(git|github|gitlab|bitbucket)\b',
        r'\b(vscode|pycharm|intellij|sublime)\b',
        r'\b(aws|azure|gcp|docker|kubernetes)\b',
    ]
    
    title_lower = window_title.lower()
    for pattern in tech_patterns:
        matches = re.findall(pattern, title_lower, re.IGNORECASE)
        entities.extend(matches)
    
    # Extract capitalized words (potential project names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', window_title)
    entities.extend(capitalized[:3])  # Limit to 3
    
    return entities


def detect_domain(window_title: str, app: str) -> Optional[str]:
    """Detect domain/category from window title and app."""
    title_lower = window_title.lower()
    app_lower = app.lower() if app else ""
    
    scores = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0
        
        # Check keywords
        for keyword in patterns.get("keywords", []):
            if keyword in title_lower:
                score += 2
        
        # Check file extensions
        for ext in patterns.get("file_extensions", []):
            if ext in title_lower:
                score += 3
        
        # Check domains
        for domain_pattern in patterns.get("domains", []):
            if domain_pattern in title_lower:
                score += 3
        
        # Check apps
        for app_pattern in patterns.get("apps", []):
            if app_pattern in app_lower:
                score += 4
        
        if score > 0:
            scores[domain] = score
    
    if not scores:
        return None
    
    # Return domain with highest score
    best_domain = max(scores.items(), key=lambda x: x[1])
    return best_domain[0] if best_domain[1] >= 2 else None


def parse_window_context(app: str, window_title: str) -> WindowContext:
    """Parse window into structured context with normalization."""
    # Normalize inputs
    normalized_title = normalize_window_title(window_title)
    normalized_app = normalize_app_name(app)
    
    # Extract features from normalized title
    keywords = extract_keywords(normalized_title)
    entities = extract_entities(normalized_title)
    domain = detect_domain(normalized_title, normalized_app)
    
    category_hint = None
    if domain and domain in DOMAIN_PATTERNS:
        category_hint = DOMAIN_PATTERNS[domain].get("category")
    
    return WindowContext(
        app=normalized_app,
        window_title=window_title or "",
        normalized_title=normalized_title,
        domain_keywords=keywords,
        entities=entities,
        category_hint=category_hint
    )


def generate_smart_task_name(
    base_category: str,
    app: str,
    window_title: str,
    behavioral_confidence: float
) -> str:
    """
    Generate intelligent task name combining behavioral + contextual analysis.
    
    Args:
        base_category: Behavioral category (e.g., "deep_development")
        app: Process name (e.g., "firefox.exe")
        window_title: Window title
        behavioral_confidence: Confidence from behavioral analysis
    
    Returns:
        Smart task name
    """
    # Parse window context (with normalization)
    context = parse_window_context(app, window_title)
    
    # If we have high confidence in domain detection, use it
    if context.category_hint and behavioral_confidence < 0.7:
        # Domain detection overrides low-confidence behavioral classification
        base_category = context.category_hint
    
    # Build task name components
    components = []
    
    # 1. Base category (human-readable)
    category_display = base_category.replace("_", " ").title()
    components.append(category_display)
    
    # 2. Add primary entity/keyword if meaningful
    if context.entities:
        primary_entity = context.entities[0]
        if len(primary_entity) > 2:
            components.append(primary_entity)
    elif context.domain_keywords:
        # Use first meaningful keyword
        for keyword in context.domain_keywords:
            if len(keyword) > 3:
                components.append(keyword.title())
                break
    
    # 3. Add app hint if it's meaningful and different from browser
    if context.app and context.app not in ["Firefox", "Chrome", "Edge", "Safari"]:
        # Only add app if it's not a generic browser
        if len(components) < 3:
            components.append(f"({context.app})")
    
    # Join components with clean separator
    if len(components) == 1:
        # No additional context - use normalized title (truncated)
        if context.normalized_title:
            truncated = truncate_title(context.normalized_title, max_length=50)
            return f"{components[0]}: {truncated}"
        return components[0]
    
    return " - ".join(components)


def truncate_title(title: str, max_length: int = 50) -> str:
    """Truncate long title intelligently."""
    if len(title) <= max_length:
        return title
    
    # Try to break at word boundary
    truncated = title[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # At least 70% of max length
        return truncated[:last_space] + "..."
    
    return truncated + "..."


def format_task_for_display(task_id: str) -> str:
    """Format a task ID for user-friendly display."""
    # If it's already a smart name, return as-is
    if " - " in task_id or ": " in task_id:
        return task_id
    
    # Handle old format: "category | app | window"
    if " | " in task_id:
        parts = task_id.split(" | ")
        if len(parts) >= 3:
            category = parts[0].replace("_", " ").title()
            window = parts[2]
            return f"{category}: {truncate_title(window, max_length=50)}"
        elif len(parts) >= 2:
            category = parts[0].replace("_", " ").title()
            app = parts[1].replace(".exe", "").title()
            return f"{category} ({app})"
        else:
            return task_id.replace("_", " ").title()
    
    # Simple category
    return task_id.replace("_", " ").title()

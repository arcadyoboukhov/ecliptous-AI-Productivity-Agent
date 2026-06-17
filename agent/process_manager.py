"""
Process manager for main.py agent.
Handles starting, stopping, and checking status of the background agent process.
"""
import subprocess
import os
import sys
import time
from pathlib import Path

from agent.error_handling import log_component_error, ComponentType, ErrorSeverity

# psutil is an optional dependency; import lazily and handle absence gracefully
try:
    import psutil
except Exception:
    psutil = None


# Path to the project's main.py (used as target for launching the agent)
MAIN_PY_PATH = Path(__file__).parent.parent / "main.py"

# Use a global runtime directory in the user's home so agent can be controlled from any CWD
RUNTIME_DIR = Path.home() / ".productivity_agent"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
PIDFILE = RUNTIME_DIR / "agent.pid"
LOGFILE = RUNTIME_DIR / "agent.log"


def is_running():
    """Check if main.py is currently running."""
    if not PIDFILE.exists():
        return False
    
    try:
        with open(PIDFILE, 'r') as f:
            pid = int(f.read().strip())
        # Check if process with this PID exists (psutil optional)
        if psutil is None:
            # Unable to verify process without psutil; assume not running
            return False
        return psutil.pid_exists(pid)
    except Exception:
        return False


def start():
    """Start main.py as a background process."""
    try:
        if is_running():
            return False, "Agent is already running"
        
        try:
            # Ensure the runtime directory exists (in case invoked from a different user session)
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            # Start main.py in background (no console window on Windows)
            # Start the agent with its working directory set to the repository root
            cwd = MAIN_PY_PATH.parent
            # Start the agent with stdout/stderr redirected to a logfile so startup errors are captured
            # Use unbuffered file handle to ensure immediate writes
            logfile = open(LOGFILE, 'a', buffering=1)
            
            # Use the current Python executable to ensure venv consistency
            python_exe = sys.executable
            
            # Copy current environment and ensure venv is activated
            env = os.environ.copy()
            
            # Disable bytecode writing to force fresh imports
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            
            # Force unbuffered Python output to ensure logs are written immediately
            env["PYTHONUNBUFFERED"] = "1"
            
            # On Windows, explicitly set the VIRTUAL_ENV and update PATH to prioritize venv
            if sys.prefix != sys.base_prefix:  # Check if in virtual environment
                venv_path = Path(sys.prefix)
                env["VIRTUAL_ENV"] = str(venv_path)
                # Prepend venv bin/scripts to PATH
                scripts_dir = venv_path / "Scripts"
                env["PATH"] = str(venv_path) + os.pathsep + str(scripts_dir) + os.pathsep + env.get("PATH", "")
                # Add DLLs directory for pywin32
                env["PATH"] = str(venv_path / "Lib" / "site-packages" / "pywin32_system32") + os.pathsep + env["PATH"]
        except Exception as e:
            log_component_error(
                ComponentType.PROCESS,
                "process_start_init",
                e,
                ErrorSeverity.CRITICAL
            )
            return False, f"Failed to initialize agent process: {e}"
        
        if sys.platform == "win32":
            # On Windows, use DETACHED_PROCESS to completely detach from parent
            # but redirect stdio to files to avoid broken pipes
            # CREATE_NO_WINDOW would still tie the process lifecycle to the parent console
            DETACHED_PROCESS = 0x00000008
            CREATE_NO_WINDOW = 0x08000000
            proc = subprocess.Popen(
                [python_exe, "-u", str(MAIN_PY_PATH)],  # -u for unbuffered output
                stdout=logfile,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for cleaner log
                stdin=subprocess.DEVNULL,  # Close stdin to avoid blocking
                cwd=str(cwd),
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                env=env,
                close_fds=False  # Keep file handles open for logging
            )
        else:
            proc = subprocess.Popen(
                [python_exe, "-u", str(MAIN_PY_PATH)],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                start_new_session=True,
                env=env
            )
        
        # Write PID to the global runtime dir
        with open(PIDFILE, 'w') as f:
            f.write(str(proc.pid))
        
        # Give process more time to initialize and start main loop
        # The agent needs time to: init DB, start inference, attach hooks
        time.sleep(2.0)
        
        # Verify process is still alive and remove pidfile if it died
        try:
            # Check if process still exists
            proc_running = False
            if psutil is not None:
                try:
                    proc_running = psutil.pid_exists(proc.pid)
                    # Double-check by trying to get process object
                    if proc_running:
                        p = psutil.Process(proc.pid)
                        # Check if it's actually our Python process
                        proc_running = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc_running = False
            else:
                # Without psutil, assume it's running if we got this far
                proc_running = True
            
            if not proc_running:
                # read tail of log for diagnostic
                try:
                    logfile.flush()  # Ensure any buffered content is written
                    time.sleep(0.2)  # Brief wait for filesystem
                    with open(LOGFILE, 'r', encoding='utf-8', errors='replace') as lf:
                        data = lf.read()
                        tail = data[-1200:] if len(data) > 1200 else data
                except Exception:
                    tail = "(could not read logfile)"
                # clean up pidfile
                try:
                    if PIDFILE.exists():
                        PIDFILE.unlink()
                except Exception:
                    pass
                logfile.close()
                return False, f"Agent process died shortly after start; log tail:\n{tail}"
        except Exception as check_err:
            # If we can't verify, assume it started successfully
            print(f"Warning: Could not verify process status: {check_err}")
            pass

        # Keep logfile open - it will be closed when the agent process exits
        # logfile.close()  # Don't close - agent needs it
        
        return True, f"Agent started with PID {proc.pid} (cwd={cwd})"
    except Exception as e:
        log_component_error(
            ComponentType.PROCESS,
            "process_start",
            e,
            ErrorSeverity.CRITICAL
        )
        return False, f"Failed to start agent: {e}"


def stop():
    """Stop the running main.py process gracefully."""
    try:
        if not is_running():
            return False, "Agent is not running"
        
        pid = None
        try:
            with open(PIDFILE, 'r') as f:
                pid = int(f.read().strip())

            # Try to terminate gracefully using psutil if available
            if psutil is not None:
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=2)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    log_component_error(
                        ComponentType.PROCESS,
                        "process_stop_terminate",
                        e,
                        ErrorSeverity.ERROR,
                        pid=pid
                    )
            else:
                # psutil not available; best-effort using os.kill where possible
                try:
                    if sys.platform == 'win32':
                        # On Windows, os.kill exists but may require special permissions
                        os.kill(pid, 0)
                    else:
                        os.kill(pid, 15)
                except Exception:
                    pass
        except Exception as e:
            log_component_error(
                ComponentType.PROCESS,
                "process_stop_read_pid",
                e,
                ErrorSeverity.ERROR
            )

        # Clean up PID file
        try:
            if PIDFILE.exists():
                PIDFILE.unlink()
        except Exception:
            pass

        time.sleep(0.5)
        return True, f"Agent stopped (PID {pid})" if pid else "Agent stopped"
    except ProcessLookupError:
        try:
            if PIDFILE.exists():
                PIDFILE.unlink()
        except Exception:
            pass
        return True, "Agent was already stopped"
    except Exception as e:
        log_component_error(
            ComponentType.PROCESS,
            "process_stop",
            e,
            ErrorSeverity.ERROR
        )
        return False, f"Failed to stop agent: {e}"


def status():
    """Get the current status of the agent."""
    if is_running():
        try:
            with open(PIDFILE, 'r') as f:
                pid = int(f.read().strip())
            return True, f"Agent is running (PID {pid})"
        except Exception:
            return True, "Agent is running"
    else:
        return False, "Agent is not running"

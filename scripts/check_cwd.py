import os, sys
print('cwd=', os.getcwd())
print('dir listing=', os.listdir())
print('has main_window.py=', 'main_window.py' in os.listdir())
print('sys.path[0]=', sys.path[0])
print('python exe=', sys.executable)

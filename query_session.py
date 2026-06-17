"""Quick query to show tasks in session bdb1c950"""
import sqlite3

conn = sqlite3.connect('agent/storage/events.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute('''
    SELECT id, task_id, start_time, end_time, confidence, reason
    FROM task_segments
    WHERE session_id = '9fc56aa4'
    ORDER BY start_time
''')

segments = c.fetchall()

print(f'\nTasks in session 9fc56aa4:\n')
print('='*80)

for i, s in enumerate(segments, 1):
    start = s["start_time"][11:19] if len(s["start_time"]) > 19 else s["start_time"]
    end = s["end_time"][11:19] if s["end_time"] and len(s["end_time"]) > 19 else "active"
    
    print(f'{i}. Segment ID {s["id"]}:')
    print(f'   Task: {s["task_id"]}')
    print(f'   Time: {start} - {end}')
    print(f'   Confidence: {s["confidence"]:.2f}, Reason: {s["reason"]}')
    print()

print(f'Total: {len(segments)} task segments')
print('='*80)

conn.close()

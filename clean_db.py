import sqlite3


# python clean_db.py

conn = sqlite3.connect("fraud_monitor.db")

# delete first two corrupted rows
conn.execute("DELETE FROM predictions WHERE rowid <= 2")

conn.commit()

print("First two rows deleted successfully")

conn.close()
import sqlite3

connection = sqlite3.connect('ocr.db')



cur = connection.cursor()

# cur.execute("INSERT INTO list_recognition (exp_date, product_code) VALUES (?, ?)",("JUN202J", "14342TGR2309"))

print(cur.execute("SELECT * FROM list_recognition").fetchall())
# print(cur.fetchall())

# cur.execute("INSERT INTO posts (title, content) VALUES (?, ?)",
#             ('Second Post', 'Content for the second post')
#             )
# connection.commit()
connection.close()
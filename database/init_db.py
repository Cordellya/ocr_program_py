import sqlite3

connection = sqlite3.connect('ocr.db')


with open('database/schema.sql') as f:
    sql_script = f.read()

cur = connection.cursor()

cur.executescript(sql_script)

cur.execute("INSERT INTO list_recognition (exp_date, product_code) VALUES (?, ?)",('JUN2023', '12342TGR2309'))


# cur.execute("INSERT INTO posts (title, content) VALUES (?, ?)",
#             ('Second Post', 'Content for the second post')
#             )

connection.commit()
connection.close()
import pandas as pd
import sqlite3

df=pd.read_excel('data/churn.xlsx')
conn=sqlite3.connect("data/churn.db")
df.to_sql('customer',conn,if_exists='replace',index=False)

conn.close()

print('database created with', len(df), 'rows')
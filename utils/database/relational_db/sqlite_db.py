import sqlite3

import pandas as pd

from .base_relational_db import BaseRelationalDB
from ...formatting import truncate_str


class SQLiteDB(BaseRelationalDB):
    # TODO: future: all dbs should have config rather than individual args
    def __init__(self, db_path=':memory:', schema_script="", schema_path=""):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.initialize_schema(schema_script, schema_path)

    def initialize_schema(self, schema_script="", schema_path=""):
        if not schema_script and schema_path:
            with open(schema_path, 'r') as schema_file:
                schema_script = schema_file.read()
        self.cursor.executescript(schema_script)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def count(self):
        self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        return self.cursor.fetchone()[0]

    def print_all_sqlite(self, max_length=50):
        cursor = self.conn.cursor()
        # Execute the query to get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        # Fetch all table names
        tables = cursor.fetchall()

        # Iterate over each table and print its contents
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")

            # Query to get all entries from the table
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.conn)

            # Truncate each cell's content
            df = df.applymap(lambda x: truncate_str(str(x), max_length) if isinstance(x, str) else x)

            # Print the table entries in a pretty format
            print(df.to_string(index=False))
            print("\n" + "=" * 50 + "\n")

    def create_table(self, table_name, schema):
        """
        Creates a table in the relational database.

        Args:
            table_name (str): Name of the table.
            schema (str): SQL schema for the table.
        """
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
        self.conn.commit()

    def update(self, table_name, entry, **kwargs):
        """
        Inserts an entry into the relational database.

        Args:
            table_name (str): Name of the table.
            entry (dict): Dictionary containing column-value pairs.
        """
        columns = ', '.join(entry.keys())
        placeholders = ', '.join('?' * len(entry))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(sql, tuple(entry.values()))
        self.conn.commit()

    def retrieve(self, table_name, query_conditions, **kwargs):
        """
        Queries the relational database.

        Args:
            table_name (str): Name of the table.
            query_conditions (str): SQL conditions for the query.

        Returns:
            list: List of query results.
        """
        sql = f"SELECT * FROM {table_name} WHERE {query_conditions}"
        self.cursor.execute(sql)
        return self.cursor.fetchall()


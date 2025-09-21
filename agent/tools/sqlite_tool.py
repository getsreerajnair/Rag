import sqlite3
from typing import List, Tuple, Any



class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path  

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_tables(self) -> List[str]:
        with self._connect() as conn:  
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            return [row["name"] for row in cursor.fetchall()]

    def get_columns(self, table_name: str) -> List[str]:
        # validate table existence to avoid accidental injection via identifier
        if table_name not in set(self.get_tables()):
            raise ValueError(f"Table '{table_name}' does not exist")
        with self._connect() as conn:  
            # quote the identifier to handle spaces/special chars
            cursor = conn.execute(f'PRAGMA table_info("{table_name}");')
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            return [row["name"] for row in cursor.fetchall()]

    def execute_query(self, query: str) -> dict:
        try:
            with self._connect() as conn:
                cursor = conn.execute(query)
                columns = [col[0] for col in cursor.description] if cursor.description else []
                rows = [tuple(row) for row in cursor.fetchall()]
                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e)
            }


if __name__ == '__main__':
    query_ins=SQLiteTool("data/northwind.sqlite")
    print(f"list of tables: {query_ins.get_tables()}\n")
    print(f"list of columns: {query_ins.get_columns('Orders')}\n")
    sql = 'SELECT * FROM "Order Details" LIMIT 5;'
    result = query_ins.execute_query(sql)
    print(f"query success: {result['success']}")
    print(f"query columns: {result['columns']}")
    print(f"query rows: {result['rows']}")
    
    # Test error handling
    invalid_sql = 'SELECT * FROM NonExistentTable;'
    error_result = query_ins.execute_query(invalid_sql)
    print(f"invalid query success: {error_result['success']}")
    print(f"invalid query error: {error_result['error']}")

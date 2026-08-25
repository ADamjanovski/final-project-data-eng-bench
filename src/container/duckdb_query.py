import json
import os
import sys
import duckdb


def main() -> int:
    limit = int(sys.argv[1])
    query = sys.stdin.read()
    database = os.environ.get("DUCKDB_PATH", "/app/database/retail.duckdb")

    try:
        connection = duckdb.connect(database, read_only=True)
        try:
            cursor = connection.execute(query)
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
        finally:
            connection.close()
        payload = {
            "columns": columns,
            "rows": [
                ["NULL" if value is None else str(value) for value in row]
                for row in rows[:limit]
            ],
            "truncated": len(rows) > limit,
        }
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

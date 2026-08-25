"""BigQuery storage backend — drop-in BaseStorage replacement."""

from __future__ import annotations

from typing import Any, Optional

from agent_metering.storage import BaseStorage, UsageRecord

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - optional dependency
    bigquery = None  # type: ignore[assignment]


class BigQueryStorage(BaseStorage):
    """Persist usage records in Google BigQuery.

    Drop-in replacement::

        Meter(storage=BigQueryStorage(
            project_id="my-project",
            dataset_id="metering",
            table_id="usage_log",
        ))
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "usage_log",
        client: Any = None,
    ) -> None:
        if bigquery is None and client is None:
            raise ImportError(
                "google-cloud-bigquery is required for BigQueryStorage. "
                "Install it with: pip install google-cloud-bigquery"
            )
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self._client = client if client is not None else bigquery.Client(project=project_id)
        self._table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
        try:
            self._client.get_dataset(dataset_ref)
        except Exception:
            self._client.create_dataset(dataset_ref, exists_ok=True)

        schema = [
            bigquery.SchemaField("timestamp", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("customer_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("feature", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("provider", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("input_tokens", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("output_tokens", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("cost_usd", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("latency_ms", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("extra_metadata", "STRING", mode="NULLABLE"),
        ]
        table = bigquery.Table(self._table_ref, schema=schema)
        self._client.create_table(table, exists_ok=True)

    def write(self, record: UsageRecord) -> None:
        row = {
            "timestamp": record.timestamp,
            "customer_id": record.customer_id,
            "feature": record.feature,
            "provider": record.provider,
            "model": record.model,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": record.cost_usd,
            "latency_ms": record.latency_ms,
            "extra_metadata": record.extra_metadata,
        }
        errors = self._client.insert_rows_json(self._table_ref, [row])
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

    def _aggregate(
        self, group_column: str, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        if since_ts is None:
            since_ts = 0.0
        sql = f"""
            SELECT
                COALESCE({group_column}, 'unknown') AS group_key,
                SUM(cost_usd) AS total_cost_usd,
                SUM(input_tokens + output_tokens) AS total_tokens,
                COUNT(*) AS call_count
            FROM `{self._table_ref}`
            WHERE timestamp >= @since_ts
            GROUP BY group_key
            ORDER BY total_cost_usd DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("since_ts", "FLOAT64", since_ts),
            ]
        )
        rows = self._client.query(sql, job_config=job_config).result()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[row["group_key"]] = {
                "total_cost_usd": float(row["total_cost_usd"] or 0.0),
                "total_tokens": int(row["total_tokens"] or 0),
                "call_count": int(row["call_count"] or 0),
            }
        return result

    def query_cost_by_customer(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self._aggregate("customer_id", since_ts)

    def query_cost_by_feature(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self._aggregate("feature", since_ts)

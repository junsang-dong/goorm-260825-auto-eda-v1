"""CSV loading with encoding / delimiter auto-detection and basic guardrails."""
from __future__ import annotations

import csv
import io

import pandas as pd

ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
MAX_ROWS_BEFORE_SAMPLING = 200_000


class CSVLoadError(Exception):
    pass


def _sniff_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv(file_bytes: bytes, filename: str = "uploaded.csv") -> tuple[pd.DataFrame, dict]:
    """Load raw CSV bytes into a DataFrame, trying multiple encodings/delimiters.

    Returns (dataframe, meta) where meta records which encoding/delimiter worked
    and whether the data was sampled down for size.
    """
    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            text = file_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue

        sample = text[:5000]
        delimiter = _sniff_delimiter(sample)

        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
        except Exception as exc:  # noqa: BLE001 - fall back to next encoding
            last_error = exc
            continue

        if df.shape[1] <= 1:
            # Sniffing likely failed; retry with pandas' own separator inference.
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        meta = {
            "filename": filename,
            "encoding": encoding,
            "delimiter": delimiter,
            "sampled": False,
            "original_rows": len(df),
        }

        if len(df) > MAX_ROWS_BEFORE_SAMPLING:
            df = df.sample(n=MAX_ROWS_BEFORE_SAMPLING, random_state=42).reset_index(drop=True)
            meta["sampled"] = True

        return df, meta

    raise CSVLoadError(
        f"CSV 파일을 읽을 수 없습니다 (인코딩/구분자 자동감지 실패): {last_error}"
    )

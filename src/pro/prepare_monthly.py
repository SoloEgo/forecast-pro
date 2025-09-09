import pandas as pd

def prepare_monthly(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["SKU"] = df["SKU"].astype(str)

    # Группировка
    if "group" not in df.columns:
        df["group"] = "_" + df["SKU"]

    # СК
    if "SK" not in df.columns:
        df["SK"] = "UNKNOWN"

    if freq == "D":
        # По-дневные данные: пересчитываем oos_flag как долю наличия
        df["oos_flag"] = df["oos_flag"].fillna(1)
        grouped = df.groupby(["SKU", df["date"].dt.to_period("M")])
        oos_share = grouped["oos_flag"].mean().reset_index()
        oos_share["date"] = oos_share["date"].dt.to_timestamp()

        # Агрегируем продажи и добавляем oos
        df = (
            df.groupby(["SKU", "date", "group", "SK"])
              .agg({"qty": "sum"})
              .reset_index()
              .merge(oos_share, on=["SKU", "date"], how="left")
        )
    else:
        # Месячные данные — oos_flag уже в нужном виде
        df["oos_flag"] = df["oos_flag"].fillna(1)

    return df

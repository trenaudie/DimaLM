import numpy as np
import pandas as pd
import pyodbc


def get_isin_mapping(isin_tuple, conn_string: str = None):
    query_str = f"""
      SELECT SYM_ISIN.isin as isin, SYM_SEC.factset_entity_id as fsym_e
      FROM [fds].[sym_v1].[sym_isin] SYM_ISIN
      JOIN [fds].[fp_v2].[fp_sec_entity] SYM_SEC
      ON SYM_ISIN.fsym_id = SYM_SEC.fsym_id
      WHERE SYM_ISIN.isin IN {isin_tuple}
    """

    # conn1 = (
    #     "DRIVER={ODBC Driver 17 for SQL Server};"
    #     "SERVER=RAMASRVSQL03.ramai.local;"
    #     "UID=linuxml;"
    #     "PWD=lymph-nBFKGPj5"
    # )
    if conn_string is None:
        conn_string = "DRIVER={SQL Server};SERVER=RAMASRVSQL03;DATABASE=fds;Trusted_Connection=yes"
    con = pyodbc.connect(conn_string)
    mapping_df = pd.read_sql(
        query_str,
        con,
    )
    con.close()

    print("mapping table:", mapping_df.shape, mapping_df.columns)

    return mapping_df


def get_all_isin(news_df):
    all_isin = []
    for i in news_df["ISIN"]:
        if i:
            all_isin += i.split(",")

    return list(set(all_isin))


# ---
if __name__ == "__main__":
    isin_list = get_all_isin(news_df)
    isin_map = get_isin_mapping(isin_tuple=tuple(isin_list))

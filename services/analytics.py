import pandas as pd
import numpy as np


def generate_dashboard_data(df):

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    total_rows = len(df)
    total_columns = len(df.columns)

    analytics = {
        'dataset': {
            'rows': total_rows,
            'columns': total_columns,
            'numeric_columns': numeric_columns
        },
        'summary': {},
        'charts': {}
    }

    # ==================================================
    # SUMMARY STATS
    # ==================================================

    for col in numeric_columns:

        analytics['summary'][col] = {
            'sum': round(float(df[col].sum()), 2),
            'mean': round(float(df[col].mean()), 2),
            'max': round(float(df[col].max()), 2),
            'min': round(float(df[col].min()), 2)
        }

    # ==================================================
    # SIMPLE CHART DATA
    # ==================================================

    if len(numeric_columns) > 0:

        col = numeric_columns[0]

        values = df[col].head(12).tolist()

        analytics['charts']['line_chart'] = {
            'labels': [str(i + 1) for i in range(len(values))],
            'values': values
        }

    return analytics
import pandas as pd


def read_file(filepath):

    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)

    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)

    elif filepath.endswith('.xls'):
        return pd.read_excel(filepath)

    elif filepath.endswith('.json'):
        return pd.read_json(filepath)

    else:
        raise Exception('Unsupported file format')
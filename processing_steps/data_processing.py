import numpy as np
import pandas as pd
import numpy as np
import pandas as pd


def symmetrize(data):
    """
    Symmetrize a DataFrame by making it symmetric, where each element (i, j)
    is equal to the minimum of (i, j) and (j, i) if both are not NaN,
    otherwise taking the non-NaN value.

    Parameters:
    data (DataFrame): Input DataFrame to be symmetrized.

    Returns:
    DataFrame: Symmetrized DataFrame.
    """
    mat = data.values
    sym_mat = np.where(np.isnan(mat), mat.T, mat)
    sym_mat = np.where(np.isnan(sym_mat), sym_mat.T, sym_mat)
    sym_mat = np.minimum(sym_mat, sym_mat.T, where=~np.isnan(sym_mat), out=sym_mat)

    return pd.DataFrame(sym_mat, index=data.index, columns=data.columns)

def intersection_of_df(df, df1):
    """
    Return the intersection of two DataFrames based on common indices and columns.

    Parameters:
    df (DataFrame): First input DataFrame.
    df1 (DataFrame): Second input DataFrame.

    Returns:
    tuple: A tuple containing the intersected DataFrames (df, df1).
    """
    # Find the common indices and columns
    common_index = df.index.intersection(df1.index)
    common_columns = df.columns.intersection(df1.columns)

    # Subset the DataFrames to the common indices and columns
    df_intersection = df.loc[common_index, common_columns]
    df1_intersection = df1.loc[common_index, common_columns]

    return df_intersection, df1_intersection


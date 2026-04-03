import pandas as pd


def normalize_chat_time(df, floor=-90, ceiling=3600):
    """Clipped min-max normalization of chatTime to [0, 1].

    Values below floor are clipped to 0, values above ceiling are clipped to 1.
    """
    clipped = df["chatTime"].clip(lower=floor, upper=ceiling)
    df["chatTimeNorm"] = (clipped - floor) / (ceiling - floor)
    return df

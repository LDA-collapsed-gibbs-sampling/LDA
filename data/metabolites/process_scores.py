#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path

scores_dir = Path("scores").resolve()
series_list = []
for item in scores_dir.iterdir():
    pwm_name = item.stem
    with open(item, "r") as f:
        pwm_dict = {
            line.strip().split()[0]: len(line.strip().split()[1:])
            for line in f
            if ">" not in line
        }
    series_list.append(pd.Series(pwm_dict, name=pwm_name))

pd.concat(series_list, axis=1).to_csv("scores.csv")

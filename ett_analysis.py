import pandas as pd
import pprint

for ett_type in ["h1", "h2", "m1", "m2"]:
    data = pd.read_pickle(f"ALL_RESULT_ett_{ett_type}.pkl")
    print(f"{ett_type}: ")
    pprint.pprint(data[ett_type])


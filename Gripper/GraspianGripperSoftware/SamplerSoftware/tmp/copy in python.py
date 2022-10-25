import pandas as pd
data = pd.DataFrame()
data = pd.read_csv('sampler.txt',sep=", ", header = None)
print(data)
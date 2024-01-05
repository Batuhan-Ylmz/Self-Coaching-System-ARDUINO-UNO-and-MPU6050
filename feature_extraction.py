import pandas as pd
import numpy as np
from scipy import stats

def getfeature(data):
    data = data.astype(float)  # Convert data to float explicitly
    fmean = np.mean(data)
    fstd = np.std(data)
    fmax = np.max(data)
    fmin = np.min(data)
    fkurtosis = stats.kurtosis(data)
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    fzero = zero_crosses.size / len(data)
    return fmean, fstd, fmax, fmin, fkurtosis, fzero

def extractFeature(raw_data, ws, hop, dfname):
    fmean = []
    fstd = []
    fmax = []
    fmin = []
    fkurtosis = []
    fzero = []
    flabel = []
    
    # Exclude the timestamp column and header
    raw_data = raw_data.iloc[:, 1:]

    for i in range(ws, len(raw_data), hop):
        m, s, ma, mi, k, z = getfeature(raw_data.iloc[i - ws + 1:i, 0])
        fmean.append(m)
        fstd.append(s)
        fmax.append(ma)
        fmin.append(mi)
        fzero.append(z)
        fkurtosis.append(k)

        flabel.append(dfname)

    rdf = pd.DataFrame(
        {'mean': fmean,
         'std': fstd,
         'max': fmax,
         'min': fmin,
         'kurtosis': fkurtosis,
         'zerocross': fzero,
         'label': flabel
         })
    return rdf

# Read CSV files excluding the header and timestamp column
df0 = pd.read_csv('first_type_of_movement.csv', header=None).iloc[:, 1:]
df0_rdf = extractFeature(df0, 10, 10, "0")

df1 = pd.read_csv('second_type_of_movement.csv', header=None).iloc[:, 1:]
df1_rdf = extractFeature(df1, 10, 10, "1")

# df2 = pd.read_csv('third_type_of_movement.csv', header=None).iloc[:, 1:]
# df2_rdf = extractFeature(df2, 10, 10, "2")
# ...

# Concatanate the dataframes
df = pd.concat([df0_rdf, df1_rdf])

# Write to CSV
df.to_csv(r'project_features.csv', index=None)

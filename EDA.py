import h5py
import numpy as np

filename = 'data/TaxiBJ/BJ15_M32x32_T30_InOut.h5'
f = h5py.File(filename)
for ke in f.keys():
    print(ke, f[ke].shape)

# We observe that the shape is 5596x2x32x32. This corresponds to TxCxHxW

with h5py.File(filename, "r") as f:
    data = np.array(f["data"], dtype=np.float32)
    dates = np.array(f["date"])

print(dates)
# Dates are of shape b'YYYYMMDDTT -> b indicates bytes
print("\n")
print(data[0].shape)

# Inflow at timestamp 0
print(data[0][0])
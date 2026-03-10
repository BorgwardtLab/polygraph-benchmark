import pickle
from pyprojroot import here

REPO_ROOT = here()
p = REPO_ROOT / "data" / "DIGRESS" / "training-iterations" / "119_steps.pkl"
with open(p, "rb") as f:
    data = pickle.load(f)
print("Type:", type(data))
print("Length:", len(data))
if len(data) > 0:
    print("First item type:", type(data[0]))

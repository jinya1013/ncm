import numpy as np, datasets
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = datasets.load_from_disk(
    "/scratch/prj0000000245/sakurai/data/lm1b/lm1b_test_bs128_wrapped.dat"
)
print(ds)
ids = np.array(ds[:3]["input_ids"])
am = np.array(ds[:3]["attention_mask"])
print(
    "shape",
    ids.shape,
    "max id",
    np.array(ds[:2000]["input_ids"]).max(),
    "mask mean",
    am.mean(),
)
for r in ids[:2]:
    print(r[:20].tolist())
    print(repr(tok.decode(r[:60])))
b = np.fromfile("data/lm1b/valid_bert_30522.bin", dtype=np.uint16)
off = np.load("data/lm1b/valid_bert_30522_offsets.npy")
print("hf bin", b.shape, b[:30].tolist(), "offsets", off[:5].tolist(), off.shape)
print(repr(tok.decode(b[:60])))

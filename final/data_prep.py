import pandas as pd
from datasets import load_dataset

ds = load_dataset("DataForGood/ome-hackathon-season-14")

df = ds['train'].to_pandas()

df['report_text_new'] = df['report_text'].apply(preprocess_text)




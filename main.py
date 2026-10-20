import kagglehub
import numpy as nm
import pandas as pd

# Download latest version
path_sp500 = kagglehub.dataset_download("pritish509/s-and-p-500-esg-risk-ratings")
path_finance = kagglehub.dataset_download("shriyashjagtap/esg-and-financial-performance-dataset")

print("Path to dataset files:", path_sp500)

print("Path to dataset files:", path_finance)


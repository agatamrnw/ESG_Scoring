import kagglehub

# Download latest version
path = kagglehub.dataset_download("pritish509/s-and-p-500-esg-risk-ratings")

print("Path to dataset files:", path)
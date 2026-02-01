import kagglehub

# Download latest version
path = kagglehub.dataset_download("pradumn203/payment-date-prediction-for-invoices-dataset")

print("Path to dataset files:", path)
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
df = pd.read_csv(DATA_PATH)
df['invoice_id'] = df['invoice_id'] .replace([np.inf, -np.inf], 0).fillna(0).astype(int)
df['due_in_date'] = pd.to_datetime(
    df['due_in_date'],
    format='%Y%m%d',
    errors='coerce'
)

invoice_texts = []

for _, row in df.iterrows():
    due_date = (
        row['due_in_date'].date()
        if not pd.isna(row['due_in_date'])
        else "unknown"
    )

for index, row in df.iterrows():
    text = f"""
Invoice ID {(row['invoice_id'])}.
Customer name is {row['name_customer']}.
Business code is {row['business_code']}.
Invoice amount is {row['total_open_amount']} {row['invoice_currency']}.
Due date is {due_date}.
Payment terms are {row['cust_payment_terms']}.
Invoice status is {"unpaid" if row['isOpen'] == 1 else "paid"}.
"""
    invoice_texts.append(text.strip())

# Preview first invoice text
print(invoice_texts[0])
print("\nTotal invoices converted:", len(invoice_texts))

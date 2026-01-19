import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize Faker for fake names, addresses, IPs
fake = Faker()

# Number of samples
n = 25000

# Possible categories
payment_types = ['Credit Card', 'Debit Card', 'PayPal', 'Net Banking']
product_categories = ['Electronics', 'Clothing', 'Toys & Games', 'Health', 'Home', 'Sports']
device_types = ['Desktop', 'Mobile', 'Tablet']

data = []

for i in range(n):
    transaction_id = fake.uuid4()[:12]
    customer_id = fake.uuid4()[:8]
    amount = round(random.uniform(10.0, 1000.0), 2)
    payment_type = random.choice(payment_types)
    product = random.choice(product_categories)
    quantity = random.randint(1, 5)
    device = random.choice(device_types)
    ip = fake.ipv4()
    shipping = fake.street_address()
    billing = fake.street_address()
    account_age = random.randint(1, 400)
    transaction_hour = random.randint(0, 23)

    # Simulate fraud (around 4% fraud cases)
    is_fraud = np.random.choice([0, 1], p=[0.96, 0.04])

    data.append([
        transaction_id, customer_id, amount, payment_type,
        product, quantity, device, ip,
        shipping, billing, is_fraud,
        account_age, transaction_hour
    ])

# Create DataFrame
columns = [
    'TransactionID', 'CustomerID', 'TransactionAmount', 'PaymentType',
    'ProductCategory', 'Quantity', 'DeviceType', 'IPAddress',
    'ShippingAddress', 'BillingAddress', 'IsFraudulent',
    'AccountAge', 'TransactionHour'
]

df = pd.DataFrame(data, columns=columns)

# Save as CSV
df.to_csv('ecommerce_fraud_data.csv', index=False)

print("✅ Dataset generated successfully!")
print(f"Total Rows: {len(df)}")
print(df.head())

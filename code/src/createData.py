import pandas as pd

# Sample email dataset
data = {
    "text": [
        "Hi, its Adjustment.",
        "Hi ,Its AU Transfer",
        "Hi, Its Closing notice",
        "Hi, Its Commitment change",
        "Hi, Its Fee Payment",
        "Hi, Its Money movement inbound",
		"Hi Its Money movement outbound"
		
    ],
    "label": ["Adjustment", "AUTransfer", "ClosingNotice", "CommitmentChange", "FeePayment", "MoneyMovementInbound", "MoneyMovementOutbound"]
}

# Create DataFrame 
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("emails.csv", index=False)

print("CSV file created successfully!")

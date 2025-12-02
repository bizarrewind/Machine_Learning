import random
import datetime

# Define the start and end dates for random date generation
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2025, 12, 31)

# Calculate the difference in days
time_between_dates = end_date - start_date
days_between_dates = time_between_dates.days

# List of strings (e.g., product SKUs or types)
product_types = ["CPU", "GPU", "RAM", "SSD", "HDD", "Monitor", "Keyboard"]

# Open a file named "SalesData.csv" in write mode ("w").
with open("SalesData.csv", "w") as f:
    # Write a header row for clarity
    f.write("TransactionID,ProductType,SalesValue,TransactionDate\n")

    # Loop 10,000 times to generate 10,000 rows of data.
    for i in range(1, 10001):
        # 1. TransactionID: The loop index 'i' serves as a unique, sequential ID.
        transaction_id = i

        # 2. ProductType: Randomly select one of the defined product strings.
        product = random.choice(product_types)

        # 3. SalesValue: Generate a random floating-point number between 50.00 and 1500.00, rounded to 2 decimal places.
        sales_value = round(random.uniform(50.00, 1500.00), 2)

        # 4. TransactionDate: Generate a random number of days from the start date and add it to get a random date.
        random_number_of_days = random.randrange(days_between_dates)
        transaction_date = start_date + datetime.timedelta(days=random_number_of_days)

        f.write(f"{transaction_id},{product},{sales_value},{transaction_date}\n")

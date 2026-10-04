import csv
import sys

def get_image_url(product_name):
    # Using extremely reliable picsum.photos
    seed = abs(hash(product_name)) % 10000
    return f"https://picsum.photos/seed/{seed}/320/240"

CSV_PATH = "data/products.csv"

def process_csv():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row["image_url"] = get_image_url(row["product_name"])
            rows.append(row)

    with open(CSV_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    process_csv()
    print("CSV images updated successfully.")

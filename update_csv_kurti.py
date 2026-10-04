import csv
import sys

new_products = [
    {
        "product_id": "P116", "product_name": "Cotton Printed Kurti", "brand": "Biba",
        "category": "Clothing", "subcategory": "Kurti", "price": "999", "original_price": "1499",
        "rating": "4.3", "review_count": "150", 
        "description": "Comfortable cotton printed kurti for daily wear.",
        "features": "\"Cotton, Printed, 3/4th Sleeves, Daily Wear\"",
        "color": "Yellow", "availability": "In Stock",
        "image_url": "https://picsum.photos/seed/kurti1/320/240"
    },
    {
        "product_id": "P117", "product_name": "Designer Anarkali Kurti", "brand": "W for Woman",
        "category": "Clothing", "subcategory": "Kurti", "price": "1999", "original_price": "2999",
        "rating": "4.6", "review_count": "320", 
        "description": "Elegant designer Anarkali kurti for festive occasions.",
        "features": "\"Anarkali, Designer, Festive wear, Full Length\"",
        "color": "Red", "availability": "In Stock",
        "image_url": "https://picsum.photos/seed/kurti2/320/240"
    },
    {
        "product_id": "P118", "product_name": "Embroidered Silk Kurti", "brand": "Aurelia",
        "category": "Clothing", "subcategory": "Kurti", "price": "1499", "original_price": "2199",
        "rating": "4.5", "review_count": "210", 
        "description": "Beautiful embroidered silk kurti with traditional motifs.",
        "features": "\"Silk, Embroidered, Traditional, Party wear\"",
        "color": "Green", "availability": "In Stock",
        "image_url": "https://picsum.photos/seed/kurti3/320/240"
    }
]

CSV_PATH = "data/products.csv"

def process_csv():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            
    for new_p in new_products:
        rows.append(new_p)

    with open(CSV_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    process_csv()
    print("Kurti products added successfully.")

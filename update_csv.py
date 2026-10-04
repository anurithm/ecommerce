import csv
import uuid
import sys

def get_image_url(product_name, category_name):
    # Using loremflickr - it redirects to a random image of that keyword.
    # To keep images constant across reloads, we lock it with a seed.
    seed = abs(hash(product_name)) % 10000
    keyword = category_name.lower()
    return f"https://loremflickr.com/320/240/{keyword}?lock={seed}"

new_products = [
    {
        "product_id": "P111", "product_name": "Floral Summer Dress", "brand": "Zara",
        "category": "Clothing", "subcategory": "Dress", "price": "2999", "original_price": "3999",
        "rating": "4.6", "review_count": "150", 
        "description": "A beautiful lightweight floral summer dress perfect for casual outings.",
        "features": "\"Floral pattern, Lightweight, Breathable fabric, V-neck\"",
        "color": "Floral/White", "availability": "In Stock"
    },
    {
        "product_id": "P112", "product_name": "Elegant Evening Black Gown", "brand": "H&M",
        "category": "Clothing", "subcategory": "Dress", "price": "4999", "original_price": "6999",
        "rating": "4.8", "review_count": "430", 
        "description": "Turn heads at your next event with this elegant evening black gown.",
        "features": "\"Sleeveless, High slit, Premium material, Evening wear\"",
        "color": "Black", "availability": "In Stock"
    },
    {
        "product_id": "P113", "product_name": "Boho Maxi Midi Dress", "brand": "Mango",
        "category": "Clothing", "subcategory": "Dress", "price": "3499", "original_price": "4499",
        "rating": "4.3", "review_count": "210", 
        "description": "Comfortable boho style maxi dress, great for beach wear or vacations.",
        "features": "\"Maxi length, Bohemian pattern, Comfortable fit, Stretchable\"",
        "color": "Red/Orange", "availability": "In Stock"
    },
    {
        "product_id": "P114", "product_name": "Formal Bodycon Dress", "brand": "Allen Solly",
        "category": "Clothing", "subcategory": "Dress", "price": "2599", "original_price": "",
        "rating": "4.5", "review_count": "180", 
        "description": "A perfect formal bodycon dress for office wear and professional meetings.",
        "features": "\"Bodycon, Formal, Knee-length, Stretchable\"",
        "color": "Navy Blue", "availability": "In Stock"
    },
    {
        "product_id": "P115", "product_name": "Casual Denim Shift Dress", "brand": "Levi's",
        "category": "Clothing", "subcategory": "Dress", "price": "3999", "original_price": "5499",
        "rating": "4.2", "review_count": "320", 
        "description": "Stylish casual denim shift dress with pockets and adjustable straps.",
        "features": "\"Denim, Shift dress, Multiple pockets, Casual wear\"",
        "color": "Blue", "availability": "In Stock"
    }
]

CSV_PATH = "data/products.csv"

def process_csv():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Update the image_url
            row["image_url"] = get_image_url(row["product_name"], row["category"])
            rows.append(row)
            
    # Add new products
    for new_p in new_products:
        new_p["image_url"] = get_image_url(new_p["product_name"], new_p["subcategory"])
        rows.append(new_p)

    with open(CSV_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    process_csv()
    print("CSV updated successfully.")

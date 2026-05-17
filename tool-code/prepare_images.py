import os
import requests
import pandas as pd


# CONFIG

EXCEL_FILE = "uploads/updated_output.xlsx" 
OUTPUT_FOLDER = "static/radiator_images"

UPDATED_EXCEL_FILE = EXCEL_FILE

IMAGE_COLUMN_NAME = "Image"

SKU_COLUMN = "Castrads SKU"

DRIVE_LINK_COLUMN = "Link to google drive image"


# CREATE IMAGE FOLDER

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# GOOGLE DRIVE LINK CONVERTER

def convert_drive_link(url):

    if pd.isna(url):
        return None

    url = str(url)

    try:

        # Format:
        # https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        if "/file/d/" in url:

            file_id = url.split("/d/")[1].split("/")[0]

            return f"https://drive.google.com/uc?export=download&id={file_id}"

        # Format:
        # https://drive.google.com/open?id=FILE_ID
        elif "open?id=" in url:

            file_id = url.split("open?id=")[1]

            return f"https://drive.google.com/uc?export=download&id={file_id}"

        # Already converted
        elif "uc?id=" in url or "uc?export=" in url:

            return url

    except Exception:
        return None

    return None


# DOWNLOAD IMAGE

def download_image(url, save_path):

    try:

        response = requests.get(url, timeout=30)

        if response.status_code == 200:

            with open(save_path, "wb") as f:
                f.write(response.content)

            return True

        return False

    except Exception as e:

        print(f"Error downloading image: {e}")

        return False


# LOAD EXCEL

print("Loading Excel file...")

df = pd.read_excel(EXCEL_FILE)

print(f"Rows found: {len(df)}")


# CREATE IMAGE COLUMN

image_filenames = []

success_count = 0
failed_count = 0


# PROCESS EACH ROW

for index, row in df.iterrows():

    sku = row.get(SKU_COLUMN)
    drive_link = row.get(DRIVE_LINK_COLUMN)

    # Skip invalid rows
    if pd.isna(sku) or pd.isna(drive_link):

        image_filenames.append("")

        failed_count += 1

        continue

    sku = str(sku).strip()

    # Create filename
    filename = f"{sku}.jpg"

    save_path = os.path.join(OUTPUT_FOLDER, filename)

    # Convert Drive link
    direct_url = convert_drive_link(drive_link)

    if not direct_url:

        print(f"Invalid Drive link for SKU: {sku}")

        image_filenames.append("")

        failed_count += 1

        continue

    print(f"Downloading: {sku}")

    success = download_image(direct_url, save_path)

    if success:

        print(f"Saved: {filename}")

        image_filenames.append(filename)

        success_count += 1

    else:

        print(f"Failed: {sku}")

        image_filenames.append("")

        failed_count += 1


# ADD IMAGE COLUMN

df[IMAGE_COLUMN_NAME] = image_filenames


# SAVE UPDATED EXCEL

df.to_excel(UPDATED_EXCEL_FILE, index=False)


# SUMMARY

print("\n======================================")
print("PROCESS COMPLETE")
print("======================================")

print(f"Images downloaded successfully: {success_count}")
print(f"Images failed: {failed_count}")

print(f"\nImages folder:")
print(OUTPUT_FOLDER)

print(f"\nUpdated Excel saved")

print("\nDONE ")
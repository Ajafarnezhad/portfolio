import qrcode
from PIL import Image
import re
import os

def validate_url(url):
    """Validate if the input is a valid URL."""
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def create_vcard(first_name, last_name, email, phone):
    """Create a vCard string for QR code."""
    vcard = (
        f"BEGIN:VCARD\n"
        f"VERSION:3.0\n"
        f"N:{last_name};{first_name};;;\n"
        f"FN:{first_name} {last_name}\n"
        f"EMAIL;TYPE=WORK:{email}\n"
        f"TEL;TYPE=CELL:{phone}\n"
        f"END:VCARD"
    )
    return vcard

def generate_barcode(data, filename, fill_color="black", back_color="white", box_size=10, border=4):
    """Generate and save a QR code with specified settings."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        img.save(filename)
        print(f"QR code saved as {filename}")
        
        # Try to display the image if possible
        try:
            img.show()
        except Exception:
            print("Preview not available. Open the saved file to view the QR code.")
            
    except Exception as e:
        print(f"Error generating QR code: {e}")

def main():
    """Main function to run the barcode generator."""
    print("Welcome to Barcode Generator! Create QR codes for URLs, text, or vCards.")
    
    while True:
        try:
            print("\nChoose QR code type:")
            print("1. URL (e.g., website link)")
            print("2. Text (e.g., a message)")
            print("3. vCard (contact information)")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "4":
                print("Goodbye!")
                break
                
            if choice not in ["1", "2", "3"]:
                print("Invalid choice! Please select 1, 2, 3, or 4.")
                continue
                
            # Get data based on choice
            if choice == "1":
                url = input("Enter URL (e.g., https://example.com): ").strip()
                if not validate_url(url):
                    print("Invalid URL! Please enter a valid URL (e.g., https://example.com).")
                    continue
                data = url
                default_filename = "url_qr.png"
                
            elif choice == "2":
                text = input("Enter text: ").strip()
                if not text:
                    print("Error: Text cannot be empty.")
                    continue
                data = text
                default_filename = "text_qr.png"
                
            else:  # choice == "3"
                first_name = input("Enter first name: ").strip()
                last_name = input("Enter last name: ").strip()
                email = input("Enter email: ").strip()
                phone = input("Enter phone number: ").strip()
                if not (first_name and last_name and email and phone):
                    print("Error: All vCard fields must be filled.")
                    continue
                data = create_vcard(first_name, last_name, email, phone)
                default_filename = "vcard_qr.png"
            
            # Get filename
            filename = input(f"Enter filename for QR code (default: {default_filename}): ").strip() or default_filename
            if not filename.endswith(".png"):
                filename += ".png"
                
            # Get customization options
            fill_color = input("Enter QR code color (default: black): ").strip() or "black"
            back_color = input("Enter background color (default: white): ").strip() or "white"
            box_size = int(input("Enter box size (default: 10): ") or 10)
            border = int(input("Enter border size (default: 4): ") or 4)
            
            # Generate and save QR code
            generate_barcode(data, filename, fill_color, back_color, box_size, border)
            
        except ValueError:
            print("Error: Please enter valid numbers for box size and border.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        # Ask to continue
        if input("\nCreate another QR code? (yes/no): ").lower().strip() != "yes":
            print("Thanks for using Barcode Generator!")
            break

if __name__ == "__main__":
    main()
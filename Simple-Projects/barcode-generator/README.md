Barcode Generator

This is a versatile Python script that generates QR codes for various data types, including URLs, text, or vCard contact information. Users can customize the QR code's appearance and save it as a PNG file.

Features



Generate QR codes for URLs, plain text, or vCards (contact details).

Customize QR code appearance (color, background, box size, border).

Validates user inputs (e.g., URL format, required fields).

Saves QR codes as PNG files with user-defined filenames.

Displays a preview of the QR code (if supported by the system).

Error handling for invalid inputs or file-saving issues.



Installation



Ensure you have Python 3.x installed.

Install the required libraries:pip install qrcode pillow





Clone the repository:git clone https://github.com/Ajafarnezhad/portfolio.git





Navigate to the project directory:cd portfolio/Simple-Projects/barcode-generator







Usage



Run the script:python barcode\_generator.py





Choose a QR code type (URL, text, or vCard) and provide the required information.

Customize the QR code appearance and specify a filename.

The QR code is saved as a PNG and may be displayed for preview.



Example interaction:

Welcome to Barcode Generator! Create QR codes for URLs, text, or vCards.



Choose QR code type:

1\. URL (e.g., website link)

2\. Text (e.g., a message)

3\. vCard (contact information)

4\. Exit

Enter your choice (1-4): 1

Enter URL (e.g., https://example.com): https://github.com/Ajafarnezhad

Enter filename for QR code (default: url\_qr.png): my\_github.png

Enter QR code color (default: black): blue

Enter background color (default: white): white

Enter box size (default: 10): 10

Enter border size (default: 4): 4

QR code saved as my\_github.png



How It Works



Uses the qrcode library to generate QR codes and PIL (Pillow) to save them as PNG files.

Validates URLs using a regular expression to ensure proper formatting.

Creates vCard strings for contact information in a standard format.

Supports customization of QR code appearance (colors, size, border).

Handles errors for invalid inputs, file-saving issues, or system limitations.



Improvements Ideas



Add support for other barcode formats (e.g., Code 128, UPC).

Create a GUI using Tkinter or PyQt for a more interactive experience.

Allow batch generation of multiple QR codes from a file.



Notes



Ensure you have write permissions in the directory where the QR code is saved.

Test the generated QR code with a QR scanner to verify functionality.



This project is part of my portfolio. Check out my other projects on GitHub: Ajafarnezhad

License: MIT (Free to use and modify)


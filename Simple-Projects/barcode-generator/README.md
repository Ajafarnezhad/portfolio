# Barcode Generator: Create Custom QR Codes with Ease 📱✨

Welcome to the **Barcode Generator**, a versatile Python script that empowers you to create QR codes for URLs, plain text, or vCard contact information with a user-friendly interface. This project allows you to customize the QR code's appearance, validate inputs, and save the output as a PNG file. With robust error handling and optional preview functionality, it’s a perfect addition to your portfolio, showcasing your skills in Python development and practical application design.

---

## 🌟 Project Highlights
This project combines simplicity with flexibility, enabling users to generate professional-grade QR codes with customizable aesthetics. Featuring input validation and error handling, it’s ideal for demonstrating clean coding practices and user-focused design.

---

## 🚀 Features
- **Versatile QR Code Generation**: Create QR codes for URLs, plain text, or vCard contact details (e.g., name, phone, email).
- **Customizable Appearance**: Adjust QR code color, background color, box size, and border for a personalized look.
- **Input Validation**: Ensures valid URL formats, required vCard fields, and proper input data.
- **PNG Output**: Saves QR codes as high-quality PNG files with user-defined filenames.
- **Preview Option**: Displays a preview of the generated QR code (if supported by your system).
- **Error Handling**: Robust checks for invalid inputs and file-saving issues, with clear user feedback.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `qrcode`
  - `Pillow`

Install dependencies with:
```bash
pip install qrcode pillow
```

---

## 🎮 How to Install
1. Ensure Python 3.8+ is installed.
2. Install required libraries:
   ```bash
   pip install qrcode pillow
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   ```
4. Navigate to the project directory:
   ```bash
   cd portfolio/Simple-Projects/barcode-generator
   ```

---

## 🎯 How to Run
1. Run the script:
   ```bash
   python barcode_generator.py
   ```
2. Follow the prompts to:
   - Choose a QR code type (URL, text, or vCard).
   - Provide the required information (e.g., URL, text, or contact details).
   - Customize appearance (color, background, box size, border).
   - Specify a filename for the output PNG.
3. The QR code is saved as a PNG file and may be displayed for preview (system-dependent).

---

## 📈 Example Interaction
```
Welcome to Barcode Generator!
Choose QR code type (1: URL, 2: Text, 3: vCard): 1
Enter URL: https://example.com
Enter fill color (e.g., black): blue
Enter background color (e.g., white): white
Enter box size (default 10): 10
Enter border size (default 4): 4
Enter output filename (e.g., qr_code.png): my_url_qr.png
QR code generated and saved as my_url_qr.png
[Preview displayed if supported]
```

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Additional QR Code Types**: Support Wi-Fi credentials, calendar events, or geolocation data.
- **Batch Generation**: Generate multiple QR codes from a CSV file.
- **Web App Deployment**: Create a Flask or Streamlit app for a browser-based interface.
- **Advanced Customization**: Add gradient fills or logo embedding in QR codes.
- **Unit Testing**: Implement `pytest` for robust validation of input handling and QR code generation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Create stunning QR codes with the **Barcode Generator** and bring your ideas to life! 🚀
# utils/report_generator.py - Report generation
# Integrates OWASP reporting best practices

from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(results, target, scan_type):
    filename = f"reports/{scan_type}_{datetime.now().strftime('%Y%m%d')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, f"Scan Report for {target}")
    y = 700
    for result in results:
        c.drawString(100, y, result)
        y -= 20
    c.save()
    return filename
# utils/validators.py - Input validation (OWASP ASVS 1.2.5 compliant)
# Prevents injection via sanitization

import re

def sanitize_input(input_str):
    # Remove potentially malicious characters
    sanitized = re.sub(r"[;\|&`]", "", input_str)
    return sanitized

def validate_ip(target):
    pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    if re.match(pattern, target):
        return True
    return False
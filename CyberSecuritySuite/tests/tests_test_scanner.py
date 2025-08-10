# tests/test_scanner.py - Unit tests with pytest
# Achieves high coverage

import pytest
from utils.validators import validate_ip, sanitize_input

def test_validate_ip():
    assert validate_ip("192.168.1.1") is True
    assert validate_ip("invalid") is False

def test_sanitize_input():
    assert sanitize_input("input; malicious|") == "input malicious"
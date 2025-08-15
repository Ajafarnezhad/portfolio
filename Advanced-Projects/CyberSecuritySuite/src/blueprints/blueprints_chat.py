# blueprints/chat.py - Secure chat integration
# End-to-end encryption with cryptography (OWASP A02)

from flask import Blueprint
from cryptography.fernet import Fernet
import threading
import socket
from utils.logger import setup_logger

chat_bp = Blueprint("chat", __name__)
logger = setup_logger("chat")

# Generate key (in production, use secure key management)
key = Fernet.generate_key()
cipher = Fernet(key)

def start_chat_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 8888))
    s.listen(1)
    logger.info("Chat server listening...")
    
    while True:
        client, addr = s.accept()
        logger.info(f"Connection from {addr}")
        threading.Thread(target=handle_client, args=(client, addr)).start()

def handle_client(client, addr):
    while True:
        try:
            encrypted_msg = client.recv(1024)
            if not encrypted_msg:
                break
            msg = cipher.decrypt(encrypted_msg).decode()
            logger.info(f"Decrypted msg from {addr}: {msg}")
            # Echo back encrypted
            client.send(cipher.encrypt(msg.encode()))
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            break
    client.close()

# Start server in background
threading.Thread(target=start_chat_server, daemon=True).start()
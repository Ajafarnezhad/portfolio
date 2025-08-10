import os
import base64
import secrets
from typing import Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken
from argon2 import PasswordHasher, low_level
from argon2.exceptions import HashingError
from zxcvbn import zxcvbn
from .utils import get_logger

logger = get_logger(__name__)

class EncryptionCore:
    """Core encryption/decryption logic with OWASP-compliant Argon2id KDF."""

    def __init__(self, time_cost: int = 2, memory_cost: int = 102400, parallelism: int = 8):
        self.ph = PasswordHasher(time_cost=time_cost, memory_cost=memory_cost, parallelism=parallelism)
        self.chunk_size = 1024 * 1024  # 1MB chunks for large files

    def validate_password(self, password: str, min_score: int = 3) -> bool:
        """Validate password strength using zxcvbn."""
        result = zxcvbn(password)
        if result['score'] < min_score:
            logger.warning(f"Weak password (score: {result['score']}). Suggestions: {result['feedback']['suggestions']}")
            return False
        return True

    @staticmethod
    def generate_salt(size: int = 16) -> bytes:
        return secrets.token_bytes(size)

    def load_salt(self, filename: str) -> bytes:
        salt_file = filename.replace(".envs", ".salt")
        if not os.path.exists(salt_file):
            raise FileNotFoundError(f"Salt file not found: {salt_file}")
        with open(salt_file, "rb") as f:
            return f.read()

    def derive_key(self, password: str, salt: bytes) -> bytes:
        try:
            key = low_level.hash_secret(
                password.encode(),
                salt,
                time_cost=self.ph.time_cost,
                memory_cost=self.ph.memory_cost,
                parallelism=self.ph.parallelism,
                hash_len=32,
                type=low_level.Type.ID
            )
            return base64.urlsafe_b64encode(key)
        except HashingError as e:
            logger.error(f"Key derivation failed: {e}")
            raise

    def generate_key(self, password: str, filename: str, load_existing_salt: bool = False, save_salt: bool = False) -> bytes:
        if load_existing_salt:
            salt = self.load_salt(filename)
        else:
            salt = self.generate_salt()
            if save_salt:
                with open(f"{filename}.salt", "wb") as f:
                    f.write(salt)
                logger.info(f"Salt saved to {filename}.salt")
        return self.derive_key(password, salt)

    def encrypt(self, filename: str, key: bytes, audit_log: bool = False) -> str:
        fernet = Fernet(key)
        encrypted_file = f"{filename}.envs"
        try:
            with open(filename, "rb") as infile, open(encrypted_file, "wb") as outfile:
                while chunk := infile.read(self.chunk_size):
                    encrypted_chunk = fernet.encrypt(chunk)
                    outfile.write(encrypted_chunk)
            if audit_log:
                logger.info(f"Encrypted {filename} to {encrypted_file}")
            return "File encrypted successfully."
        except Exception as e:
            logger.error(f"Encryption failed for {filename}: {e}")
            raise

    def decrypt(self, filename: str, key: bytes, audit_log: bool = False) -> str:
        fernet = Fernet(key)
        decrypted_file = filename.replace(".envs", "")
        try:
            with open(filename, "rb") as infile, open(decrypted_file, "wb") as outfile:
                while chunk := infile.read(self.chunk_size + 64):  # Account for padding
                    decrypted_chunk = fernet.decrypt(chunk)
                    outfile.write(decrypted_chunk)
            os.remove(filename)  # Clean up encrypted file
            salt_file = filename.replace(".envs", ".salt")
            if os.path.exists(salt_file):
                os.remove(salt_file)
            if audit_log:
                logger.info(f"Decrypted {filename} to {decrypted_file}")
            return "File decrypted successfully."
        except InvalidToken:
            logger.error("Invalid key or corrupted file.")
            raise
        except Exception as e:
            logger.error(f"Decryption failed for {filename}: {e}")
            raise
"""
Generate a secure random session secret for the backend.
Run this script and copy the output to your .env file.
"""
import secrets

# Generate a secure random string
session_secret = secrets.token_hex(32)

print("=" * 60)
print("SESSION SECRET GENERATOR")
print("=" * 60)
print("\nYour new session secret:")
print(f"\nSESSION_SECRET={session_secret}")
print("\nCopy the line above and add it to your backend/.env file")
print("=" * 60)

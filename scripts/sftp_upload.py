"""Upload a synthetic scanned page to the SFTP drop folder, as the scanner vendor would.

Usage: python scripts/sftp_upload.py --host sftp --port 22 [--name page.tif]
The SFTP credential is read from the SFTP_SECRET environment variable.
"""

import argparse
import io
import os
import time

import paramiko
from PIL import Image, ImageDraw


def sample_page() -> bytes:
    img = Image.new("L", (850, 1100), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((60, 60, 790, 140), fill=40)  # header band
    for y in range(200, 1000, 36):
        d.rectangle((60, y, 60 + (y * 7) % 600 + 120, y + 14), fill=90)  # text lines
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="tiff_lzw")
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="sftp")
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user", default="scanner")
    p.add_argument("--name", default=f"smoke_{int(time.time())}.tif")
    args = p.parse_args()

    transport = paramiko.Transport((args.host, args.port))
    transport.connect(username=args.user, password=os.environ["SFTP_SECRET"])
    sftp = paramiko.SFTPClient.from_transport(transport)
    with sftp.open(f"upload/{args.name}", "wb") as fh:
        fh.write(sample_page())
    transport.close()
    print(f"uploaded upload/{args.name}")


if __name__ == "__main__":
    main()

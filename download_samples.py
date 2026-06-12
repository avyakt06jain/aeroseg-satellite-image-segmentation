#!/usr/bin/env python3
"""
download_samples.py — Download placeholder sample images for AeroSeg demo.
"""
import urllib.request
from pathlib import Path

def main():
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Satellite_image_of_Dubai.jpg/512px-Satellite_image_of_Dubai.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Satellite_image_of_Venice.jpg/512px-Satellite_image_of_Venice.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Satellite_image_of_Cape_Town.jpg/512px-Satellite_image_of_Cape_Town.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Satellite_image_of_Singapore.jpg/512px-Satellite_image_of_Singapore.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Satellite_image_of_Sydney.jpg/512px-Satellite_image_of_Sydney.jpg"
    ]

    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls, 1):
        out_path = out_dir / f"sample_{i}.jpg"
        print(f"Downloading {url} to {out_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            # fallback to picsum
            fallback_url = f"https://picsum.photos/seed/aeroseg{i}/512/512"
            print(f"Using fallback URL: {fallback_url}")
            try:
                req = urllib.request.Request(fallback_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e2:
                print(f"Fallback also failed: {e2}")

    print("Done downloading samples.")

if __name__ == "__main__":
    main()

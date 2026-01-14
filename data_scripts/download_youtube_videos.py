import os
import time
import argparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from pytubefix import YouTube


def scrape_video_links(url, max_videos=None, scroll_pause=2.0):
    """
    Use Selenium to collect up to `max_videos` YouTube video URLs
    from a channel/playlist/search page.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    links = []
    seen = set()

    last_height = 0

    while True:
        # Grab all video elements currently loaded
        elems = driver.find_elements("css selector", 'a[href^="/watch?v="]')

        for e in elems:
            href = e.get_attribute("href")
            if not href:
                continue

            video_url = href.split("&")[0]  # strip extra params
            if video_url not in seen:
                seen.add(video_url)
                links.append(video_url)

                # Stop early if we have enough
                if max_videos is not None and len(links) >= max_videos:
                    driver.quit()
                    return links

        # Scroll further down
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            # No more content to load
            break
        last_height = new_height

        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(scroll_pause)

    driver.quit()
    return links

def download_videos(video_urls, out_dir):
    """
    Download each URL in `video_urls` to `out_dir` using pytubefix.
    """
    os.makedirs(out_dir, exist_ok=True)

    total = len(video_urls)
    for i, url in enumerate(video_urls, start=1):
        try:
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            print(f"[{i}/{total}] Downloading: {yt.title!r}")
            stream.download(output_path=out_dir)
        except Exception as e:
            print(f"[{i}/{total}] Failed to download {url}: {e}")

def download_first_n_videos(url, num_videos, out_dir):
    """
    High-level helper: scrape first `num_videos` from `url` and download them.
    """
    print(f"Scraping up to {num_videos} videos from:\n  {url}")
    links = scrape_video_links(url, max_videos=num_videos)
    print(f"Found {len(links)} video links. Starting downloads to: {out_dir}")
    download_videos(links, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.youtube.com/@COLORSxSTUDIOS/videos",
        help="YouTube page URL (channel / playlist / search results).",
    )
    parser.add_argument(
        "--num_videos", "-n",
        type=int,
        default=5,
        help="How many videos to download.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="downloads",
        help="Directory to save downloaded videos.",
    )

    args = parser.parse_args()

    download_first_n_videos(
        url=args.url,
        num_videos=args.num_videos,
        out_dir=args.out_dir,
    )

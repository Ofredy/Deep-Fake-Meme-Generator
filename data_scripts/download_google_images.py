import argparse
from icrawler.builtin import GoogleImageCrawler


def download_images(query, max_num=500000, out_dir="images"):
    """
    Download images from Google Images.

    Parameters
    ----------
    query : str
        The search term (e.g. "rockets").
    max_num : int
        Number of images to download.
    out_dir : str
        Directory where images will be saved.
    """
    crawler = GoogleImageCrawler(storage={"root_dir": out_dir})
    crawler.crawl(keyword=query, max_num=max_num)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download images from Google Images")
    parser.add_argument("-search", type=str, help="Search term for images")
    parser.add_argument(
        "-n", "--num", type=int, default=10, help="Number of images to download (default: 50)"
    )
    parser.add_argument(
        "-o", "--out", type=str, default=None, help="Output folder (default: <query>_images)"
    )

    args = parser.parse_args()
    folder_name = args.out if args.out else args.search.replace(" ", "_") + "_images"

    download_images(args.search, max_num=args.num, out_dir=folder_name)
    print(f"Downloaded {args.num} images for '{args.search}' into folder '{folder_name}/'")

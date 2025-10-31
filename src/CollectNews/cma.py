import time
import json
import re
import argparse
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DEFAULT_DELAY = 1.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CMA-News-Scraper/1.1; +https://example.com)"
}

# -------------------- 基础工具 -------------------- #
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "html.parser")

def clean_text(s: str) -> str:
    return re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", s)).strip()

# -------------------- 列表页解析 -------------------- #
def find_pagination_links(soup: BeautifulSoup, base_url: str):
    """在底部分页区 .con_pages 中提取所有页码链接（转绝对URL）"""
    links = []
    pag = soup.select_one(".con_pages")
    if not pag:
        return links
    for a in pag.find_all("a", href=True):
        href = a["href"].strip()
        if href:
            links.append(urljoin(base_url, href))
    return links

def find_article_links_on_listing(soup: BeautifulSoup, base_url: str):
    """在列表页中找到新闻详情链接（转绝对URL）"""
    urls = []
    for a in soup.select(".eventList .eventItem .eventC a[href]"):
        href = a["href"].strip()
        if not href:
            continue
        urls.append(urljoin(base_url, href))
    return urls

# -------------------- 详情页解析 -------------------- #
def parse_article(url: str, delay: float):
    soup = get_soup(url)
    time.sleep(delay)

    # 标题
    title = ""
    t = soup.select_one(".titleText")
    if t:
        title = clean_text(t.get_text(" ", strip=True))
    if not title:
        m = soup.find("meta", attrs={"name": "ArticleTitle"})
        if m and m.get("content"):
            title = clean_text(m["content"])
        if not title and soup.title:
            title = clean_text(soup.title.get_text(strip=True))

    # 日期、来源（位于 .reportBannerBox .reportBanner p）
    updated, source = "", ""
    for p in soup.select(".reportBannerBox .reportBanner p"):
        text = p.get_text(" ", strip=True)
        if "Updated" in text:
            spans = p.find_all("span")
            updated = spans[-1].get_text(strip=True) if spans else text.split(":", 1)[-1].strip()
        elif "Source" in text:
            spans = p.find_all("span")
            source = spans[-1].get_text(strip=True) if spans else text.split(":", 1)[-1].strip()

    # 正文（.reportText）
    content_text = ""
    node = soup.select_one(".reportText")
    if node:
        for tag in node.select("script, style"):
            tag.decompose()
        content_text = clean_text(node.get_text("\n", strip=True))

    return {
        "url": url,
        "title": title,
        "updated": updated,
        "source": source,
        "content_text": content_text,
    }

# -------------------- 主流程 -------------------- #
def read_start_urls(links_file: str):
    """从文件读取列表页URL（每行一个），忽略空行与#注释"""
    urls = []
    with open(links_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    return urls

def crawl_list_page(start_url: str, delay: float):
    """遍历单个入口列表页的所有分页，收集文章URL"""
    visited_listing = set()
    to_visit_listing = [start_url]
    article_urls = set()

    while to_visit_listing:
        listing_url = to_visit_listing.pop(0)
        if listing_url in visited_listing:
            continue
        visited_listing.add(listing_url)

        try:
            soup = get_soup(listing_url)
        except Exception as e:
            print(f"[WARN] Failed listing: {listing_url} -> {e}")
            continue
        time.sleep(delay)

        found_articles = find_article_links_on_listing(soup, listing_url)
        article_urls.update(found_articles)

        pag_links = find_pagination_links(soup, listing_url)
        for purl in pag_links:
            if purl not in visited_listing and purl not in to_visit_listing:
                to_visit_listing.append(purl)

        print(f"[INFO] LIST {listing_url} -> {len(found_articles)} articles, {len(pag_links)} page links")

    return article_urls

def crawl_from_links_file(links_file: str, output_json: str, delay: float):
    start_urls = read_start_urls(links_file)
    if not start_urls:
        print(f"[ERR] No URLs found in {links_file}")
        return

    # 先汇总所有文章URL（跨多个入口列表页）
    all_article_urls = set()
    for url in start_urls:
        print(f"[INFO] Start listing: {url}")
        urls = crawl_list_page(url, delay=delay)
        all_article_urls.update(urls)

    print(f"[INFO] Total article URLs collected: {len(all_article_urls)}")

    # 逐篇解析
    results = []
    for idx, url in enumerate(sorted(all_article_urls)):
        try:
            item = parse_article(url, delay=delay)
            results.append(item)
            print(f"[OK] ({idx+1}/{len(all_article_urls)}) {url}")
        except Exception as e:
            print(f"[ERR] ({idx+1}/{len(all_article_urls)}) {url} -> {e}")

    # 保存为 JSON（数组）
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(results)} items to {output_json}")

# -------------------- CLI -------------------- #
def main():
    parser = argparse.ArgumentParser(description="CMA news crawler (multi listing pages)")
    parser.add_argument("--links_file", type=str, required=True,
                        help="文本文件：每行一个新闻列表页URL（例如 https://www.cma.gov.cn/en/news/NewsEvents/news/）")
    parser.add_argument("--output", type=str, default="cma_news.json",
                        help="输出JSON文件名（默认 cma_news.json）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"请求间隔秒，默认 {DEFAULT_DELAY}")
    args = parser.parse_args()

    Path(args.links_file).expanduser().resolve()
    crawl_from_links_file(args.links_file, args.output, args.delay)

if __name__ == "__main__":
    main()

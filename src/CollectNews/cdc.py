# cdc_releases_crawler.py
# -*- coding: utf-8 -*-
import json, time, re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE = "https://www.cdc.gov"
LIST_JSON = BASE + "/media/collections/media-releases.json"
LIST_STATIC = BASE + "/media/collections/media-releases.static.json"
SITEMAP_INDEX = BASE + "/wcms-auto-sitemap-index.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CDCReleasesCrawler/1.0)",
    "Accept": "application/json, text/javascript, */*; q=0.1",
}

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_urls_from_collection_json(data):
    urls = []
    pools = []
    if isinstance(data, dict):
        for k in ("items", "results", "posts", "data"):
            if k in data and isinstance(data[k], list):
                pools.append(data[k])
        if "collection" in data and isinstance(data["collection"], dict):
            it = data["collection"].get("items")
            if isinstance(it, list):
                pools.append(it)
    elif isinstance(data, list):
        pools.append(data)

    for pool in pools:
        for it in pool:
            if isinstance(it, dict):
                u = it.get("url") or it.get("link") or it.get("permalink") or it.get("path")
                if u:
                    if u.startswith("/"):
                        u = urljoin(BASE, u)
                    urls.append(u)
    return urls

def try_paged_widget(session, delay, max_pages):
    """模拟前端翻页：?page=N 逐页取 JSON"""
    all_urls = []
    # 常见的分页参数兜底尝试
    patterns = ["?page=%d", "&page=%d", "?p=%d", "&p=%d", "?pageNumber=%d", "&pageNumber=%d"]
    page = 1
    empty_streak = 0

    while page <= max_pages and empty_streak < 2:
        got_any = False
        for pat in patterns:
            url = LIST_JSON + (pat % page if ("?" not in LIST_JSON or pat.startswith("?")) else (pat.replace("?", "&") % page))
            try:
                r = session.get(url, headers=HEADERS, timeout=20)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            ctype = r.headers.get("Content-Type", "")
            if "json" not in ctype:
                continue
            try:
                data = r.json()
            except ValueError:
                continue
            urls = extract_urls_from_collection_json(data)
            if urls:
                got_any = True
                all_urls.extend(urls)
                break  # 这一页已拿到，换下一页
        if got_any:
            page += 1
            empty_streak = 0
            time.sleep(delay)
        else:
            empty_streak += 1
            page += 1  # 有些页可能空，继续再探一页
            time.sleep(delay)

    return unique(all_urls)

def try_static_collection(session):
    try:
        r = session.get(LIST_STATIC, headers=HEADERS, timeout=20)
        if r.status_code == 200 and "json" in r.headers.get("Content-Type", ""):
            return unique(extract_urls_from_collection_json(r.json()))
    except Exception:
        pass
    return []

def from_sitemap(session):
    """最后兜底：解析 sitemap，把 /media/releases/ 下的 URL 都找出来"""
    urls = []
    try:
        r = session.get(SITEMAP_INDEX, headers=HEADERS, timeout=25)
        r.raise_for_status()
    except Exception:
        return urls

    # 粗糙但稳：不用 XML 库，直接找 <loc>…</loc>
    locs = re.findall(r"<loc>(.*?)</loc>", r.text)
    for sm in locs:
        if not sm.endswith(".xml"):
            continue
        try:
            s = session.get(sm, headers=HEADERS, timeout=25)
        except Exception:
            continue
        if s.status_code != 200:
            continue
        for u in re.findall(r"<loc>(.*?)</loc>", s.text):
            if "/media/releases/" in u and u.endswith(".html"):
                urls.append(u)
        time.sleep(0.2)
    return unique(urls)

def parse_article(session, url, delay):
    try:
        r = session.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=25)
        if r.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    # 标题
    h1 = soup.select_one(".cdc-page-title h1") or soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # 日期：优先 meta DC.date，再试 .date-long 文本
    date_iso = ""
    m = soup.find("meta", attrs={"name": "DC.date"})
    if m and m.get("content"):
        # 形如 2025-08-11T17:01:10Z -> 2025-08-11
        date_iso = m["content"][:10]
    else:
        dl = soup.select_one(".date-long")
        if dl:
            raw = dl.get_text(" ", strip=True)
            # August 11, 2025 -> 2025-08-11
            try:
                date_iso = datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")
            except Exception:
                date_iso = raw

    # 正文：data-section="cdc_news_body"
    body = soup.select_one('[data-section="cdc_news_body"]') or soup.find(id="content")
    paras = []
    if body:
        for node in body.find_all(["p", "li"]):
            t = node.get_text(" ", strip=True)
            if t:
                paras.append(t)
    content = "\n".join(paras)

    time.sleep(delay)
    return {
        "source": "CDC Newsroom Releases",
        "url": url,
        "title": title,
        "date": date_iso,
        "content": content
    }

def crawl(out_path="cdc_releases.json", delay=0.5, max_pages=200, limit=None):
    session = requests.Session()
    session.headers.update(HEADERS)
    print("Starting crawl...")

    # 1) 尝试“翻页”的 JSON 小部件
    urls = try_paged_widget(session, delay, max_pages)
    print("Found {} article URLs from paged widget.".format(len(urls)))

    # 2) 回退到 static JSON（不分页）
    if not urls:
        urls = try_static_collection(session)
        print("Found {} article URLs from static collection.".format(len(urls)))

    # 3) 最后兜底：Sitemap 全量
    if not urls:
        urls = from_sitemap(session)

    urls = [u if u.startswith("http") else urljoin(BASE, u) for u in urls]
    urls = unique(urls)
    print("Found {} unique article URLs.".format(len(urls)))
    if limit:
        urls = urls[:int(limit)]

    items = []
    for i, u in enumerate(urls, 1):
        print("doing [{}/{}] {}".format(i, len(urls), u))
        art = parse_article(session, u, delay)
        if art and art.get("title"):
            items.append(art)
        print("[{}/{}] {}".format(i, len(urls), "OK" if art else "SKIP"), u)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print("Saved {} items to {}".format(len(items), out_path))

if __name__ == "__main__":
    # 按需修改输出路径、延迟、最大页数
    crawl(out_path="src/CollectNews/cdc_news.json", delay=0.8, max_pages=400)

"""Download the Kalam and JetBrains Mono latin subsets (OFL) used by the diagrams into fonts/."""

import re
import urllib.request
from pathlib import Path

CSS = "https://fonts.googleapis.com/css2?family=Kalam:wght@400;700&family=JetBrains+Mono:wght@500"
UA = "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120 Safari/537.36"

req = urllib.request.Request(CSS, headers={"User-Agent": UA})
css = urllib.request.urlopen(req).read().decode()
Path("fonts").mkdir(exist_ok=True)


def field(pattern: str, block: str) -> str:
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"{pattern!r} not found in the Google Fonts CSS")
    return match.group(1)


for blk in re.findall(r"/\* latin \*/\s*@font-face\s*\{(.*?)\}", css, re.S):
    fam = field(r"font-family: '([^']+)'", blk).replace(" ", "")
    weight = field(r"font-weight: (\d+)", blk)
    src = field(r"url\((.*?)\)", blk)
    out = Path("fonts") / f"{fam}-{weight}.woff2"
    urllib.request.urlretrieve(src, out)
    print(out)

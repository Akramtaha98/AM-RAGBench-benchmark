"""
Step 4b: extract paragraph-level passages directly from a Wikipedia XML dump
(the .xml.bz2 file you downloaded from dumps.wikimedia.org), with no external
dependencies beyond the Python standard library.

An earlier version of this script depended on the third-party `wikiextractor`
package, which crashes on current Python versions (3.11+) with a regex
compatibility error ("global flags not at the start of the expression") --
that's a bug in wikiextractor itself, not in your data or setup. This version
parses the MediaWiki export XML and strips wikitext markup directly, so there
is nothing extra to install.

Usage:
    python extract_wikipedia.py --lang ar --dump arwiki-latest-pages-articles-multistream.xml.bz2 \
        --out ar_wiki_passages.jsonl --max-articles 2000
    python extract_wikipedia.py --lang ms --dump mswiki-latest-pages-articles-multistream.xml.bz2 \
        --out ms_wiki_passages.jsonl --max-articles 2000

`--max-articles` caps how many articles to pull passages from -- the full
dump has millions of articles; a few thousand is far more than enough
passages for a benchmark, and capping keeps this step fast.
"""

import argparse
import bz2
import json
import re
import xml.etree.ElementTree as ET


# --- wikitext -> plain text cleaning -------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_HEADER_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_FILE_LINK_RE = re.compile(
    r"\[\[\s*(?:File|Image|Fail|Berkas|Imej)\s*:.*?\]\]", re.DOTALL | re.IGNORECASE
)
_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")
_EXTLINK_RE = re.compile(r"\[(?:https?|ftp)://\S+\s+([^\]]+)\]")
_BARE_EXTLINK_RE = re.compile(r"\[(?:https?|ftp)://\S+\]")


def _strip_templates(text: str) -> str:
    """Remove {{ ... }} templates, including nested ones, by repeatedly
    collapsing the innermost balanced pair until none remain."""
    pattern = re.compile(r"\{\{[^{}]*\}\}")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub("", text)
    return text


def wikitext_to_plaintext(text: str) -> str:
    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = _strip_templates(text)
    text = _FILE_LINK_RE.sub("", text)
    text = _EXTLINK_RE.sub(r"\1", text)
    text = _BARE_EXTLINK_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _HEADER_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    # collapse excess blank lines but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_REDIRECT_RE = re.compile(r"^\s*#(REDIRECT|redirect|ALIH|alih|تحويل)", re.IGNORECASE)


def paragraphs_from_plaintext(text: str, min_words: int = 25, max_words: int = 120):
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        word_count = len(para.split())
        if min_words <= word_count <= max_words:
            yield para


# --- streaming dump parsing ------------------------------------------------

def iter_articles(dump_path: str):
    """
    Yields (page_id, title, wikitext) for main-namespace (ns=0), non-redirect
    pages, streaming the bz2 XML so the whole dump never has to fit in memory.
    Namespace-agnostic: matches on tag suffix so it works across MediaWiki
    export schema versions (0.10, 0.11, etc.).
    """
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        page_id, title, ns, wikitext = None, None, None, None
        for _, elem in context:
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = elem.text
            elif tag == "id" and page_id is None:
                page_id = elem.text
            elif tag == "text":
                wikitext = elem.text or ""
            elif tag == "page":
                if ns == "0" and wikitext and not _REDIRECT_RE.match(wikitext):
                    yield page_id, title, wikitext
                page_id, title, ns, wikitext = None, None, None, None
                elem.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "ms"])
    ap.add_argument("--dump", required=True, help="path to the .xml.bz2 dump file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-articles", type=int, default=2000)
    ap.add_argument("--max-passages-per-article", type=int, default=3)
    args = ap.parse_args()

    n_articles = 0
    n_passages = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for page_id, title, wikitext in iter_articles(args.dump):
            if n_articles >= args.max_articles:
                break
            plain = wikitext_to_plaintext(wikitext)
            kept = 0
            for para in paragraphs_from_plaintext(plain):
                if kept >= args.max_passages_per_article:
                    break
                record = {
                    "passage_id": f"{args.lang}-wiki-{page_id}-{kept}",
                    "language": args.lang,
                    "domain": "wikipedia_general",
                    "article_title": title,
                    "text": para,
                    "source_citation": (
                        f"https://{args.lang}.wikipedia.org/?curid={page_id} "
                        f"(Wikipedia, CC BY-SA, article id {page_id})"
                    ),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                n_passages += 1
            n_articles += 1
            if n_articles % 200 == 0:
                print(f"...processed {n_articles} articles, {n_passages} passages so far")

    print(f"\nProcessed {n_articles} articles, wrote {n_passages} passages to {args.out}")
    print("Note: this takes articles in dump order (not a random sample). Good enough for a "
          "benchmark; if you want a less biased sample later, shuffle before capping.")
    print("Next: feed the output into question generation (Step 5), same as quran_passages.jsonl.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the результаты-slides of the deck from a spec.

Keeps every figure slide in the same shape: heading at the top margin, the figure on a
white card, a short takeaway line, and a pinned footer. Writes one file per slide into
<deck>/project/slides/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DARK, LIGHT, ALT = "#16212E", "#FBFAF7", "#F1EEE8"
BLUE, ORANGE = "#2A78D6", "#EB6834"
MUTED, FAINT = "#4A5568", "#8B98A6"
SERIF = "'Source Serif 4', Georgia, serif"
SANS = "'IBM Plex Sans', Arial, sans-serif"


def figure_slide(sid: str, title: str, sub: str, img: str, alt: str, takeaway: str,
                 notes: str = "", img_h: int = 470, section: str | None = None) -> str:
    sec = f' data-section="{section}"' if section else ""
    return f'''<section id="{sid}"{sec} style="background:{LIGHT};color:{DARK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:28px">
<h2 style="font-family:{SERIF};font-size:56px;font-weight:600;line-height:1.1;width:1600px">{title}</h2>
<p style="font-size:28px;color:{MUTED};line-height:1.35;width:1600px">{sub}</p>
<img src="{img}" alt="{alt}" style="width:1664px;height:{img_h}px;object-fit:contain;background:#FFFFFF;border:1px solid #E3E0D8;border-radius:16px;padding:16px">
<p style="font-size:30px;color:{DARK};background:{ALT};padding:28px;border-radius:14px;line-height:1.35;width:1664px">{takeaway}</p>
<p style="position:absolute;left:128px;bottom:64px;width:1300px;font-size:24px;color:{FAINT}">Воспроизведение · CIFAR-10 / ResNet-18 · RTX 3090</p>
<aside>{notes}</aside>
</section>
'''


def table_slide(sid: str, title: str, sub: str, header: list[str], rows: list[list[str]],
                takeaway: str, notes: str = "", widths: list[int] | None = None,
                fs: int = 28) -> str:
    widths = widths or [round(100 / len(header))] * len(header)
    th = "".join(f'<th style="width:{w}%;text-align:left;font-weight:600">{h}</th>'
                 for h, w in zip(header, widths))
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'''<section id="{sid}" style="background:{LIGHT};color:{DARK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:36px">
<h2 style="font-family:{SERIF};font-size:56px;font-weight:600;line-height:1.1;width:1600px">{title}</h2>
<p style="font-size:28px;color:{MUTED};line-height:1.35;width:1600px">{sub}</p>
<table style="font-family:{SANS};font-size:{fs}px;color:{DARK};width:1664px">
<tr>{th}</tr>
{tr}
</table>
<p style="font-size:30px;color:{DARK};background:{ALT};padding:28px;border-radius:14px;line-height:1.35;width:1664px">{takeaway}</p>
<p style="position:absolute;left:128px;bottom:64px;width:1300px;font-size:24px;color:{FAINT}">Воспроизведение · CIFAR-10 / ResNet-18 · RTX 3090</p>
<aside>{notes}</aside>
</section>
'''


def cards_slide(sid: str, title: str, cards: list[tuple[str, str, str]], takeaway: str,
                notes: str = "", accent: str = ORANGE) -> str:
    inner = "".join(
        f'''<div style="flex:1;display:flex;flex-direction:column;gap:14px;background:#FFFFFF;padding:40px;border:1px solid #E3E0D8;border-radius:18px">
<h1 style="font-size:84px;font-weight:600;color:{accent};line-height:1">{big}</h1>
<h3 style="font-size:30px;font-weight:600;line-height:1.2">{head}</h3>
<p style="font-size:26px;color:{MUTED};line-height:1.4">{body}</p>
</div>''' for big, head, body in cards)
    return f'''<section id="{sid}" style="background:{LIGHT};color:{DARK};font-family:{SANS};padding:128px 128px 160px;display:flex;flex-direction:column;gap:48px">
<h2 style="font-family:{SERIF};font-size:56px;font-weight:600;line-height:1.1;width:1600px">{title}</h2>
<div style="display:flex;gap:28px">{inner}</div>
<p style="font-size:30px;color:{DARK};background:{ALT};padding:28px;border-radius:14px;line-height:1.35;width:1664px">{takeaway}</p>
<p style="position:absolute;left:128px;bottom:64px;width:1300px;font-size:24px;color:{FAINT}">Воспроизведение · CIFAR-10 / ResNet-18 · RTX 3090</p>
<aside>{notes}</aside>
</section>
'''


def write(deck_dir: str, slides: dict[str, str]):
    d = Path(deck_dir) / "project" / "slides"
    d.mkdir(parents=True, exist_ok=True)
    for sid, html in slides.items():
        (d / f"{sid}.html").write_text(html)
        print(f"  -> {d/sid}.html")


if __name__ == "__main__":
    print(__doc__)

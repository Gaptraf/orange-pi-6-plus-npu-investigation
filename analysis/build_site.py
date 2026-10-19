"""Build a dependency-free static page from this project's limited Markdown subset."""
from pathlib import Path
import html,re,urllib.parse
ROOT=Path(__file__).resolve().parents[1]
OWNER='Gaptraf'
REPOSITORY='orange-pi-6-plus-npu-investigation'
REPO=f'https://github.com/{OWNER}/{REPOSITORY}'
def inline(text):
    protected=[]
    def code(m):
        protected.append('<code>'+html.escape(m[1])+'</code>');return f'@@CODE{len(protected)-1}@@'
    text=re.sub(r'`([^`]+)`',code,text)
    text=html.escape(text,quote=False)
    def link(m):
        label,url=m[1],html.unescape(m[2])
        if not re.match(r'https?://|#',url):url=REPO+'/blob/main/'+url
        return '<a href="'+html.escape(url,quote=True)+'">'+label+'</a>'
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    for i,s in enumerate(protected):text=text.replace(f'@@CODE{i}@@',s)
    return text
def render(source):
    lines=source.splitlines();body=[];toc=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            body.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>');i+=1;continue
        h=re.match(r'^(#{1,3}) (.*)',line)
        if h:
            level=len(h[1]);label=h[2];ident=re.sub('[^a-z0-9]+','-',label.lower()).strip('-')
            body.append(f'<h{level} id="{ident}">{inline(label)}</h{level}>')
            if level==2:toc.append(f'<a href="#{ident}">{html.escape(label)}</a>')
            i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):table.append(lines[i]);i+=1
            rows=[[c.strip() for c in row.strip().strip('|').split('|')] for row in table]
            body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            for row in rows[2:]:body.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            body.append('</tbody></table></div>');continue
        if line.startswith('- '):
            body.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):body.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            body.append('</ul>');continue
        if line.startswith('> '):body.append('<blockquote><p>'+inline(line[2:])+'</p></blockquote>');i+=1;continue
        paragraph=[]
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||- |>|```)',lines[i]):paragraph.append(lines[i]);i+=1
        body.append('<p>'+inline(' '.join(paragraph))+'</p>')
    return '\n'.join(body),'\n'.join(toc)
def build():
    body,toc=render((ROOT/'ARTICLE.md').read_text(encoding='utf8'))
    page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An owner’s documented investigation of Orange Pi 6 Plus NPU software, correctness failures, runtime performance and the unresolved bandwidth bottleneck.">
<title>Orange Pi 6 Plus: 30 TOPS, but where is the usable LLM acceleration?</title>
<style>
:root{color-scheme:light;--ink:#19232b;--muted:#52616d;--paper:#faf9f6;--accent:#a74314;--line:#d8ddd9;--panel:#eeeee8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:2rem}body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.75 Georgia,'Times New Roman',serif}
a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}header{border-top:7px solid var(--accent);border-bottom:1px solid var(--line);padding:1rem max(5vw,1.2rem);display:flex;align-items:center;justify-content:space-between;gap:1rem;font:600 13px/1.5 system-ui,sans-serif;letter-spacing:.045em}header .brand{color:var(--ink);text-decoration:none}header nav{display:flex;gap:1.2rem}
.layout{max-width:1330px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:4rem;padding:3.5rem 3rem 6rem}.contents{position:sticky;top:2rem;align-self:start;font:13px/1.55 system-ui,sans-serif}.contents strong{display:block;margin-bottom:1rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}.contents a{display:block;color:var(--muted);text-decoration:none;margin:.75rem 0}.contents a:hover{color:var(--accent)}article{max-width:850px;min-width:0}h1,h2{font-family:system-ui,sans-serif;line-height:1.16;letter-spacing:-.035em}h1{font-size:clamp(2.1rem,4.1vw,3.5rem);margin:0 0:2rem}h2{font-size:1.7rem;margin:3.3rem 0 1.1rem;padding-top:1.3rem;border-top:1px solid var(--line)}p{margin:1.2rem 0}article>p:first-of-type{font:13px/1.6 system-ui,sans-serif;color:var(--muted);letter-spacing:.04em;margin-top:1.5rem}strong{font-weight:700}blockquote{margin:2rem 0;padding:1rem 1.4rem;border-left:4px solid var(--accent);background:var(--panel);font-size:1.2rem}blockquote p{margin:0}.table-wrap{overflow-x:auto;margin:1.8rem 0;border:1px solid var(--line);border-radius:4px}table{width:100%;border-collapse:collapse;font:13px/1.55 system-ui,sans-serif}th,td{padding:.8rem .9rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:var(--panel);font-weight:650}tr:last-child td{border-bottom:0}code{font: .82em/1.5 ui-monospace,Consolas,monospace;background:var(--panel);padding:.12em .25em;border-radius:3px;overflow-wrap:anywhere}pre{overflow:auto;padding:1rem;background:var(--panel)}ul{padding-left:1.4rem}li{margin:.65rem 0}footer{max-width:850px;margin:0 auto;padding:2rem;border-top:1px solid var(--line);font:13px/1.6 system-ui,sans-serif;color:var(--muted)}.skip{position:absolute;left:-10000px}.skip:focus{left:1rem;top:1rem;background:white;padding:.8rem}
@media(max-width:1000px){.layout{grid-template-columns:1fr;padding:2.4rem 1.4rem 4rem;gap:1rem;max-width:850px}.contents{position:static;border-bottom:1px solid var(--line);padding-bottom:1rem}.contents a{display:inline-block;margin:.2rem .85rem .2rem 0;font-size:12px}header{align-items:flex-start}header nav{flex-wrap:wrap;gap:.6rem}}
@media print{header,.contents,.skip{display:none}.layout{display:block;padding:0}body{background:white;font-size:11pt}article{max-width:none}h1{font-size:25pt}h2{font-size:16pt;break-after:avoid}.table-wrap{overflow:visible}a{color:inherit}footer{max-width:none}}
</style></head><body><a class="skip" href="#article">Skip to article</a>
<header><a class="brand" href="REPO">GAPTRAF / HARDWARE INVESTIGATION</a><nav><a href="REPO">Repository</a><a href="REPO/blob/main/RESULTS.md">Results</a><a href="REPO/blob/main/REPRODUCING.md">Verify</a></nav></header>
<div class="layout"><aside class="contents" aria-label="Article contents"><strong>Inside the investigation</strong>TOC</aside><main><article id="article">BODY</article></main></div>
<footer>Owner-led, Codex-assisted investigation. Measurements: 15–19 October 2026. One board, recorded software versions. <a href="REPO/blob/main/CORRECTIONS.md">Corrections</a> · <a href="REPO/blob/main/METHODS.md">Methods and limitations</a> · <a href="REPO/blob/main/SOURCES.md">Sources</a></footer>
</body></html>'''
    # Replace full sentinel words, not arbitrary occurrences in article content.
    page=page.replace('REPO',REPO).replace('TOC',toc).replace('BODY',body)
    page=page.replace('margin:0 0:2rem','margin:0 0 2rem')
    (ROOT/'docs/index.html').write_text(page,encoding='utf8')
    return dict(page='docs/index.html',article_sections=len(re.findall('<h2 ',body)),tables=body.count('<table>'))
if __name__=='__main__':
    import json
    print(json.dumps(build()))

"""Offline integrity, arithmetic, syntax, link and publication-hygiene validation."""
from pathlib import Path
import ast,hashlib,html,ipaddress,json,re,urllib.parse
from recompute import calculate
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'manifest.sha256.json').read_text(encoding='utf8'))
    for rel,expected in manifest['files'].items():
        p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT.resolve()),rel
        assert p.is_file() and digest(p)==expected,rel
    provenance=json.loads((ROOT/'evidence/PROVENANCE.json').read_text(encoding='utf8'))
    for item in provenance['files']:assert digest(ROOT/item['published'])==item['published_sha256'],item['published']
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p.relative_to(ROOT)))
    for p in ROOT.rglob('*.json'):json.loads(p.read_text(encoding='utf8'))
    checked_links=0
    for p in ROOT.glob('*.md'):
        text=p.read_text(encoding='utf8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if re.match(r'https?://|#',target):continue
            clean=urllib.parse.unquote(target.split('#')[0])
            assert (p.parent/clean).exists(),(p.name,target)
            checked_links+=1
    html_text=(ROOT/'docs/index.html').read_text(encoding='utf8')
    ids=set(re.findall(r'\bid="([^"]+)"',html_text))
    for target in re.findall(r'href="#([^"]+)"',html_text):assert target in ids,target
    assert html_text.count('<table>')==html_text.count('</table>')==5
    assert '<script' not in html_text.lower()
    bad=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix=='.pyc':continue
        if '__pycache__' in p.parts:continue
        assert p.suffix.lower() not in ['.pub','.key','.pem','.onnx','.safetensors','.gguf','.whl','.ko','.so','.bin'],p.name
        text=p.read_text(encoding='utf8')
        if p.resolve()==Path(__file__).resolve():continue  # The scanner contains its own detection patterns.
        # Generic patterns must not embed an owner's real identifiers.
        patterns=[r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                  r'(?i)[a-z]:[\\/]+(?:Users|codex projects)[\\/]+',
                  r'/home/(?!researcher/)[^/\s"\x27<>]+/',
                  r'\b(?:ssh-(?:ed25519|rsa|dss)|ecdsa-sha2-\S+)\s+[A-Za-z0-9+/]{32,}',
                  r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b',
                  r'ssh://[^\s]+@',r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b']
        for pattern in patterns:
            if re.search(pattern,text):bad.append((p.relative_to(ROOT).as_posix(),'restricted pattern'))
        for value in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text):
            try:ipaddress.ip_address(value)
            except ValueError:continue
            bad.append((p.relative_to(ROOT).as_posix(),'network address; review required'))
    assert not bad,bad
    metrics=calculate()
    print(json.dumps(dict(status='passed',manifest_files=len(manifest['files']),
                          provenance_files=len(provenance['files']),local_markdown_links=checked_links,
                          html_tables=html_text.count('<table>'),calculation_checks=True,board_access=False)))
if __name__=='__main__':main()

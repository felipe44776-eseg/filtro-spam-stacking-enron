"""De um e-mail qualquer (bruto, com cabeçalhos, ou texto colado) para o formato pré-processado da AUEB,
e daí para o vetor de 3000 contagens. É a mesma conversão usada na demonstração e na 3ª base (SpamAssassin).

Regras, deduzidas comparando os 3669 e-mails brutos da caixa farmer-d com a versão pré-processada do Enron1.
Resultado: 90,5% dos vetores saem idênticos a uma linha do CSV; nos demais, a diferença mediana é de 1% das
contagens (casos raros, como um domínio de e-mail que a AUEB apagou sem regra aparente).
1. Do cabeçalho, só o assunto fica: primeira linha `Subject: <assunto>`.
2. Do corpo, as partes de texto (text/plain; se só houver HTML, ele é convertido em texto). Tudo o que está entre
   `<` e `>` é apagado, também no texto puro: endereços `<x@y.com>`, `<mailto:...>` e parte de `<< File: a.xls >>`.
3. Tudo em minúsculas; sequências de letras, sequências de dígitos e cada sinal de pontuação viram unidades
   separadas por espaço ("2:00 p.m." -> "2 : 00 p . m ."; "x3-6992" -> "x 3 - 6992").
4. Linhas vazias são descartadas.
"""
from __future__ import annotations

import email
import html
import re
from email import policy
from email.header import decode_header, make_header

import numpy as np

# palavra = letra seguida de letras ou dos dígitos 0 e 1, lidos como "o" e "l" (v1agra -> vlagra; reveff061200 -> reveffo 61200)
_TOKEN = re.compile(r"[a-z][a-z01]*|[0-9]+|[^\sa-z0-9]")
_DIGITO_LETRA = str.maketrans({"0": "o", "1": "l"})
_TAG = re.compile(r"<[^>]*>", re.S)
_ENTIDADE = re.compile(r"&[^;]*;", re.S)      # a AUEB apaga de "&" até o próximo ";", mesmo atravessando linhas
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def tokenizar_linha(linha: str) -> str:
    return " ".join(t.translate(_DIGITO_LETRA) if "a" <= t[0] <= "z" else t for t in _TOKEN.findall(linha.lower()))


def _texto_html(h: str) -> str:
    h = _SCRIPT.sub(" ", h)
    h = re.sub(r"<\s*(br|p|div|tr|li)[^>]*>", "\n", h, flags=re.I)
    return html.unescape(_TAG.sub(" ", h))


def _decodificar_parte(parte, ja_texto: bool = False) -> str:
    # Texto colado (str) já está decodificado: só base64 e quoted-printable precisam ser desfeitos. Sem isto, o
    # e-mail cai no raw-unicode-escape e "’" vira "’".
    if ja_texto and (parte.get("Content-Transfer-Encoding") or "").strip().lower() not in ("base64", "quoted-printable"):
        p = parte.get_payload()
        return p if isinstance(p, str) else ""
    dados = parte.get_payload(decode=True)
    if dados is None:
        p = parte.get_payload()
        return p if isinstance(p, str) else ""
    charset = parte.get_content_charset() or "latin-1"
    try:
        return dados.decode(charset, errors="replace")
    except LookupError:
        return dados.decode("latin-1", errors="replace")


def _assunto(msg) -> str:
    bruto = msg.get("Subject")
    if bruto is None:
        return ""
    try:
        return str(make_header(decode_header(str(bruto))))
    except Exception:
        return str(bruto)


def _parece_mensagem(texto: str) -> bool:
    """Tem cabeçalhos de e-mail (Subject:, From:, ...) antes de uma linha vazia?"""
    cab = texto.lstrip().split("\n\n", 1)[0]
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9-]*:", cab)) and bool(re.search(r"^(subject|from|to|date|message-id):", cab, re.I | re.M))


def extrair(bruto: str | bytes) -> tuple[str, str]:
    """(assunto, corpo) de um e-mail bruto ou de um texto colado."""
    if isinstance(bruto, bytes):
        texto = bruto.decode("latin-1")
    else:
        texto = bruto
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    if not _parece_mensagem(texto):
        # texto colado: se a primeira linha for "Subject: ..." (ou "Assunto: ..."), ela é o assunto
        linhas = texto.strip("\n").split("\n")
        m = re.match(r"^\s*(subject|assunto)\s*:\s*(.*)$", linhas[0], re.I) if linhas else None
        return (m.group(2), "\n".join(linhas[1:])) if m else ("", texto)
    msg = email.message_from_string(texto, policy=policy.compat32)
    textos, htmls = [], []
    for parte in (msg.walk() if msg.is_multipart() else [msg]):
        if parte.is_multipart():
            continue
        tipo = parte.get_content_type()
        if parte.get_content_disposition() == "attachment":
            continue
        ja_texto = isinstance(bruto, str)
        if tipo == "text/plain":
            textos.append(_decodificar_parte(parte, ja_texto))
        elif tipo == "text/html":
            htmls.append(_texto_html(_decodificar_parte(parte, ja_texto)))
    corpo = "\n".join(textos) if textos else "\n".join(htmls)
    return _assunto(msg), corpo


def para_aueb(bruto: str | bytes) -> str:
    """Texto no formato pré-processado da AUEB."""
    assunto, corpo = extrair(bruto)
    corpo = _ENTIDADE.sub("", _TAG.sub("", corpo))   # a AUEB apaga <...> e &...; também no texto puro
    linhas = [tokenizar_linha(l) for l in corpo.split("\n")]
    return "\n".join(["Subject: " + tokenizar_linha(assunto)] + [l for l in linhas if l])


def vetor(texto_aueb: str, vocab) -> np.ndarray:
    """Contagem por substring (como o CSV)."""
    return np.array([texto_aueb.count(w) for w in vocab], dtype=np.int32)


def cobertura(texto_aueb: str, vocab) -> float:
    """Fração das unidades do texto (palavras com 2+ letras, sem o 'Subject:') que são palavras do vocabulário."""
    palavras = [p for p in re.findall(r"[a-z]{2,}", texto_aueb.split("\n", 1)[-1] if "\n" in texto_aueb else texto_aueb[9:])]
    if not palavras:
        return 0.0
    v = set(vocab)
    return float(np.mean([p in v for p in palavras]))

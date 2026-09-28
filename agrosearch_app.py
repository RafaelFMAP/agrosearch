"""
AgroSearch — Motor de Busca Inteligente (TF-IDF implementado do zero).

Executar com:
    streamlit run agrosearch_app.py

Restrição do desafio: nada de scikit-learn/TfidfVectorizer. O pré-processamento (inclusive o stemmer),
o índice invertido, o TF-IDF e a similaridade de cosseno são implementados abaixo apenas com a biblioteca
padrão do Python. Streamlit e pandas são usados só para a interface e para exibir tabelas.

    Fase 1  preprocessar()        tokenização → normalização → stopwords → stemming
    Fase 2  construir_indice()    índice invertido  termo → {doc: frequência}
    Fase 3  ranquear_tfidf()      TF, IDF, TF-IDF e score acumulado da consulta
    Bônus   ranquear_cosseno()    similaridade de cosseno entre o vetor da consulta e os vetores TF-IDF
"""

import math
import re
import unicodedata
from collections import Counter

import pandas as pd
import streamlit as st

# =============================================================================
# Base de documentos (hardcode sugerido no enunciado)
# =============================================================================
DOCUMENTOS_PADRAO = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]
EXEMPLOS = ["lagartas na soja", "irrigar a soja", "controle de pragas", "adubação do milho", "economia de água no cultivo"]

STOPWORDS = """
a à ao aos as às até com como da das de dela dele deles do dos e é ela ele eles em entre era essa esse esta está
estão este eu foi foram há isso isto já la lá mais mas me mesmo meu minha muito na nas não nem no nos nós o os ou
para pela pelas pelo pelos por qual quando que quem se sem ser seu sua são só também te tem tinha um uma umas uns
você vocês ali aqui ter todo toda todos todas nesta neste nessa nesse desde após sobre porque pois então pode podem
cada outro outra outros outras seja sejam sendo sido num numa lhe lhes durante
""".split()


# =============================================================================
# Fase 1 — Pipeline de pré-processamento
# =============================================================================
def tokenizar(texto):
    """Etapa 1: separa o texto em palavras (sequências de letras e dígitos), descartando pontuação."""
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", texto)


def normalizar(token):
    """Etapa 2: minúsculas e remoção de acentos (ç → c, ã → a, ...)."""
    decomposto = unicodedata.normalize("NFD", token.lower())
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


STOPWORDS_NORM = {normalizar(p) for p in STOPWORDS}


class StemmerPT:
    """Stemmer simplificado para o português, inspirado nas etapas do algoritmo RSLP (Orengo & Huyck, 2001).

    Cada regra é (sufixo, tamanho mínimo do radical que deve sobrar, substituição). As etapas rodam em ordem:
    plural → feminino → advérbio → diminutivo → sufixo nominal → (se nenhum) sufixo verbal → (se nenhum) vogal final.
    Ex.: irrigações → irrigação → irrig ; lagartas → lagarta → lagart ; cultivar → cultiv ; cultivo → cultiv
    """

    PLURAL = [("oes", 1, "ao"), ("aes", 1, "ao"), ("ais", 1, "al"), ("eis", 2, "el"), ("ois", 1, "ol"),
              ("is", 2, "il"), ("ns", 1, "m"), ("res", 3, "r"), ("les", 3, "l"), ("s", 2, "")]
    FEMININO = [("ona", 3, "ao"), ("ora", 3, "or"), ("na", 4, "no"), ("inha", 3, "inho"), ("esa", 3, "es"),
                ("osa", 3, "oso"), ("iva", 3, "ivo"), ("ada", 2, "ado"), ("ida", 3, "ido"), ("ica", 3, "ico")]
    ADVERBIO = [("mente", 4, "")]
    DIMINUTIVO = [("zinho", 2, ""), ("inho", 3, ""), ("zito", 2, ""), ("ito", 3, ""), ("issimo", 3, "")]
    NOMINAL = [("amento", 3, ""), ("imento", 3, ""), ("mento", 4, ""), ("acao", 3, ""), ("icao", 3, ""),
               ("ucao", 3, ""), ("sao", 3, ""), ("cao", 3, ""), ("idade", 4, ""), ("dade", 4, ""), ("ismo", 3, ""),
               ("ista", 4, ""), ("avel", 2, ""), ("ivel", 3, ""), ("ador", 3, ""), ("edor", 3, ""), ("idor", 3, ""),
               ("ante", 2, ""), ("ico", 4, ""), ("oso", 3, ""), ("ura", 4, ""), ("al", 4, ""), ("ario", 3, ""),
               ("eiro", 3, "")]
    VERBAL = [("ariam", 2, ""), ("eriam", 2, ""), ("iriam", 3, ""), ("assem", 2, ""), ("essem", 2, ""),
              ("issem", 3, ""), ("ando", 2, ""), ("endo", 3, ""), ("indo", 3, ""), ("aram", 2, ""), ("eram", 3, ""),
              ("iram", 3, ""), ("avam", 2, ""), ("ava", 2, ""), ("ado", 2, ""), ("ido", 3, ""), ("am", 2, ""),
              ("em", 2, ""), ("ar", 2, ""), ("er", 2, ""), ("ir", 3, ""), ("ou", 3, ""), ("eu", 3, ""), ("iu", 3, "")]
    VOGAL = [("a", 3, ""), ("e", 3, ""), ("o", 3, "")]

    @staticmethod
    def _aplicar(palavra, regras):
        for sufixo, minimo, troca in regras:
            if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= minimo:
                return palavra[: len(palavra) - len(sufixo)] + troca, True
        return palavra, False

    def stem(self, palavra):
        if len(palavra) <= 3:
            return palavra
        if palavra.endswith("s"):
            palavra, _ = self._aplicar(palavra, self.PLURAL)
        if palavra.endswith("a"):
            palavra, _ = self._aplicar(palavra, self.FEMININO)
        palavra, _ = self._aplicar(palavra, self.ADVERBIO)
        palavra, _ = self._aplicar(palavra, self.DIMINUTIVO)
        palavra, mudou = self._aplicar(palavra, self.NOMINAL)
        if not mudou:
            palavra, mudou = self._aplicar(palavra, self.VERBAL)
        if not mudou:
            palavra, _ = self._aplicar(palavra, self.VOGAL)
        return palavra


STEMMER = StemmerPT()


def preprocessar(texto, usar_stopwords=True, usar_stemming=True):
    """Executa as 4 etapas e devolve os tokens finais e o resultado de cada etapa (para exibição)."""
    etapa1 = tokenizar(texto)
    etapa2 = [normalizar(t) for t in etapa1]
    etapa3 = [t for t in etapa2 if t not in STOPWORDS_NORM] if usar_stopwords else list(etapa2)
    etapa4 = [STEMMER.stem(t) for t in etapa3] if usar_stemming else list(etapa3)
    return etapa4, {"1. tokenização": etapa1, "2. normalização": etapa2,
                    "3. stopwords": etapa3, "4. stemming": etapa4}


# =============================================================================
# Fase 2 — Índice invertido
# =============================================================================
def construir_indice(tokens_por_doc):
    """termo → {id_doc: frequência do termo no documento}, com os termos em ordem alfabética."""
    indice = {}
    for doc_id, tokens in tokens_por_doc.items():
        for termo, freq in Counter(tokens).items():
            indice.setdefault(termo, {})[doc_id] = freq
    return dict(sorted(indice.items()))


# =============================================================================
# Fase 3 — TF, IDF e TF-IDF
# =============================================================================
ESQUEMAS_TF = {
    "Normalizada: f(t,d) / |d|": lambda f, n: f / n if n else 0.0,
    "Bruta: f(t,d)": lambda f, n: float(f),
    "Logarítmica: 1 + log₁₀ f(t,d)": lambda f, n: 1 + math.log10(f) if f > 0 else 0.0,
}


def calcular_tf(freq, tamanho_doc, esquema):
    return ESQUEMAS_TF[esquema](freq, tamanho_doc)


def calcular_idf(indice, n_docs):
    """IDF(t) = log₁₀(N / df(t)). Termo presente em todos os documentos tem IDF = 0 (não discrimina nada)."""
    return {termo: math.log10(n_docs / len(postings)) for termo, postings in indice.items()}


def tfidf_documento(termo, doc_id, indice, idf, tamanhos, esquema):
    freq = indice.get(termo, {}).get(doc_id, 0)
    return calcular_tf(freq, tamanhos[doc_id], esquema) * idf.get(termo, 0.0)


def ranquear_tfidf(termos_consulta, indice, idf, tamanhos, esquema):
    """Score(q, d) = Σ TF-IDF(t, d) para cada termo distinto t da consulta presente no vocabulário."""
    termos = [t for t in dict.fromkeys(termos_consulta) if t in indice]
    scores = {d: sum(tfidf_documento(t, d, indice, idf, tamanhos, esquema) for t in termos) for d in tamanhos}
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])), termos


# =============================================================================
# Bônus — Similaridade de cosseno
# =============================================================================
def vetor_documento(doc_id, indice, idf, tamanhos, esquema):
    return {t: tfidf_documento(t, doc_id, indice, idf, tamanhos, esquema) for t, p in indice.items() if doc_id in p}


def vetor_consulta(termos_consulta, idf, esquema):
    contagem = Counter(t for t in termos_consulta if t in idf)
    n = sum(contagem.values())
    return {t: calcular_tf(f, n, esquema) * idf[t] for t, f in contagem.items()}


def cosseno(u, v):
    produto = sum(peso * v.get(t, 0.0) for t, peso in u.items())
    norma_u = math.sqrt(sum(p * p for p in u.values()))
    norma_v = math.sqrt(sum(p * p for p in v.values()))
    return produto / (norma_u * norma_v) if norma_u and norma_v else 0.0


def ranquear_cosseno(termos_consulta, indice, idf, tamanhos, esquema):
    q = vetor_consulta(termos_consulta, idf, esquema)
    scores = {d: cosseno(q, vetor_documento(d, indice, idf, tamanhos, esquema)) for d in tamanhos}
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])), q


# =============================================================================
# Interface Streamlit
# =============================================================================
st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
    .tok {display:inline-block; padding:.05rem .45rem; margin:.1rem; border-radius:6px; font-family:monospace;
          font-size:.85rem; background:rgba(46,125,50,.12); border:1px solid rgba(46,125,50,.35);}
    .tok.fora {background:rgba(198,40,40,.10); border-color:rgba(198,40,40,.35); text-decoration:line-through;}
    .vencedor {border:2px solid #2e7d32; border-radius:10px; padding:.8rem 1rem; background:rgba(46,125,50,.08);}
    mark {background: rgba(255,213,79,.6); color: inherit; padding: 0 .1rem; border-radius: 3px;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🌱 AgroSearch")
    st.caption("Motor de busca TF-IDF implementado do zero")
    st.subheader("Pipeline de pré-processamento")
    usar_stopwords = st.checkbox("Remover stopwords", value=True)
    usar_stemming = st.checkbox("Aplicar stemming", value=True)
    st.caption("Tokenização e normalização (minúsculas + sem acentos) estão sempre ativas.")
    st.subheader("Ranqueamento")
    esquema_tf = st.radio("Cálculo do TF", list(ESQUEMAS_TF), index=0)
    st.caption("IDF(t) = log₁₀(N / df(t))")

    with st.expander("📄 Base de documentos (editável)"):
        base = st.data_editor(pd.DataFrame({"texto": DOCUMENTOS_PADRAO}, index=[f"Doc {i + 1}" for i in range(5)]),
                              num_rows="dynamic", width="stretch", key="base")
        st.caption("Edite ou adicione documentos: o índice é reconstruído na hora.")

docs = {(str(i) if str(i).startswith("Doc") else f"Doc {k + 1}"): str(t)
        for k, (i, t) in enumerate(zip(base.index, base["texto"])) if isinstance(t, str) and t.strip()}
if not docs:
    st.error("A base de documentos está vazia.")
    st.stop()

processados = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in docs.items()}
tokens_por_doc = {d: toks for d, (toks, _) in processados.items()}
indice = construir_indice(tokens_por_doc)
idf = calcular_idf(indice, len(docs))
tamanhos = {d: len(t) for d, t in tokens_por_doc.items()}

st.title("🌱 AgroSearch — Motor de Busca Inteligente")
st.caption(f"{len(docs)} documentos · vocabulário de {len(indice)} termos · "
           f"stopwords {'removidas' if usar_stopwords else 'mantidas'} · stemming {'ligado' if usar_stemming else 'desligado'}")

aba_pre, aba_indice, aba_busca, aba_cos = st.tabs(
    ["1️⃣ Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca TF-IDF", "⭐ Similaridade de Cosseno (bônus)"]
)

# --- Fase 1 --------------------------------------------------------------------
with aba_pre:
    st.markdown("#### Pipeline em 4 etapas")
    st.markdown("**1. Tokenização** → **2. Normalização** (minúsculas, sem acentos) → **3. Stopwords** → **4. Stemming**")

    doc_sel = st.selectbox("Ver etapa por etapa do documento", list(docs))
    _, etapas = processados[doc_sel]
    st.caption(docs[doc_sel])
    for nome, toks in etapas.items():
        ativo = not ((nome.startswith("3") and not usar_stopwords) or (nome.startswith("4") and not usar_stemming))
        c1, c2 = st.columns([1, 5])
        c1.markdown(f"**{nome}**" + ("" if ativo else " _(desligada)_") + f"<br><small>{len(toks)} tokens</small>",
                    unsafe_allow_html=True)
        if nome.startswith("3") and usar_stopwords:
            html = "".join(f'<span class="tok{" fora" if t in STOPWORDS_NORM else ""}">{t}</span>'
                           for t in etapas["2. normalização"])
        else:
            html = "".join(f'<span class="tok">{t}</span>' for t in toks)
        c2.markdown(html, unsafe_allow_html=True)

    st.markdown("#### Como as opções mudam o vocabulário")

    def tamanho_vocab(sw, stm):
        return len({t for texto in docs.values() for t in preprocessar(texto, sw, stm)[0]})

    combos = pd.DataFrame([{"stopwords removidas": sw, "stemming": stm, "termos no vocabulário": tamanho_vocab(sw, stm),
                            "configuração atual": "👉" if (sw, stm) == (usar_stopwords, usar_stemming) else ""}
                           for sw in (False, True) for stm in (False, True)])
    v1, v2 = st.columns([2, 3])
    v1.dataframe(combos, hide_index=True, width="stretch")
    v1.metric("Vocabulário atual", len(indice),
              delta=len(indice) - tamanho_vocab(False, False), delta_color="inverse",
              help="Diferença em relação ao vocabulário sem stopwords nem stemming")

    if usar_stemming:
        grupos = {}
        for texto in docs.values():
            for palavra in preprocessar(texto, usar_stopwords, False)[0]:
                grupos.setdefault(STEMMER.stem(palavra), set()).add(palavra)
        unidos = {r: sorted(p) for r, p in grupos.items() if len(p) > 1}
        v2.markdown("**Palavras unificadas pelo stemming neste corpus**")
        if unidos:
            v2.dataframe(pd.DataFrame([{"radical": r, "palavras": ", ".join(p)} for r, p in sorted(unidos.items())]),
                         hide_index=True, width="stretch")
        else:
            v2.caption("Nenhuma: no corpus padrão cada radical vem de uma única forma. O ganho aparece na consulta "
                       "(ex.: “irrigar” → irrig casa com “irrigação”).")
        v2.markdown("**Exemplos do stemmer** (formas diferentes → mesmo radical)")
        v2.dataframe(pd.DataFrame([{"palavra": p, "radical": STEMMER.stem(normalizar(p))} for p in
                                   ["irrigação", "irrigações", "irrigar", "lagarta", "lagartas", "cultivo", "cultivar",
                                    "adubação", "adubar", "orgânica", "orgânicos"]]),
                     hide_index=True, width="stretch", height=250)

# --- Fase 2 --------------------------------------------------------------------
with aba_indice:
    st.markdown("#### Índice invertido: termo → documentos")
    st.caption("Construído a partir dos tokens pré-processados. df(t) = número de documentos que contêm o termo.")
    filtro = st.text_input("Filtrar termos", placeholder="ex.: soj, irrig, lagart")
    linhas = [{"termo": t, "df": len(p), "documentos": ", ".join(p),
               "postings (doc: freq)": ", ".join(f"{d}: {f}" for d, f in p.items()), "IDF": idf[t]}
              for t, p in indice.items() if filtro.strip().lower() in t]
    i1, i2 = st.columns([3, 2])
    i1.dataframe(pd.DataFrame(linhas).style.format({"IDF": "{:.4f}"}).bar(subset=["df"], color="#a5d6a7", vmin=0),
                 hide_index=True, width="stretch", height=460)
    with i2:
        st.markdown("**Estrutura em JSON** (termo → [IDs dos documentos])")
        st.json({t: list(p) for t, p in indice.items() if filtro.strip().lower() in t}, expanded=True)

# --- Fase 3 --------------------------------------------------------------------
if "consulta" not in st.session_state:
    st.session_state.consulta = "lagartas na soja"


def formulario_consulta(chave):
    exemplo = st.pills("Exemplos", EXEMPLOS, selection_mode="single", key=f"pills_{chave}")
    if exemplo and exemplo != st.session_state.get(f"_ultimo_{chave}"):
        st.session_state.consulta = exemplo
    st.session_state[f"_ultimo_{chave}"] = exemplo
    return st.text_input("🔎 Consulta", key="consulta")


def destacar(texto, termos):
    alvo = set(termos)
    return re.sub(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+",
                  lambda m: f"<mark>{m.group(0)}</mark>" if (
                      STEMMER.stem(normalizar(m.group(0))) if usar_stemming else normalizar(m.group(0))) in alvo
                  else m.group(0), texto)


with aba_busca:
    consulta = formulario_consulta("tfidf")
    termos_q, etapas_q = preprocessar(consulta, usar_stopwords, usar_stemming)
    ranking, termos_validos = ranquear_tfidf(termos_q, indice, idf, tamanhos, esquema_tf)
    ignorados = [t for t in dict.fromkeys(termos_q) if t not in indice]

    st.markdown("**Consulta pré-processada:** " + ("".join(f'<span class="tok{" fora" if t in ignorados else ""}">{t}</span>'
                                                         for t in dict.fromkeys(termos_q)) or "_(vazia)_"),
                unsafe_allow_html=True)
    if ignorados:
        st.caption(f"Termos riscados não existem no vocabulário e não contribuem: {', '.join(ignorados)}")

    if not termos_validos:
        st.warning("Nenhum termo da consulta está no índice. Tente outra consulta ou ligue o stemming.")
    else:
        st.latex(r"\text{TF-IDF}(t,d)=\text{TF}(t,d)\times\log_{10}\frac{N}{df(t)}\qquad"
                 r"\text{Score}(q,d)=\sum_{t\in q}\text{TF-IDF}(t,d)")
        vencedor, nota = ranking[0]
        if nota > 0:
            st.markdown(f'<div class="vencedor">🏆 <b>Documento vencedor: {vencedor}</b> — TF-IDF acumulado = '
                        f'<b>{nota:.4f}</b><br>{destacar(docs[vencedor], termos_validos)}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Todos os documentos têm score 0: os termos da consulta aparecem em todos os documentos (IDF = 0).")

        st.markdown("#### Ranking por TF-IDF acumulado")
        tabela = pd.DataFrame([{"posição": k + 1, "documento": d, "TF-IDF acumulado": s,
                                **{f"TF-IDF({t})": tfidf_documento(t, d, indice, idf, tamanhos, esquema_tf)
                                   for t in termos_validos},
                                "texto": docs[d]} for k, (d, s) in enumerate(ranking)])

        def cor_vencedor(linha):
            ok = linha["posição"] == 1 and linha["TF-IDF acumulado"] > 0
            return ["background-color: rgba(46,125,50,.25); font-weight: 700" if ok else ""] * len(linha)

        st.dataframe(tabela.style.apply(cor_vencedor, axis=1)
                     .format({c: "{:.4f}" for c in tabela.columns if c.startswith("TF-IDF")}),
                     hide_index=True, width="stretch")
        st.bar_chart(tabela.set_index("documento")[[c for c in tabela.columns if c.startswith("TF-IDF(")]],
                     horizontal=True, height=240)

        st.markdown("#### Passo a passo do cálculo")
        passos = []
        for t in termos_validos:
            for d in docs:
                f = indice[t].get(d, 0)
                passos.append({"termo": t, "documento": d, "f(t,d)": f, "|d|": tamanhos[d],
                               "TF": calcular_tf(f, tamanhos[d], esquema_tf), "df(t)": len(indice[t]),
                               "N": len(docs), "IDF": idf[t],
                               "TF-IDF": calcular_tf(f, tamanhos[d], esquema_tf) * idf[t]})
        st.dataframe(pd.DataFrame(passos).style.format({"TF": "{:.4f}", "IDF": "{:.4f}", "TF-IDF": "{:.4f}"})
                     .highlight_between(subset=["TF-IDF"], left=1e-12, color="rgba(255,213,79,.45)"),
                     hide_index=True, width="stretch")
        st.caption(f"TF: {esquema_tf}. Cada termo distinto da consulta conta uma vez no score acumulado.")

    with st.expander("🧮 Matriz TF-IDF completa (termos × documentos)"):
        matriz = pd.DataFrame({d: {t: tfidf_documento(t, d, indice, idf, tamanhos, esquema_tf) for t in indice}
                               for d in docs})
        st.dataframe(matriz.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None),
                     width="stretch", height=380)

# --- Bônus: cosseno ------------------------------------------------------------
with aba_cos:
    st.markdown("#### Similaridade de cosseno entre o vetor da consulta e os vetores TF-IDF dos documentos")
    st.latex(r"\cos(\vec q,\vec d)=\frac{\sum_t w_{t,q}\,w_{t,d}}{\sqrt{\sum_t w_{t,q}^2}\;\sqrt{\sum_t w_{t,d}^2}}"
             r"\qquad w_{t,x}=\text{TF}(t,x)\times\text{IDF}(t)")
    consulta_c = st.session_state.consulta
    st.caption(f"Consulta atual: “{consulta_c}” (altere na aba 3️⃣)")
    termos_c, _ = preprocessar(consulta_c, usar_stopwords, usar_stemming)
    rank_cos, vq = ranquear_cosseno(termos_c, indice, idf, tamanhos, esquema_tf)
    rank_tfidf, _ = ranquear_tfidf(termos_c, indice, idf, tamanhos, esquema_tf)
    pos_tfidf = {d: k + 1 for k, (d, _) in enumerate(rank_tfidf)}

    if not vq:
        st.warning("O vetor da consulta é nulo (nenhum termo no vocabulário ou todos com IDF = 0).")
    else:
        st.markdown("**Vetor da consulta:** " + " · ".join(f"`{t}` = {w:.4f}" for t, w in vq.items()))
        comp = pd.DataFrame([{"posição (cosseno)": k + 1, "documento": d, "cosseno": s,
                              "posição (TF-IDF acumulado)": pos_tfidf[d],
                              "TF-IDF acumulado": dict(rank_tfidf)[d],
                              "norma ‖d‖": math.sqrt(sum(w * w for w in vetor_documento(
                                  d, indice, idf, tamanhos, esquema_tf).values())),
                              "texto": docs[d]} for k, (d, s) in enumerate(rank_cos)])
        st.dataframe(comp.style.format({"cosseno": "{:.4f}", "TF-IDF acumulado": "{:.4f}", "norma ‖d‖": "{:.4f}"})
                     .background_gradient(subset=["cosseno"], cmap="Greens", vmin=0, vmax=1),
                     hide_index=True, width="stretch")
        mudou = [d for d in docs if pos_tfidf[d] != comp.set_index("documento").loc[d, "posição (cosseno)"]]
        if mudou:
            st.info(f"O cosseno reordenou {', '.join(mudou)} em relação ao TF-IDF acumulado: ele divide pela norma do "
                    "documento, então favorece documentos cujo conteúdo é *proporcionalmente* mais sobre a consulta.")
        else:
            st.success("Os dois métodos produzem a mesma ordem para esta consulta.")
        st.markdown("""
**Por que o cosseno ajuda em consultas de várias palavras?**
O TF-IDF acumulado só soma pesos, então um documento pode vencer por ter um único termo raro com peso alto.
O cosseno compara a **direção** dos vetores: premia documentos que cobrem *vários* termos da consulta na mesma
proporção e normaliza o tamanho do documento (a norma ‖d‖), o que evita que documentos com muitos termos
distintos sejam favorecidos ou penalizados só pelo tamanho.
        """)

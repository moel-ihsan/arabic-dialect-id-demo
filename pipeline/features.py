# ============================================================
# features.py  (FULL FEATURE STACK — CV SAFE)
# REVISED: semua bug diperbaiki, overlap char TF-IDF diselesaikan,
#          _doc_stats duplikat kolom dihapus, typo "sigmod" diperbaiki,
#          tfidf_char (3,4) komplementer dengan char_vec23/45 (MI),
#          bm25c (2,3) tidak overlap dengan tfidf_char (3,4),
#          OOF Interaction (product/diff/max) diurus di sini,
#          chi2_k & mi_k sekarang ADAPTIF via percentile vocab
# ============================================================
import re
import os
import unicodedata
import numpy as np
from scipy.sparse import hstack, csr_matrix, diags
from scipy.stats  import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MaxAbsScaler

# ============================================================
# STOPWORDS
# ============================================================
FEATURES_VERSION = "Shami Extractor"
BASE_DIR      = os.path.dirname(__file__)
STOPWORD_PATH = os.path.join(BASE_DIR, "stopwords.txt")

manual_stopwords = set([
    "شو","ما","من","عن","مع","في","على","هو","هون","هونيك",
    "هيك","هي","هدا","هديك","هدول","هدي","هنيك","لك","طيب",
    "بس","إذا","إنه","إنو","انو","و","وما","هالقد",
    "قديش","كتير","شوي","كل","ولا","كمان","بعد","قبل","لسا",
    "بقى","معك","إلك","إلي","إلنا","عنا","عندي","عندو",
    "عنها","مافي","فيه","فيها","هايك","وهلق","قوم","نام",
    "صار","كان","كانو","كانت","كنت","صرلي","كيف",
    "هل","ها","إيه","اي","آه","انا","الي","ان","الناس","حتى","عليه",
    "يوم","اليوم","مره","عند","واحد","عليها","يكون","ناس",
    "وحده","انه","معي","بكل","طبعا","يمكن","بعض","قال",
    "هذا","يا","او","والله","الله","في","من","إلى","عن",
    "أن","لا","لن","لم","إن","قد","ثم","بل","أي","أين","انت",
])
file_stopwords = set()
if os.path.exists(STOPWORD_PATH):
    with open(STOPWORD_PATH, encoding="utf-8") as f:
        file_stopwords = set(w.strip() for w in f if w.strip())
topic_stopwords = {
    "لبنان","لبناني","لبنانيه","بيروت",
    "سوريا","سوري","سوريه","شام",
    "اردن","اردني","اردنيه","عمان",
    "فلسطين","فلسطيني","فلسطينيه","قدس",
    "اليسا","نانسي","فيروز","مدريد","بايرن","ميونخ","ترامب",
}
AR_STOP = manual_stopwords.union(file_stopwords).union(topic_stopwords)

# ============================================================
# REGEX
# ============================================================
_NOISE_RE = re.compile( r"(<.*?>|" r"http\S+|www\S+|" r"@\w+|#\w+|" r"\d+[-\d]*|" r"[:;=8][\-^]?[)DPOp]|" r"[~^*_+=\[\]{}|\\/<>-]+|" r"[!?؟]+|" r"\s{2,})" )

TOK_AR         = re.compile(r"[\u0621-\u064A]{2,}")
_WORD_RE       = re.compile(r"[A-Za-z\u0621-\u064A]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?؟؛…]+")

# ============================================================
# CLEAN & NORMALIZE
# ============================================================
def clean_noise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _NOISE_RE.sub(" ", text).strip()

def normalize_light(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return text.replace("ـ", "")

def normalize_aggr(text):
    text = normalize_light(text)
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub("ى",    "ي", text)
    text = re.sub("ؤ",    "و", text)
    text = re.sub("ئ",    "ي", text)
    text = re.sub("ة",    "ه", text)
    return text

def clean_char(text):
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)

# ============================================================
# ANALYZERS
# ============================================================
def analyzer_light(text: str):
    toks = TOK_AR.findall(normalize_light(clean_noise(text)))
    return [t for t in toks if t not in AR_STOP]

def analyzer_aggr(text: str):
    toks = TOK_AR.findall(normalize_aggr(clean_noise(text)))
    return [t for t in toks if t not in AR_STOP]

# ============================================================
# NB-SVM
# ============================================================
def _nbsvm_fit(texts, y, analyzer, ngram_range, min_df, alpha=1.0, stop_words=None):
    v = CountVectorizer(
        analyzer=analyzer, lowercase=False,
        ngram_range=ngram_range, min_df=min_df,
        token_pattern=None if callable(analyzer) else None,
        stop_words=stop_words, binary=True,
    )
    X = v.fit_transform(texts)
    y = np.asarray(y).ravel()
    r_list = []
    for c in np.unique(y):
        pos = X[y == c].sum(axis=0)
        neg = X[y != c].sum(axis=0)
        r   = np.log((np.asarray(pos).ravel() + alpha) /
                     (np.asarray(neg).ravel() + alpha))
        r_list.append(r.astype(np.float32))
    return v, r_list

def _nbsvm_transform(vectorizer, r_list, texts):
    X = vectorizer.transform(texts)
    return csr_matrix(np.hstack([(X @ r[:, None]) for r in r_list]))

# ============================================================
# BM25
# ============================================================
class BM25Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, analyzer="word", ngram_range=(1, 2),
                 min_df=3, max_df=0.95, k1=1.2, b=0.75):
        self.analyzer    = analyzer
        self.ngram_range = ngram_range
        self.min_df      = min_df
        self.max_df      = max_df
        self.k1          = k1
        self.b           = b
        self.cv          = None

    def fit(self, X, y=None):
        self.cv = CountVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range,
            min_df=self.min_df, max_df=self.max_df,
        )
        Xc          = self.cv.fit_transform(X).tocsr()
        N           = Xc.shape[0]
        self.dl_    = np.asarray(Xc.sum(axis=1)).ravel()
        self.avgdl_ = self.dl_.mean() + 1e-9
        df          = np.diff(Xc.tocsc().indptr)
        self.idf_   = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def transform(self, X):
        Xc    = self.cv.transform(X).tocsr().tocoo()
        dl    = np.asarray(self.cv.transform(X).tocsr().sum(axis=1)).ravel()
        denom = self.k1 * (1 - self.b + self.b * dl / self.avgdl_)
        w     = (self.idf_[Xc.col]
                 * ((self.k1 + 1) * Xc.data)
                 / (denom[Xc.row] + Xc.data))
        return csr_matrix((w, (Xc.row, Xc.col)), shape=Xc.shape)

# ============================================================
# CHI2 SCALER
# ============================================================
class Chi2Scaler:
    def __init__(self, pow=0.5):
        self.pow = pow

    def fit(self, X, y):
        self.w_ = np.power(chi2(X, y)[0] + 1e-12, self.pow)
        return self

    def transform(self, X):
        return X @ diags(self.w_, 0)

# ============================================================
# MUTUAL INFORMATION SELECTOR
# ============================================================
class MutualInfoSelector:
    """Seleksi fitur berbasis MI untuk char n-gram.

    Rank disimpan saat fit() sehingga update_k() hanya menggeser
    cutoff tanpa re-komputasi MI (O(1), tidak mahal).
    """
    def __init__(self, k=5000, pow=0.4):
        self.k   = k
        self.pow = pow

    def fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        scores        = mutual_info_classif(X, y, discrete_features="auto", random_state=42)
        self._rank    = np.argsort(scores)[::-1]
        self._scores  = scores
        self._n_total = X.shape[1]
        self._apply_from_rank()
        return self

    def _apply_from_rank(self):
        k_eff      = min(self.k, self._n_total)
        self._cols = self._rank[:k_eff]
        self._w    = np.power(self._scores[self._cols] + 1e-12, self.pow).astype(np.float32)

    def update_k(self, new_k):
        """Geser cutoff tanpa re-fit MI."""
        if not hasattr(self, "_rank"):
            raise ValueError("MutualInfoSelector belum di-fit.")
        self.k = new_k
        self._apply_from_rank()

    def transform(self, X):
        return X[:, self._cols] @ diags(self._w, 0)

    @property
    def n_selected(self):
        return len(self._cols)

# ============================================================
# MAIN FEATURE EXTRACTOR
# ============================================================
class ShamiFeatureExtractor(BaseEstimator, TransformerMixin):
    """Ekstraksi fitur lengkap untuk klasifikasi dialek Syami.

    Parameter percentile (menggantikan angka absolut hardcoded):
    ─────────────────────────────────────────────────────────────
    chi2_pct  : float, default 0.15
        Proporsi lexical vocab yang dipertahankan setelah Chi2.
        Contoh: vocab=30.000, chi2_pct=0.15 -> chi2_k=4.500.
    mi_pct23  : float, default 0.10
        Proporsi char(2,3) vocab via MI. Lebih agresif karena noisy.
    mi_pct45  : float, default 0.15
        Proporsi char(4,5) vocab via MI. Lebih longgar karena informatif.
    chi2_min  : int, default 100  -- floor absolut chi2_k
    mi_min23  : int, default 500  -- floor absolut mi_k23
    mi_min45  : int, default 500  -- floor absolut mi_k45

    k efektif tersimpan di: _chi2_k_eff, _mi_k23_eff, _mi_k45_eff
    dan bisa di-inspect via feat.mi_info.

    Tuning tanpa re-fit dari nol:
        feat.update_chi2_pct(0.10)
        feat.update_mi_pct(new_pct23=0.07, new_pct45=0.12)
    Atau masih bisa pakai absolut (backward-compat):
        feat.update_chi2_k(3000)
        feat.update_mi_k(new_k23=2000, new_k45=3000)
    """

    def __init__(self,
                 chi2_pct=0.15, mi_pct23=0.10, mi_pct45=0.15,
                 chi2_min=100,  mi_min23=500,   mi_min45=500,
                 use_oof_lr=True, use_oof_interaction=True):

        # percentile params
        self.chi2_pct = chi2_pct
        self.mi_pct23 = mi_pct23
        self.mi_pct45 = mi_pct45
        self.chi2_min = chi2_min
        self.mi_min23 = mi_min23
        self.mi_min45 = mi_min45

        # k efektif (diisi saat fit)
        self._chi2_k_eff = None
        self._mi_k23_eff = None
        self._mi_k45_eff = None
        self.chi2_k      = None   # alias agar update_chi2_k tetap kompatibel

        # TF-IDF word-level
        self.tfidf_wL_uni = TfidfVectorizer(
            analyzer=analyzer_light, ngram_range=(1, 1),
            min_df=3, max_df=0.95, sublinear_tf=True)
        self.tfidf_wL_23  = TfidfVectorizer(
            analyzer=analyzer_light, ngram_range=(2, 3),
            min_df=3, max_df=0.95, sublinear_tf=True)
        self.tfidf_wA_uni = TfidfVectorizer(
            analyzer=analyzer_aggr, ngram_range=(1, 1),
            min_df=3, max_df=0.95, sublinear_tf=True)
        self.tfidf_wA_23  = TfidfVectorizer(
            analyzer=analyzer_aggr, ngram_range=(2, 3),
            min_df=3, max_df=0.95, sublinear_tf=True)

        # TF-IDF char-level
        # tfidf_char     (3,4) char_wb -> lexical block
        # tfidf_char_svm (3,5) char_wb -> OOF SVM input
        # char_vec23/45  "char" (bukan char_wb) -> MI block, tidak overlap
        self.tfidf_char     = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4),
            min_df=5, max_df=0.95, sublinear_tf=True)
        self.tfidf_char_svm = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=5, max_df=0.95, sublinear_tf=True)
        self.char_vec23 = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 3), min_df=3, sublinear_tf=True)
        self.char_vec45 = TfidfVectorizer(
            analyzer="char", ngram_range=(4, 5), min_df=3, sublinear_tf=True)

        # MI selectors (k diisi saat fit setelah vocab diketahui)
        self._mi_sel23 = MutualInfoSelector(pow=0.4)
        self._mi_sel45 = MutualInfoSelector(pow=0.4)

        # raw train matrices (agar update_mi_pct bisa re-apply tanpa re-fit MI)
        self._X_char23_tr = None
        self._X_char45_tr = None

        # BM25
        self.bm25w = BM25Vectorizer(analyzer="word",    ngram_range=(1, 3), min_df=3)
        self.bm25c = BM25Vectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5)

        # scalers
        self.scaler_t = MaxAbsScaler()
        self.scaler_b = MaxAbsScaler()

        # NB-SVM
        self.nb_alpha       = 1.0
        self.nb_char_ngram  = (3, 5)
        self.nb_char_min_df = 5

        # Chi2 state
        self._chi2_cols         = None
        self._chi2_scaler       = None
        self._chi2_rank         = None
        self._X_lexical_full_tr = None
        self._y_tr              = None

        # OOF
        self.use_oof_lr          = use_oof_lr
        self.use_oof_interaction = use_oof_interaction
        self.oof_lr_seeds        = (42, 1337)
        self.oof_lr_splits       = 5

    # ──────────────────────────────────────────────────────────
    # INTERNAL MATRIX BUILDERS
    # ──────────────────────────────────────────────────────────
    def _build_lr_matrix(self, cw):
        return hstack([
            self.tfidf_wL_uni.transform(cw), self.tfidf_wL_23.transform(cw),
            self.tfidf_wA_uni.transform(cw), self.tfidf_wA_23.transform(cw),
        ], format="csr")

    def _build_svm_matrix(self, cc):
        return self.tfidf_char_svm.transform(cc)

    # ──────────────────────────────────────────────────────────
    # OOF INTERACTION
    # ──────────────────────────────────────────────────────────
    def _build_oof_interactions(self, X_oof_lr, X_oof_svm):
        """n_classes*2 + 2 kolom: product, diff, lr_max, svm_max."""
        n_seeds   = len(self.oof_lr_seeds)
        n_classes = len(self._classes)
        lr_arr  = X_oof_lr.toarray()  if hasattr(X_oof_lr,  "toarray") else np.asarray(X_oof_lr)
        svm_arr = X_oof_svm.toarray() if hasattr(X_oof_svm, "toarray") else np.asarray(X_oof_svm)
        lr_mean  = np.zeros((lr_arr.shape[0],  n_classes), dtype=np.float32)
        svm_mean = np.zeros((svm_arr.shape[0], n_classes), dtype=np.float32)
        for s in range(n_seeds):
            lr_mean  += lr_arr[:,  s * n_classes:(s + 1) * n_classes]
            svm_mean += svm_arr[:, s * n_classes:(s + 1) * n_classes]
        lr_mean  /= n_seeds
        svm_mean /= n_seeds
        return csr_matrix(np.hstack([
            (lr_mean * svm_mean).astype(np.float32),
            (lr_mean - svm_mean).astype(np.float32),
            lr_mean.max(axis=1,  keepdims=True).astype(np.float32),
            svm_mean.max(axis=1, keepdims=True).astype(np.float32),
        ]))

    def _log_oof_correlation(self, X_oof_lr, X_oof_svm):
        n_classes = len(self._classes)
        n_seeds   = len(self.oof_lr_seeds)
        lr_arr    = X_oof_lr.toarray()
        svm_arr   = X_oof_svm.toarray()
        lr_mean   = sum(lr_arr[:,  s*n_classes:(s+1)*n_classes]
                        for s in range(n_seeds)) / n_seeds
        svm_mean  = sum(svm_arr[:, s*n_classes:(s+1)*n_classes]
                        for s in range(n_seeds)) / n_seeds
        print("\n=== Korelasi LR vs SVM (OOF train) ===")
        for i, c in enumerate(self._classes):
            r, _ = pearsonr(lr_mean[:, i], svm_mean[:, i])
            print(f"  class {c}: r={r:.3f}")
        disagree = (np.argmax(lr_mean, axis=1) != np.argmax(svm_mean, axis=1)).mean()
        print(f"  Disagreement rate: {disagree:.3f}")

    # ──────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────
    def fit(self, X, y=None):
        cw  = X["clean_word"]
        cc  = X["clean_char"]
        raw = X["text"]

        self._y_tr    = np.asarray(y)
        self._classes = np.unique(self._y_tr)
        C             = len(self._classes)

        # fit semua vectorizer
        self.tfidf_wL_uni.fit(cw);  self.tfidf_wL_23.fit(cw)
        self.tfidf_wA_uni.fit(cw);  self.tfidf_wA_23.fit(cw)
        self.tfidf_char.fit(cc);    self.tfidf_char_svm.fit(cc)
        self.bm25w.fit(cw);         self.bm25c.fit(cc)
        self.char_vec23.fit(cc);    self.char_vec45.fit(cc)

        # simpan raw MI matrices
        self._X_char23_tr = self.char_vec23.transform(cc)
        self._X_char45_tr = self.char_vec45.transform(cc)

        # hitung k efektif dari percentile
        lexical_vocab = (
            len(self.tfidf_wL_uni.vocabulary_) +
            len(self.tfidf_wL_23.vocabulary_)  +
            len(self.tfidf_wA_uni.vocabulary_) +
            len(self.tfidf_wA_23.vocabulary_)  +
            len(self.tfidf_char.vocabulary_)   +
            len(self.bm25w.cv.vocabulary_)     +
            len(self.bm25c.cv.vocabulary_)
        )
        self._chi2_k_eff = max(self.chi2_min, int(lexical_vocab * self.chi2_pct))
        self._mi_k23_eff = max(self.mi_min23, int(self._X_char23_tr.shape[1] * self.mi_pct23))
        self._mi_k45_eff = max(self.mi_min45, int(self._X_char45_tr.shape[1] * self.mi_pct45))
        self.chi2_k      = self._chi2_k_eff

        print(f"[vocab]     lexical={lexical_vocab:,}  "
              f"char23={self._X_char23_tr.shape[1]:,}  "
              f"char45={self._X_char45_tr.shape[1]:,}")
        print(f"[k efektif] chi2_k={self._chi2_k_eff:,}  "
              f"mi_k23={self._mi_k23_eff:,}  "
              f"mi_k45={self._mi_k45_eff:,}")

        # fit MI selectors
        self._mi_sel23.k = self._mi_k23_eff
        self._mi_sel45.k = self._mi_k45_eff
        self._mi_sel23.fit(self._X_char23_tr, self._y_tr)
        self._mi_sel45.fit(self._X_char45_tr, self._y_tr)

        # build & scale lexical train matrix
        X_tfidf_tr = hstack([
            self.tfidf_wL_uni.transform(cw), self.tfidf_wL_23.transform(cw),
            self.tfidf_wA_uni.transform(cw), self.tfidf_wA_23.transform(cw),
            self.tfidf_char.transform(cc),
        ])
        X_bm25_tr = hstack([self.bm25w.transform(cw), self.bm25c.transform(cc)])
        self.scaler_t.fit(X_tfidf_tr)
        self.scaler_b.fit(X_bm25_tr)
        self._X_lexical_full_tr = hstack([
            self.scaler_t.transform(X_tfidf_tr),
            self.scaler_b.transform(X_bm25_tr),
        ], format="csr")

        # NB-SVM
        self._nb_char_v, self._nb_char_r = _nbsvm_fit(
            texts=cc, y=self._y_tr, analyzer="char_wb",
            ngram_range=self.nb_char_ngram, min_df=self.nb_char_min_df,
            alpha=self.nb_alpha,
        )

        # Chi2 selection
        X_abs      = self._X_lexical_full_tr.copy()
        X_abs.data = np.abs(X_abs.data)
        scores, _  = chi2(X_abs, self._y_tr)
        self._chi2_rank = np.argsort(scores)[::-1]
        self._apply_chi2_from_rank()

        # block sizes
        self.block_sizes = {
            "lexical":     len(self._chi2_cols),
            "char_mi23":   self._mi_sel23.n_selected,
            "char_mi45":   self._mi_sel45.n_selected,
            "doc_stats":   self._doc_stats(raw).shape[1],
            "stopword":    self._stopword_stats(raw).shape[1],
            "morph":       self._morph_block(raw).shape[1],
            "morph_inter": self._morph_interactions(raw).shape[1],
            "colloc_dom":  1,
            "nbsvm":       C,
        }

        # collocation dominance
        self.colloc_vec = CountVectorizer(
            analyzer=analyzer_light, ngram_range=(2, 2), min_df=5)
        X_colloc = self.colloc_vec.fit_transform(cw)
        self.colloc_scores = np.zeros(X_colloc.shape[1], dtype=np.float32)
        for idx in range(X_colloc.shape[1]):
            col = X_colloc[:, idx].toarray().ravel()
            if col.sum() == 0:
                continue
            per_class = [col[self._y_tr == c].sum() for c in self._classes]
            self.colloc_scores[idx] = max(per_class) / (sum(per_class) + 1e-9)

        # OOF stacking
        if self.use_oof_lr:
            texts_cw = list(cw)
            texts_cc = list(cc)
            n        = len(texts_cw)

            self._oof_tr_lr  = np.zeros((n, C * len(self.oof_lr_seeds)), dtype=np.float32)
            self._oof_tr_svm = np.zeros((n, C * len(self.oof_lr_seeds)), dtype=np.float32)
            self._lr_models  = []
            self._svm_models = []

            for s_idx, seed in enumerate(self.oof_lr_seeds):
                skf = StratifiedKFold(n_splits=self.oof_lr_splits,
                                      shuffle=True, random_state=seed)
                for tr_idx, va_idx in skf.split(texts_cw, self._y_tr):
                    Xtr_lr  = self._build_lr_matrix([texts_cw[i] for i in tr_idx])
                    Xva_lr  = self._build_lr_matrix([texts_cw[i] for i in va_idx])
                    Xtr_svm = self._build_svm_matrix([texts_cc[i] for i in tr_idx])
                    Xva_svm = self._build_svm_matrix([texts_cc[i] for i in va_idx])

                    clf_lr = LogisticRegression(
                        C=5.0, solver="saga", multi_class="multinomial",
                        max_iter=500, n_jobs=-1, random_state=seed)
                    clf_lr.fit(Xtr_lr, self._y_tr[tr_idx])

                    clf_svm = CalibratedClassifierCV(
                        OneVsRestClassifier(
                            LinearSVC(C=1.0, max_iter=4000,
                                      random_state=seed, dual=True),
                            n_jobs=-1),
                        method="sigmoid", cv=3)
                    clf_svm.fit(Xtr_svm, self._y_tr[tr_idx])

                    sc = s_idx * C
                    self._oof_tr_lr[va_idx,  sc:sc+C] = clf_lr.predict_proba(Xva_lr)
                    self._oof_tr_svm[va_idx, sc:sc+C] = clf_svm.predict_proba(Xva_svm)

                # full-data models
                clf_lr_full = LogisticRegression(
                    C=5.0, solver="saga", multi_class="multinomial",
                    max_iter=500, n_jobs=-1, random_state=seed)
                clf_lr_full.fit(self._build_lr_matrix(texts_cw), self._y_tr)
                self._lr_models.append(clf_lr_full)

                clf_svm_full = CalibratedClassifierCV(
                    OneVsRestClassifier(
                        LinearSVC(C=1.0, max_iter=4000,
                                  random_state=seed, dual=True),
                        n_jobs=-1),
                    method="sigmoid", cv=3)
                clf_svm_full.fit(self._build_svm_matrix(texts_cc), self._y_tr)
                self._svm_models.append(clf_svm_full)

            self._oof_tr_lr  = csr_matrix(self._oof_tr_lr)
            self._oof_tr_svm = csr_matrix(self._oof_tr_svm)
            self.block_sizes["oof_lr"]  = C * len(self.oof_lr_seeds)
            self.block_sizes["oof_svm"] = C * len(self.oof_lr_seeds)

            if self.use_oof_interaction:
                self._oof_tr_inter = self._build_oof_interactions(
                    self._oof_tr_lr, self._oof_tr_svm)
                self.block_sizes["oof_interactions"] = self._oof_tr_inter.shape[1]
                self._log_oof_correlation(self._oof_tr_lr, self._oof_tr_svm)

        return self

    # ──────────────────────────────────────────────────────────
    # TRANSFORM
    # ──────────────────────────────────────────────────────────
    def transform(self, X, is_train=False):
        cw  = X["clean_word"]
        cc  = X["clean_char"]
        raw = X["text"]

        X_tfidf = hstack([
            self.tfidf_wL_uni.transform(cw), self.tfidf_wL_23.transform(cw),
            self.tfidf_wA_uni.transform(cw), self.tfidf_wA_23.transform(cw),
            self.tfidf_char.transform(cc),
        ])
        X_bm25    = hstack([self.bm25w.transform(cw), self.bm25c.transform(cc)])
        X_lexical = hstack([
            self.scaler_t.transform(X_tfidf),
            self.scaler_b.transform(X_bm25),
        ], format="csr")
        X_lex_sel = self._chi2_scaler.transform(X_lexical[:, self._chi2_cols])

        X_c23 = self._mi_sel23.transform(self.char_vec23.transform(cc))
        X_c45 = self._mi_sel45.transform(self.char_vec45.transform(cc))

        X_stats  = self._doc_stats(raw)
        X_sw     = self._stopword_stats(raw)
        X_morph  = self._morph_block(raw)
        X_minter = self._morph_interactions(raw)
        X_colloc = csr_matrix(self.colloc_vec.transform(cw)
                               @ self.colloc_scores[:, None])
        X_nb     = _nbsvm_transform(self._nb_char_v, self._nb_char_r, cc)

        if self.use_oof_lr:
            if is_train:
                X_lr  = self._oof_tr_lr
                X_svm = self._oof_tr_svm
            else:
                X_lrm  = self._build_lr_matrix(list(cw))
                X_svmm = self._build_svm_matrix(list(cc))
                X_lr   = csr_matrix(np.hstack(
                    [m.predict_proba(X_lrm)  for m in self._lr_models]
                ).astype(np.float32))
                X_svm  = csr_matrix(np.hstack(
                    [m.predict_proba(X_svmm) for m in self._svm_models]
                ).astype(np.float32))

            if self.use_oof_interaction:
                X_inter = (self._oof_tr_inter if is_train
                           else self._build_oof_interactions(X_lr, X_svm))
            else:
                X_inter = csr_matrix((len(raw), 0))
        else:
            X_lr    = csr_matrix((len(raw), 0))
            X_svm   = csr_matrix((len(raw), 0))
            X_inter = csr_matrix((len(raw), 0))

        return hstack([
            X_lex_sel,
            X_c23, X_c45,
            X_stats, X_sw, X_morph, X_minter,
            X_colloc, X_nb,
            X_lr, X_svm, X_inter,
        ], format="csr")

    # ──────────────────────────────────────────────────────────
    # UPDATE HELPERS
    # ──────────────────────────────────────────────────────────
    def update_chi2_pct(self, new_pct):
        """Ubah percentile chi2 dan re-apply ke rank tersimpan (tidak re-fit)."""
        if self._X_lexical_full_tr is None:
            raise ValueError("Belum di-fit.")
        self.chi2_pct    = new_pct
        self._chi2_k_eff = max(self.chi2_min,
                               int(self._X_lexical_full_tr.shape[1] * new_pct))
        self.chi2_k      = self._chi2_k_eff
        self._apply_chi2_from_rank()
        if hasattr(self, "block_sizes"):
            self.block_sizes["lexical"] = len(self._chi2_cols)
        print(f"[update] chi2_pct={new_pct:.2%} -> chi2_k={self._chi2_k_eff:,}")

    def update_mi_pct(self, new_pct23=None, new_pct45=None):
        """Ubah percentile MI dan re-apply ke rank tersimpan (tidak re-fit)."""
        if self._X_char23_tr is None:
            raise ValueError("Belum di-fit.")
        if new_pct23 is not None:
            self.mi_pct23    = new_pct23
            self._mi_k23_eff = max(self.mi_min23,
                                   int(self._X_char23_tr.shape[1] * new_pct23))
            self._mi_sel23.update_k(self._mi_k23_eff)
            if hasattr(self, "block_sizes"):
                self.block_sizes["char_mi23"] = self._mi_sel23.n_selected
            print(f"[update] mi_pct23={new_pct23:.2%} -> mi_k23={self._mi_k23_eff:,}")
        if new_pct45 is not None:
            self.mi_pct45    = new_pct45
            self._mi_k45_eff = max(self.mi_min45,
                                   int(self._X_char45_tr.shape[1] * new_pct45))
            self._mi_sel45.update_k(self._mi_k45_eff)
            if hasattr(self, "block_sizes"):
                self.block_sizes["char_mi45"] = self._mi_sel45.n_selected
            print(f"[update] mi_pct45={new_pct45:.2%} -> mi_k45={self._mi_k45_eff:,}")

    # backward-compat: update_chi2_k / update_mi_k tetap bisa dipakai
    def update_chi2_k(self, new_k):
        if not hasattr(self, "_chi2_rank"):
            raise ValueError("Belum di-fit.")

        # ✅ AUTO DETECT
        if isinstance(new_k, float) and 0 < new_k < 1:
            # treat as percentile
            k_eff = int(self._X_lexical_full_tr.shape[1] * new_k)
            print(f"[auto] Detected pct: {new_k} → k={k_eff}")
            self.chi2_k = k_eff
        else:
            # treat as absolute
            self.chi2_k = int(new_k)

        self._apply_chi2_from_rank()

        if hasattr(self, "block_sizes"):
            self.block_sizes["lexical"] = len(self._chi2_cols)

    def update_mi_k(self, new_k23=None, new_k45=None):
        if self._X_char23_tr is None:
            raise ValueError("Belum di-fit.")

        if new_k23 is not None:
            if isinstance(new_k23, float) and 0 < new_k23 < 1:
                k23 = int(self._X_char23_tr.shape[1] * new_k23)
                print(f"[auto] mi23 pct {new_k23} → k={k23}")
            else:
                k23 = int(new_k23)
            self._mi_sel23.update_k(k23)
            if hasattr(self, "block_sizes"):
                self.block_sizes["char_mi23"] = self._mi_sel23.n_selected

        if new_k45 is not None:
            if isinstance(new_k45, float) and 0 < new_k45 < 1:
                k45 = int(self._X_char45_tr.shape[1] * new_k45)
                print(f"[auto] mi45 pct {new_k45} → k={k45}")
            else:
                k45 = int(new_k45)
            self._mi_sel45.update_k(k45)
            if hasattr(self, "block_sizes"):
                self.block_sizes["char_mi45"] = self._mi_sel45.n_selected
    def _apply_chi2_from_rank(self):
        k_eff           = min(self.chi2_k, self._X_lexical_full_tr.shape[1])
        self._chi2_cols = self._chi2_rank[:k_eff]
        X_sel           = self._X_lexical_full_tr[:, self._chi2_cols]
        X_abs           = X_sel.copy()
        X_abs.data      = np.abs(X_abs.data)
        self._chi2_scaler = Chi2Scaler(pow=0.5).fit(X_abs, self._y_tr)

    @property
    def mi_info(self):
        """Ringkasan dimensi aktual -- untuk debug dan tesis."""
        if self._X_char23_tr is None:
            return {"status": "belum di-fit"}
        return {
            "chi2_pct":        self.chi2_pct,
            "chi2_k_eff":      self._chi2_k_eff,
            "mi_pct23":        self.mi_pct23,
            "mi_k23_eff":      self._mi_k23_eff,
            "char23_vocab":    self._X_char23_tr.shape[1],
            "char23_selected": self._mi_sel23.n_selected,
            "mi_pct45":        self.mi_pct45,
            "mi_k45_eff":      self._mi_k45_eff,
            "char45_vocab":    self._X_char45_tr.shape[1],
            "char45_selected": self._mi_sel45.n_selected,
        }

    # ──────────────────────────────────────────────────────────
    # STRUCTURAL FEATURES
    # ──────────────────────────────────────────────────────────
    def _morph_interactions(self, texts):
        """8 fitur interaksi morfologi."""
        n          = len(texts)
        M          = np.zeros((n, 8), dtype=np.float32)
        texts_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        for i, s in enumerate(texts_list):
            s    = str(s) if s is not None else ""
            toks = TOK_AR.findall(normalize_light(clean_noise(s)))
            tot  = max(len(toks), 1)
            M[i] = [
                sum(1 for j in range(len(toks)-1)
                    if toks[j] == "ما" and toks[j+1].startswith("ي")) / tot,
                sum(1 for t in toks if t.startswith("ب") and len(t) >= 3) / tot,
                sum(1 for j in range(len(toks)-1)
                    if toks[j] == "ما" and toks[j+1].endswith("ش")) / tot,
                sum(1 for j in range(len(toks)-1)
                    if toks[j] in {"رح", "راح"} and toks[j+1].startswith("ي")) / tot,
                sum(1 for t in toks if t in {"لسا","لسه","لساتو","لسات"}) / tot,
                sum(1 for j in range(len(toks)-1)
                    if toks[j] == "عم" and toks[j+1][0] in "يتنب") / tot,
                (sum(1 for t in toks if t in {"مش","موش","ميش"})
                 - sum(1 for t in toks if t == "مو")) / tot,
                (sum(1 for t in toks if t == "شو")
                 - sum(1 for t in toks if t in {"ايش","اش","ويش"})) / tot,
            ]
        return csr_matrix(M)

    def _morph_block(self, texts):
        """26 fitur morfologi."""
        n        = len(texts)
        M        = np.zeros((n, 26), dtype=np.float32)
        prefixes = ["ب","رح","ح","عم","ما","راح"]
        suffixes = ["ك","كن","كم","هم","نا","ني","ها","ه"]
        for i, s in enumerate(texts):
            toks  = TOK_AR.findall(normalize_light(clean_noise(s)))
            total = max(len(toks), 1)
            for j, p in enumerate(prefixes):
                M[i, j] = sum(t.startswith(p) for t in toks) / total
            off = len(prefixes)
            for j, sf in enumerate(suffixes):
                M[i, off+j] = sum(t.endswith(sf) for t in toks) / total
            off += len(suffixes)
            M[i, off] = sum(t == "ما" for t in toks) / total
            off += 1
            M[i, off]   = sum(t.startswith("ي") for t in toks) / total
            M[i, off+1] = sum(t.startswith("ت") for t in toks) / total
            M[i, off+2] = sum(t.startswith("ن") for t in toks) / total
            M[i, off+3] = sum(3 <= len(t) <= 4 for t in toks) / total
            off += 4
            lens = [len(t) for t in toks]
            M[i, off]   = sum(lens) / total
            M[i, off+1] = sum(l == 3 for l in lens) / total
            M[i, off+2] = sum(l == 4 for l in lens) / total
            M[i, off+3] = sum(l >= 6 for l in lens) / total
            off += 4
            M[i, off]   = int("لسا" in toks)
            M[i, off+1] = int("هلأ" in toks)
            M[i, off+2] = int("هلق" in toks)
        return csr_matrix(M)

    def _doc_stats(self, texts):
        """7 fitur: len_char, n_words, n_unique, avg_len, n_sent, ttr, fusha_ratio."""
        n = len(texts)
        M = np.zeros((n, 7), dtype=np.float32)
        fusha = {
            "الله","اللهم","الرسول","سبحان","الحمد","صلاه","ربنا","رحمه",
            "عز","وجل","تعالى","استغفر","صل","عليه","وسلم","بارك",
            "نحن","أنتم","أنتن","هؤلاء","هذان","هاتان","الذين",
            "اللاتي","اللواتي","إياك","إياكم","أولئك",
            "هل","لماذا","متى","أين","كيفما","بينما","حيثما","عندما",
            "لذلك","إذن","بيد","رغم","إلا","لكن","حتى","علما","كذلك",
            "يعتبر","تعتبر","سيتم","سوف","قام","قامت","أعلن","صرح",
            "مشيرا","أضاف","أكد","تجاه","بشأن","حول","وفقا","بناء",
            "يجب","يمكن","استطاع","أصبح","أمسى","بات","رسميا","عاجل",
        }
        for i, s in enumerate(texts):
            s      = s if isinstance(s, str) else ""
            words  = _WORD_RE.findall(s)
            nw     = len(words)
            nu     = len(set(w.lower() for w in words))
            n_sent = max(len(_SENT_SPLIT_RE.split(s)), 1)
            M[i]   = [
                len(s),
                nw,
                nu,
                sum(len(w) for w in words) / max(nw, 1),
                n_sent,
                nu / max(nw, 1),
                sum(1 for w in words if w in fusha) / max(nw, 1),
            ]
        return csr_matrix(M)

    def _stopword_stats(self, texts):
        n = len(texts)
        M = np.zeros((n, 2), dtype=np.float32)
        for i, s in enumerate(texts):
            toks = TOK_AR.findall(normalize_light(clean_noise(s)))
            sw   = sum(1 for t in toks if t in AR_STOP)
            M[i] = [sw, sw / max(len(toks), 1)]
        return csr_matrix(M)

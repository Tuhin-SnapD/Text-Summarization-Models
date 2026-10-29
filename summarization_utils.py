"""Shared, dependency-free summarization examples and evaluation helpers."""

from collections import Counter, defaultdict
from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parent
ARTICLE_PATH = ROOT / "article.txt"
REFERENCE_PATH = ROOT / "reference_summary.txt"

def load_article(path=ARTICLE_PATH):
    return Path(path).read_text(encoding="utf-8").strip()

def load_reference(path=REFERENCE_PATH):
    return Path(path).read_text(encoding="utf-8").strip()

def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

def words(text):
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.casefold())

STOP_WORDS = set("a an and are as at be been by for from had has have in is it its of on or that the their there these they this to was were which while with".split())

def content_words(text):
    return [word for word in words(text) if word not in STOP_WORDS]

def _vectors(items):
    docs = [Counter(content_words(item)) for item in items]
    n = max(1, len(docs))
    vocab = set().union(*(doc.keys() for doc in docs)) if docs else set()
    idf = {word: math.log((1 + n) / (1 + sum(word in doc for doc in docs))) + 1 for word in vocab}
    return [{word: count * idf[word] for word, count in doc.items()} for doc in docs]

def _cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0

def _pagerank(items, damping=0.85, iterations=60):
    vectors = _vectors(items)
    size = len(items)
    if not size:
        return []
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            similarity = _cosine(vectors[i], vectors[j])
            matrix[i][j] = matrix[j][i] = similarity
    scores = [1.0 / size] * size
    for _ in range(iterations):
        updated = [(1.0 - damping) / size] * size
        for source in range(size):
            total = sum(matrix[source])
            if total == 0:
                share = damping * scores[source] / size
                updated = [score + share for score in updated]
            else:
                for target in range(size):
                    updated[target] += damping * scores[source] * matrix[source][target] / total
        if max(abs(a - b) for a, b in zip(scores, updated)) < 1e-9:
            scores = updated
            break
        scores = updated
    return scores

def _select(items, scores, count=3, unique=True):
    if not items:
        return ""
    ranked = sorted(range(len(items)), key=lambda i: (-scores[i], i))
    chosen = ranked[:max(1, min(count, len(items)))]
    chosen.sort()
    result = []
    seen = set()
    for index in chosen:
        key = items[index].casefold()
        if not unique or key not in seen:
            result.append(items[index])
            seen.add(key)
    return " ".join(result)

def _frequency_scores(items):
    frequencies = Counter(content_words(" ".join(items)))
    maximum = max(frequencies.values(), default=1)
    return [sum(frequencies[word] / maximum for word in content_words(sentence)) / max(1, len(content_words(sentence))) for sentence in items]

def _tfidf_scores(items):
    vectors = _vectors(items)
    return [sum(vector.values()) / max(1, len(content_words(sentence))) for vector, sentence in zip(vectors, items)]

def _sum_basic(items, count):
    probabilities = Counter(content_words(" ".join(items)))
    total = max(1, sum(probabilities.values()))
    probabilities = {word: value / total for word, value in probabilities.items()}
    selected = []
    for _ in range(min(count, len(items))):
        candidates = []
        for index, sentence in enumerate(items):
            if index in selected:
                continue
            tokens = content_words(sentence)
            score = sum(probabilities.get(word, 0.0) for word in tokens) / max(1, len(tokens))
            candidates.append((score, -index, index))
        if not candidates:
            break
        index = max(candidates)[2]
        selected.append(index)
        for word in content_words(items[index]):
            probabilities[word] = probabilities.get(word, 0.0) ** 2
    return " ".join(items[index] for index in sorted(selected))

def _cluster_representatives(items, count):
    vectors = _vectors(items)
    if len(items) <= count:
        return " ".join(items)
    # Deterministic farthest-first representatives; avoids random, version-dependent KMeans behavior.
    centers = [0]
    while len(centers) < count:
        candidate = max((i for i in range(len(items)) if i not in centers), key=lambda i: (min(1 - _cosine(vectors[i], vectors[c]) for c in centers), -i))
        centers.append(candidate)
    centers.sort()
    return " ".join(items[index] for index in centers)

def _connected_dominating_summary(items, count):
    """Greedily cover a sentence-similarity graph, adding paths to keep it connected."""
    vectors = _vectors(items)
    size = len(items)
    if size <= count:
        return " ".join(items)
    adjacency = [set() for _ in items]
    for i in range(size):
        for j in range(i + 1, size):
            if _cosine(vectors[i], vectors[j]) >= 0.08:
                adjacency[i].add(j)
                adjacency[j].add(i)
    selected = [max(range(size), key=lambda i: (len(adjacency[i]), -i))]
    dominated = set(selected) | adjacency[selected[0]]
    while len(dominated) < size and len(selected) < count:
        candidate = max((i for i in range(size) if i not in dominated), key=lambda i: (len(adjacency[i] - dominated), -i))
        # Find a shortest path from any selected node to the candidate.
        queue = [(node, [node]) for node in selected]
        visited = set(selected)
        path = None
        while queue and path is None:
            node, route = queue.pop(0)
            if node == candidate:
                path = route
                break
            for neighbor in sorted(adjacency[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, route + [neighbor]))
        additions = [candidate] if path is None else [node for node in path if node not in selected]
        for node in additions:
            if len(selected) >= count:
                break
            selected.append(node)
            dominated.add(node)
            dominated.update(adjacency[node])
    selected = sorted(selected[:count])
    return " ".join(items[index] for index in selected)

def _budgeted_summary(items, scores, word_budget=75):
    ranked = sorted(range(len(items)), key=lambda i: (-scores[i], i))
    chosen = []
    used = 0
    for index in ranked:
        cost = len(words(items[index]))
        if used + cost <= word_budget:
            chosen.append(index)
            used += cost
    if not chosen and items:
        chosen = [min(range(len(items)), key=lambda i: len(words(items[i])))]
    return " ".join(items[index] for index in sorted(chosen))

def _lsa_scores(items):
    """Rank sentences by projection onto the leading term-space component."""
    vectors = _vectors(items)
    vocabulary = sorted(set().union(*(vector.keys() for vector in vectors))) if vectors else []
    if not vocabulary:
        return [0.0] * len(items)
    columns = [{word: vector.get(word, 0.0) for word in vocabulary} for vector in vectors]
    direction = {word: 1.0 / math.sqrt(len(vocabulary)) for word in vocabulary}
    for _ in range(35):
        projected = [sum(vector[word] * direction[word] for word in vocabulary) for vector in columns]
        updated = {word: sum(columns[i][word] * projected[i] for i in range(len(columns))) for word in vocabulary}
        norm = math.sqrt(sum(value * value for value in updated.values()))
        if norm == 0:
            return [0.0] * len(items)
        direction = {word: value / norm for word, value in updated.items()}
    return [abs(sum(vector[word] * direction[word] for word in vocabulary)) for vector in columns]

def summarize(text, method="tfidf", sentence_count=3):
    """Run one of the repository's classical, extractive demonstrations."""
    items = sentences(text)
    key = method.casefold().replace(" ", "_").replace("-", "_").replace("'", "")
    if not items:
        return ""
    if key in {"nltk", "luhn", "luhns_model"}:
        scores = _frequency_scores(items)
        if key.startswith("luhn") or key == "luhn":
            scores = [score * math.sqrt(max(1, len(content_words(s)))) for score, s in zip(scores, items)]
        return _select(items, scores, sentence_count)
    if key in {"tfidf", "sentence_ranking"}:
        return _select(items, _tfidf_scores(items), sentence_count)
    if key in {"textrank", "lexrank", "gensim", "graph_based_summary"}:
        return _select(items, _pagerank(items), sentence_count)
    if key == "lsa":
        return _select(items, _lsa_scores(items), sentence_count)
    if key in {"sumbasic", "sum_basic"}:
        return _sum_basic(items, sentence_count)
    if key in {"k_clustering", "cluster_based"}:
        return _cluster_representatives(items, sentence_count)
    if key == "sumy":
        scores = _pagerank(items)
        selected = []
        vectors = _vectors(items)
        for index in sorted(range(len(items)), key=lambda i: (-scores[i], i)):
            if all(_cosine(vectors[index], vectors[other]) < 0.82 for other in selected):
                selected.append(index)
            if len(selected) == sentence_count:
                break
        return " ".join(items[i] for i in sorted(selected))
    if key in {"connected_dominating_set", "cds"}:
        return _connected_dominating_summary(items, sentence_count)
    if key in {"compressive_document_summarization", "cds_compressive"}:
        return _budgeted_summary(items, _tfidf_scores(items))
    if key in {"pegasus", "pegasus_single", "pegasus_multi", "pegasus_models"}:
        raise RuntimeError("Pegasus inference is not available in the standard-library runtime; use a Transformers-enabled environment and follow the notebook's model instructions.")
    if key in {"bert", "bart", "t5", "gpt_2", "distilbart", "ai_transformer_pipeline", "transformer_pipeline"}:
        raise RuntimeError("Pretrained-model inference requires optional model dependencies and downloaded weights; use a Transformers-enabled environment.")
    # The proposed-architecture graph demo is a simple lexical-centrality baseline.
    if key in {"graph_based", "graph_based_summarisation"}:
        return _select(items, _pagerank(items), sentence_count)
    raise ValueError(f"Unknown summarization method: {method}")

def summarize_pretrained(text, model_id, task="summarization", sentence_count=3):
    """Run a Hugging Face pipeline when its optional dependencies and weights are available."""
    try:
        from transformers import pipeline
        generator = pipeline(task, model=model_id)
        if task == "text-generation":
            result = generator(text, max_new_tokens=80, do_sample=False, truncation=True)
        else:
            result = generator(text, max_length=120, min_length=30, do_sample=False, truncation=True)
    except (ImportError, OSError) as exc:
        raise RuntimeError("Transformers/PyTorch or the requested model weights are unavailable; install the optional requirements and retry.") from exc
    return result[0].get("summary_text", result[0].get("generated_text", "")).strip()

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]

def _lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j-1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return previous[-1]

def evaluate_summary(prediction, reference):
    """Return deterministic ROUGE-1/2/L F1 and smoothed single-reference BLEU-4."""
    pred = words(prediction)
    ref = words(reference)
    result = {}
    for label, n in (("rouge1_f1", 1), ("rouge2_f1", 2)):
        pgrams, rgrams = Counter(_ngrams(pred, n)), Counter(_ngrams(ref, n))
        overlap = sum((pgrams & rgrams).values())
        precision = overlap / max(1, sum(pgrams.values()))
        recall = overlap / max(1, sum(rgrams.values()))
        result[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    overlap_lcs = _lcs_length(pred, ref)
    precision_lcs = overlap_lcs / max(1, len(pred))
    recall_lcs = overlap_lcs / max(1, len(ref))
    result["rougeL_f1"] = 2 * precision_lcs * recall_lcs / (precision_lcs + recall_lcs) if precision_lcs + recall_lcs else 0.0
    if not pred or not ref:
        result["bleu4"] = 0.0
        return result
    log_precision = 0.0
    for n in range(1, 5):
        predicted = Counter(_ngrams(pred, n))
        reference_counts = Counter(_ngrams(ref, n))
        overlap = sum((predicted & reference_counts).values())
        log_precision += math.log((overlap + 1) / (sum(predicted.values()) + 1))
    brevity = min(1.0, math.exp(1 - len(ref) / len(pred)))
    result["bleu4"] = brevity * math.exp(log_precision / 4)
    return result

def format_scores(scores):
    return ", ".join(f"{name}={value:.3f}" for name, value in scores.items())


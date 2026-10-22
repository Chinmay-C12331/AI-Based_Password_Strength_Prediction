"""Reference attackers: Frequency-rank Dictionary and Character n-gram Markov with Monte Carlo estimation."""

import math
import bisect
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DictionaryAttacker:
    """Frequency-rank dictionary attacker.

    Sorts unique passwords by count descending; ties receive the mean rank of their tie group.
    """

    def __init__(self):
        self.ranks: Dict[str, float] = {}

    def fit(self, data: List[Tuple[str, int]]) -> "DictionaryAttacker":
        """Fit dictionary attacker on a list of (password, count) tuples."""
        # Sort descending by count
        sorted_items = sorted(data, key=lambda x: x[1], reverse=True)

        self.ranks = {}
        i = 0
        n = len(sorted_items)
        while i < n:
            current_count = sorted_items[i][1]
            j = i
            while j < n and sorted_items[j][1] == current_count:
                j += 1
            # Tie group spans indices [i, j-1] (1-indexed ranks from i+1 to j)
            mean_rank = ( (i + 1) + j ) / 2.0
            for k in range(i, j):
                self.ranks[sorted_items[k][0]] = mean_rank
            i = j
        return self

    def guess_number(self, pw: str) -> Optional[float]:
        """Return the rank if present in the dictionary, otherwise None."""
        return self.ranks.get(pw, None)


class MarkovAttacker:
    """Character n-gram Markov attacker with Witten-Bell smoothing and Monte Carlo guess estimation."""

    START_CHAR = "\x01"
    END_CHAR = "\x02"

    def __init__(self, order: int = 4):
        self.order = order
        self.context_len = order - 1
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: Dict[str, int] = defaultdict(int)
        self.vocab: set = {self.END_CHAR}
        self.unigram_counts: Dict[str, int] = defaultdict(int)
        self.total_unigrams = 0

        # Monte Carlo state (Dell'Amico and Filippone CCS 2015)
        # We store ONLY sorted negative log probabilities and prefix sums (NO strings, per Project Rule 3)
        self.sorted_neg_log_probs: List[float] = []
        self.prefix_mc_sums: List[float] = []
        self.mc_fitted = False

    def fit(self, passwords: List[Tuple[str, int]]) -> "MarkovAttacker":
        """Fit transition counts from password tuples with counts."""
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_totals = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.total_unigrams = 0
        self.vocab = {self.END_CHAR}

        for pw, cnt in passwords:
            for ch in pw:
                self.vocab.add(ch)
                self.unigram_counts[ch] += cnt
                self.total_unigrams += cnt
            self.unigram_counts[self.END_CHAR] += cnt
            self.total_unigrams += cnt

            # Padded string with start tokens and end token
            padded = (self.START_CHAR * self.context_len) + pw + self.END_CHAR
            for i in range(self.context_len, len(padded)):
                char = padded[i]
                # Collect n-gram contexts of variable lengths 1..context_len
                for ctx_len in range(1, self.context_len + 1):
                    ctx = padded[i - ctx_len : i]
                    self.counts[ctx][char] += cnt
                    self.context_totals[ctx] += cnt

        return self

    def _prob_char(self, ctx: str, char: str) -> float:
        """Witten-Bell smoothed conditional probability P(char | ctx)."""
        vocab_size = max(len(self.vocab), 1)
        uniform_fallback = 1.0 / vocab_size

        if not ctx:
            # Unigram with Laplace smoothing
            return (self.unigram_counts.get(char, 0) + 1.0) / (self.total_unigrams + vocab_size)

        n_ctx = self.context_totals.get(ctx, 0)
        if n_ctx == 0:
            # Back off to shorter context
            return self._prob_char(ctx[1:], char)

        # Number of unique characters following ctx
        t_ctx = len(self.counts[ctx])
        lambda_ctx = n_ctx / (n_ctx + t_ctx)

        seen_count = self.counts[ctx].get(char, 0)
        p_higher = (seen_count / n_ctx) if n_ctx > 0 else 0.0
        p_lower = self._prob_char(ctx[1:], char)

        return lambda_ctx * p_higher + (1.0 - lambda_ctx) * p_lower

    def log_prob(self, pw: str) -> float:
        """Compute log probability ln P(pw) including the end symbol."""
        padded = (self.START_CHAR * self.context_len) + pw + self.END_CHAR
        total_log_p = 0.0

        for i in range(self.context_len, len(padded)):
            ctx = padded[i - self.context_len : i]
            char = padded[i]
            p = self._prob_char(ctx, char)
            total_log_p += math.log(max(p, 1e-30))

        return total_log_p

    def sample_password(self, rng: random.Random, max_len: int = 64) -> str:
        """Ancestral sampling of a single password from the Markov model."""
        ctx = self.START_CHAR * self.context_len
        vocab_list = list(self.vocab)
        chars = []

        for _ in range(max_len):
            weights = [self._prob_char(ctx, ch) for ch in vocab_list]
            chosen = rng.choices(vocab_list, weights=weights, k=1)[0]
            if chosen == self.END_CHAR:
                break
            chars.append(chosen)
            ctx = (ctx + chosen)[-self.context_len :]

        return "".join(chars)

    def fit_monte_carlo(self, n_samples: int = 200000, seed: int = 42) -> "MarkovAttacker":
        """Compute Monte Carlo guess estimator table (Dell'Amico & Filippone 2015).

        PROJECT RULE 3 ETHICS ENFORCEMENT:
        Sampled password strings are DISCARDED immediately after calculating log_prob.
        Only sorted negative log probabilities and prefix sums are retained.
        """
        rng = random.Random(seed)
        log_probs = []

        # Draw n_samples
        for _ in range(n_samples):
            sampled_pw = self.sample_password(rng=rng, max_len=64)
            lp = self.log_prob(sampled_pw)
            log_probs.append(lp)
            # sampled_pw discarded here

        # Sort by probability descending <=> -log_prob ascending
        neg_lps = sorted([-lp for lp in log_probs])
        n = len(neg_lps)

        # Precompute prefix sums of 1 / (n * P(x_i)) = exp(-log_prob) / n
        prefix = [0.0] * (n + 1)
        running_sum = 0.0
        for i, nlp in enumerate(neg_lps):
            # 1 / (n * P) = exp(nlp) / n
            term = math.exp(min(nlp, 100.0)) / float(n)
            running_sum += term
            prefix[i + 1] = running_sum

        self.sorted_neg_log_probs = neg_lps
        self.prefix_mc_sums = prefix
        self.mc_fitted = True
        return self

    def guess_number(self, pw: str) -> float:
        """Estimate Monte Carlo guess number G_MC(pw) via binary search."""
        if not self.mc_fitted or not self.sorted_neg_log_probs:
            # Fallback direct probability inverse if MC table not generated
            lp = self.log_prob(pw)
            return max(1.0, math.exp(min(-lp, 36.0)))

        lp = self.log_prob(pw)
        query_nlp = -lp

        # Find index where neg_log_prob >= query_nlp (i.e. P(sample) > P(pw))
        idx = bisect.bisect_left(self.sorted_neg_log_probs, query_nlp)
        est = self.prefix_mc_sums[idx]

        if est < 1.0:
            est = 1.0

        # If query is beyond sampled distribution, extrapolate with 1/P(pw)
        if idx == len(self.sorted_neg_log_probs):
            extrap = math.exp(min(query_nlp, 40.0))
            est = max(est, extrap)

        return min(est, 1e16)

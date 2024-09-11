# src/evaluation/evaluation.py

from rouge_score import rouge_scorer
import sacrebleu

class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_rouge(self, reference, prediction):
        scores = self.rouge_scorer.score(reference, prediction)
        return scores

    def compute_bleu(self, reference, prediction):
        bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
        return bleu.score

    def evaluate(self, references, predictions):
        rouge_scores = [self.compute_rouge(ref, pred) for ref, pred in zip(references, predictions)]
        bleu_scores = [self.compute_bleu(ref, pred) for ref, pred in zip(references, predictions)]
        return rouge_scores, bleu_scores

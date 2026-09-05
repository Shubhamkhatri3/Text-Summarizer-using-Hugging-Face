## Model Evaluation

The fine-tuned T5 model was evaluated against the original `t5-base` model using ROUGE and BERTScore metrics.

| Metric | Fine-tuned T5 | T5-base | Improvement |
|---|---:|---:|---:|
| **ROUGE-1** | **0.4973** | 0.2715 | **+0.2258** |
| **ROUGE-2** | **0.2620** | 0.0939 | **+0.1681** |
| **ROUGE-L** | **0.4170** | 0.2132 | **+0.2037** |
| **BERTScore Precision** | **0.9195** | 0.8438 | **+0.0757** |
| **BERTScore Recall** | **0.9158** | 0.8813 | **+0.0345** |
| **BERTScore F1** | **0.9175** | 0.8618 | **+0.0557** |
| **Average Summary Length (words)** | **15.76** | 25.18 | — |
| **Reference Summary Length (words)** | 18.62 | — | — |

### Key Findings

- The fine-tuned model significantly outperformed the original `t5-base` across all ROUGE metrics.
- **ROUGE-1:** 0.4973 vs 0.2715
- **ROUGE-2:** 0.2620 vs 0.0939
- **ROUGE-L:** 0.4170 vs 0.2132
- **BERTScore F1:** 0.9175 vs 0.8618
- The fine-tuned model generated more concise summaries, averaging **15.76 words**, compared with **25.18 words** for the base model.

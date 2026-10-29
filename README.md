# Text Summarization Models - A Comprehensive Educational Repository

This repository contains Python implementations of text summarization methods, ranging from basic statistical approaches to advanced transformer-based architectures. It serves as an educational resource for understanding these techniques and their practical applications. The current curated layout contains **30 notebooks: 18 basic/advanced methods and 12 Pegasus checkpoint examples**. The earlier README count of 31 referred to the prior repository organization.

## 📊 Project Overview

Text summarization is the process of generating a shorter version of a longer text while preserving its most important information. This repository demonstrates both **extractive** (selecting key sentences) and **abstractive** (generating new content) summarization approaches.

### Evaluation Metrics
All models are evaluated using:
- **ROUGE Score** (Recall-Oriented Understudy for Gisting Evaluation)
  - Precision: Measures how much of the generated summary is relevant
  - Recall: Measures how much of the reference summary is captured
  - F1-Score: Harmonic mean of precision and recall
- **BLEU Score** (Bilingual Evaluation Understudy)
  - Measures the quality of machine-generated text against reference text

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Python 3.9+ required
python --version

# Install core dependencies
pip install -r requirements-models.txt
```

For a complete setup of all notebooks and model checkpoints, see [How to Install All Models](#how-to-install-all-models).

### Basic Usage Example
```python
# Example: Using BERT for summarization
from summarizer import Summarizer

model = Summarizer()
text = "Your long text here..."
summary = model(text, num_sentences=3)
print(summary)
```

### Running a Specific Model
```bash
# Navigate to the model directory
cd "Basic and Advanced Text Summarisation Models"

# Run any Jupyter notebook
jupyter notebook BERT.ipynb
```

## 📊 Model Performance Comparison Table

These are the latest saved notebook results for the shared NOAA article and `reference_summary.txt`. Most basic/advanced notebooks report their existing ROUGE-1 F1 and unsmoothed NLTK sentence BLEU; Graph-Based Summary uses the shared ROUGE/BLEU helper. BLEU values from different methods are not directly comparable. Pegasus checkpoints are listed in the separate table below using the shared evaluator.

| Model | Type | ROUGE F1 | BLEU | Pros | Cons | Best For |
|-------|------|----------|------|------|------|----------|
| **AI Transformer Pipeline** | Abstractive | 0.291 | 0.000 | Simple pipeline interface | Requires downloaded weights | Quick transformer summaries |
| **BART** | Abstractive | 0.232 | 0.000 | Coherent abstractive output | Large model, slow on CPU | News summaries |
| **BERT** | Extractive | 0.355 | 0.064 | Context-aware sentence selection | Extractive and resource-heavy | Context-heavy documents |
| **Compressive Document Summarization** | Extractive | 0.264 | 0.000 | Concise sentence selection | Can lose context | Length-constrained summaries |
| **Connected Dominating Set** | Extractive | 0.180 | 0.000 | Graph-based selection | Limited semantic coverage | Graph summarization study |
| **DistilBART** | Abstractive | 0.291 | 0.000 | Smaller and faster than BART | May produce noisy output | CPU-friendly transformer demos |
| **GPT-2** | Abstractive | 0.301 | 0.039 | Generative and flexible | Less controlled output | Language-model experiments |
| **Graph-Based Summary** | Extractive | 0.321 | 0.081 | Combines graph and lexical signals | Limited generation ability | Graph-based baselines |
| **K-Clustering** | Extractive | 0.359 | 0.861 | Groups related sentences | Sensitive to clustering choices | Topic-based summaries |
| **LexRank** | Extractive | 0.400 | 0.000 | Strong centrality-based selection | Extractive only | Long articles |
| **LSA** | Extractive | 0.140 | 0.000 | Captures latent term structure | Can miss sentence context | Semantic analysis demos |
| **Luhn's Model** | Extractive | 0.346 | 0.000 | Simple frequency-based baseline | Limited semantic understanding | Introductory NLP |
| **NLTK** | Extractive | 0.390 | 0.000 | Clear, inspectable algorithm | Short outputs yield zero unsmoothed BLEU | Beginners |
| **SumBasic** | Extractive | 0.136 | 0.000 | Simple probabilistic selection | Repetition and coverage limits | Single-document baselines |
| **SUMY** | Extractive | 0.119 | 0.000 | Multiple summarization algorithms | Output depends on parser/settings | Summarization toolkit demos |
| **T5** | Abstractive | 0.268 | 0.000 | Text-to-text architecture | Resource-intensive | Abstractive summaries |
| **TextRank** | Extractive | 0.269 | 0.000 | Unsupervised graph ranking | Limited context | General-purpose baseline |
| **TF-IDF** | Extractive | 0.357 | 0.000 | Fast and interpretable | Lexical rather than semantic | Keyword-focused summaries |

### Pegasus Checkpoint Evaluation Table

Current Pegasus values use the same shared article/reference and shared ROUGE-1/2/L F1 plus smoothed BLEU-4 evaluator. Original values are retained for provenance; they used a different source article and comparison target and are not directly comparable.

| Checkpoint | Current ROUGE-1 F1 | Current ROUGE-2 F1 | Current ROUGE-L F1 | Current BLEU-4 | Original ROUGE F1 | Original BLEU |
|---|---:|---:|---:|---:|---:|---:|
| XSum | 0.202 | 0.046 | 0.157 | 0.012 | 0.085 | 0.462 |
| CNN/DailyMail | 0.186 | 0.063 | 0.103 | 0.033 | 0.517 | 0.796 |
| Newsroom | 0.339 | 0.130 | 0.222 | 0.084 | 0.690 | 0.863 |
| MultiNews | 0.276 | 0.093 | 0.195 | 0.062 | 0.557 | 0.510 |
| Gigaword | 0.120 | 0.049 | 0.120 | 0.003 | 0.142 | 0.437 |
| WikiHow | 0.176 | 0.022 | 0.176 | 0.011 | 0.209 | 0.420 |
| Reddit-TIFU | 0.356 | 0.159 | 0.244 | 0.033 | 0.311 | 0.887 |
| BigPatent | 0.295 | 0.023 | 0.250 | 0.009 | 0.261 | 0.790 |
| ArXiv | 0.236 | 0.048 | 0.142 | 0.048 | 0.291 | 0.074 |
| PubMed | 0.176 | 0.038 | 0.101 | 0.037 | 0.349 | 0.150 |
| AESLC | 0.075 | 0.000 | 0.075 | 0.001 | 0.233 | 0.917 |
| BillSum | 0.340 | 0.083 | 0.204 | 0.047 | 0.449 | 0.924 |

Machine-readable results from the latest full run are in [`evaluation_results.csv`](evaluation_results.csv).

## 🏗️ Model Categories

### 1. Basic Statistical Models
These models use fundamental NLP techniques and statistical approaches:

#### **NLTK-based Summarization** ([`NLTK.ipynb`](<Basic and Advanced Text Summarisation Models/NLTK.ipynb>))
- **Approach**: Frequency-based sentence scoring using NLTK library
- **Pros**: Simple implementation, language-independent, well-documented
- **Cons**: Limited accuracy for complex texts, resource-intensive
- **Performance**: ROUGE-1 F1: 0.390, unsmoothed BLEU: 0.000
- **Use Case**: Educational purposes, simple documents

#### **TF-IDF Summarization** ([`TFIDF.ipynb`](<Basic and Advanced Text Summarisation Models/TFIDF.ipynb>))
- **Approach**: Term Frequency-Inverse Document Frequency weighting
- **Pros**: Effective for identifying important terms, widely used
- **Cons**: May miss context, struggles with technical jargon
- **Performance**: Evaluated using standard TF-IDF scoring
- **Use Case**: Keyword extraction, document classification

#### **SumBasic Algorithm** ([`SumBasic.ipynb`](<Basic and Advanced Text Summarisation Models/SumBasic.ipynb>))
- **Approach**: Probabilistic sentence selection based on word frequency
- **Pros**: Simple implementation, good for single documents
- **Cons**: Limited coverage, may lack coherence
- **Performance**: ROUGE-1 F1: 0.136, unsmoothed BLEU: 0.000
- **Use Case**: Single document summarization

### 2. Graph-Based Models
These models represent text as graphs and use centrality algorithms:

#### **TextRank** ([`TextRank.ipynb`](<Basic and Advanced Text Summarisation Models/TextRank.ipynb>))
- **Approach**: Graph-based ranking using sentence similarity
- **Pros**: Automatic, unbiased, consistent results
- **Cons**: Limited context understanding, may miss nuances
- **Performance**: ROUGE-1 F1: 0.269, unsmoothed BLEU: 0.000
- **Use Case**: General purpose summarization

#### **LexRank** ([`LexRank.ipynb`](<Basic and Advanced Text Summarisation Models/LexRank.ipynb>))
- **Approach**: Graph-based lexical centrality using cosine similarity
- **Pros**: Handles complex texts, good performance on benchmarks
- **Cons**: Limited coverage, extractive only
- **Performance**: ROUGE-1 F1: 0.400, unsmoothed BLEU: 0.000
- **Use Case**: Long documents, multi-document summarization

#### **Connected Dominating Set** ([`Connected_Dominating_Set.ipynb`](<Basic and Advanced Text Summarisation Models/Connected_Dominating_Set.ipynb>))
- **Approach**: Graph theory-based sentence selection
- **Pros**: Theoretical foundation, systematic approach
- **Cons**: Computational complexity, may not capture semantics
- **Use Case**: Research applications

### 3. Matrix Decomposition Models
These models use mathematical techniques to identify important content:

#### **LSA (Latent Semantic Analysis)** ([`LSA.ipynb`](<Basic and Advanced Text Summarisation Models/LSA.ipynb>))
- **Approach**: Singular Value Decomposition for semantic relationships
- **Pros**: Captures semantic relationships, good for multi-document
- **Cons**: Requires large training data, limited coverage
- **Performance**: ROUGE-1 F1: 0.140, unsmoothed BLEU: 0.000
- **Use Case**: Multi-document summarization, semantic analysis

#### **Sentence Ranking** (overlapping with the TF-IDF sentence-scoring notebook and retained in the original-source archive)
- **Approach**: Matrix-based sentence importance scoring
- **Pros**: Systematic ranking, handles large documents
- **Cons**: May miss context, computational expense
- **Use Case**: Large document processing

### 4. Clustering-Based Models
These models group similar content together:

#### **K-Clustering** ([`K_Clustering.ipynb`](<Basic and Advanced Text Summarisation Models/K_Clustering.ipynb>))
- **Approach**: K-means clustering of sentences
- **Pros**: Groups similar content, systematic approach
- **Cons**: Requires parameter tuning, may lose context
- **Use Case**: Topic-based summarization

#### **Cluster-Based Summarization** (overlapping with K-Clustering and retained in the original-source archive)
- **Approach**: Hierarchical clustering for document structure
- **Pros**: Preserves document structure, systematic
- **Cons**: Computational complexity, parameter sensitivity
- **Use Case**: Structured document summarization

### 5. Advanced NLP Models
These models use sophisticated NLP techniques:

#### **Luhn's Model** ([`Luhn's_Model.ipynb`](<Basic and Advanced Text Summarisation Models/Luhn's_Model.ipynb>))
- **Approach**: Classic statistical approach using word frequency
- **Pros**: Historical significance, simple implementation
- **Cons**: Basic approach, limited effectiveness
- **Use Case**: Educational purposes, historical context

#### **Compressive Document Summarization** ([`Compressive_Document_Summarization.ipynb`](<Basic and Advanced Text Summarisation Models/Compressive_Document_Summarization.ipynb>))
- **Approach**: Sentence compression and selection
- **Pros**: More concise summaries, preserves key information
- **Cons**: Complex implementation, may lose context
- **Use Case**: Length-constrained summaries

### 6. Transformer-Based Models
These are state-of-the-art deep learning models:

#### **BERT Extractive Summarization** ([`BERT.ipynb`](<Basic and Advanced Text Summarisation Models/BERT.ipynb>))
- **Approach**: Pre-trained BERT model for sentence extraction
- **Pros**: High accuracy, captures context well
- **Cons**: Limited to extractive, depends on pre-trained models
- **Performance**: ROUGE-1 F1: 0.355, unsmoothed BLEU: 0.064
- **Use Case**: Context-heavy documents

#### **BART** ([`BART.ipynb`](<Basic and Advanced Text Summarisation Models/BART.ipynb>))
- **Approach**: Bidirectional and Auto-Regressive Transformer
- **Pros**: State-of-the-art performance, high accuracy
- **Cons**: Resource-intensive, large model size
- **Performance**: ROUGE-1 F1: 0.232, unsmoothed BLEU: 0.000
- **Use Case**: High-quality abstractive summaries

#### **T5** ([`T5.ipynb`](<Basic and Advanced Text Summarisation Models/T5.ipynb>))
- **Approach**: Text-To-Text Transfer Transformer
- **Pros**: High accuracy, customizable, multilingual
- **Cons**: Resource-intensive, technical complexity
- **Performance**: ROUGE-1 F1: 0.268, unsmoothed BLEU: 0.000
- **Use Case**: Production systems, multilingual applications

#### **GPT-2** ([`GPT_2.ipynb`](<Basic and Advanced Text Summarisation Models/GPT_2.ipynb>))
- **Approach**: Generative Pre-trained Transformer 2
- **Pros**: Large language model, contextual understanding
- **Cons**: Lack of control, high computational costs
- **Performance**: ROUGE-1 F1: 0.301, unsmoothed BLEU: 0.039
- **Use Case**: Research, creative summarization

#### **DistilBART** ([`Distilbart.ipynb`](<Basic and Advanced Text Summarisation Models/Distilbart.ipynb>))
- **Approach**: Distilled version of BART for efficiency
- **Pros**: Faster inference, smaller model size
- **Cons**: Slightly lower accuracy than full BART
- **Use Case**: Real-time applications

### 7. Specialized Models

#### **Gensim** (the original example is retained in the original-source archive; it overlapped with the graph-based sentence-ranking examples)
- **Approach**: Topic modeling and document similarity
- **Pros**: Good for topic-based summarization
- **Cons**: May miss specific details
- **Use Case**: Topic extraction, document similarity

#### **SUMY** ([`SUMY.ipynb`](<Basic and Advanced Text Summarisation Models/SUMY.ipynb>))
- **Approach**: Multi-document summarization toolkit
- **Pros**: Specialized for multi-document scenarios
- **Cons**: Limited to specific use cases
- **Use Case**: Multi-document summarization

#### **AI Transformer Pipeline** ([`AI_Transformer_pipeline.ipynb`](<Basic and Advanced Text Summarisation Models/AI_Transformer_pipeline.ipynb>))
- **Approach**: Custom transformer-based pipeline
- **Pros**: Customizable architecture, modern approach
- **Cons**: Complex implementation, requires expertise
- **Use Case**: Custom applications, research

## 🚀 Advanced Architecture Models

### Pegasus Models (`Pegasus Checkpoint Models/`)

The repository includes **12 different Pegasus model implementations**:

#### **Single Document Pegasus** ([`Pegasus_XSum.ipynb`](<Pegasus Checkpoint Models/Pegasus_XSum.ipynb>))
- **Approach**: Single document abstractive summarization
- **Pros**: High-quality summaries, generalization capability
- **Cons**: Large computational requirements, limited control
- **Performance**: ROUGE-1 F1: 0.202, BLEU-4: 0.012
- **Use Case**: Single document abstractive summarization

#### **Multi-Document Pegasus** ([`Pegasus_MultiNews.ipynb`](<Pegasus Checkpoint Models/Pegasus_MultiNews.ipynb>))
- **Approach**: Multi-document summarization
- **Pros**: Handles multiple documents, comprehensive coverage
- **Cons**: Complex processing, resource-intensive
- **Use Case**: Multi-document scenarios

#### **All 12 Pegasus Models** ([12 checkpoint-specific notebooks](<Pegasus Checkpoint Models/>))
Includes implementations of:
- **Pegasus-XSUM**: BBC News summarization
- **Pegasus-CNN/DailyMail**: News article summarization
- **Pegasus-Newsroom**: Newsroom dataset
- **Pegasus-MultiNews**: Multi-document news summarization
- **Pegasus-Gigaword**: Headline generation
- **Pegasus-WikiHow**: How-to article summarization
- **Pegasus-Reddit-TIFU**: Reddit story summarization
- **Pegasus-BigPatent**: Patent document summarization
- And more specialized variants

#### **Graph-Based Summary** ([`Graph_Based_Summary.ipynb`](<Basic and Advanced Text Summarisation Models/Graph_Based_Summary.ipynb>))
- **Approach**: Novel graph-based approach using LCS, TF-IDF, and matrix methods
- **Pros**: Combines multiple techniques, systematic approach
- **Cons**: Complex implementation, computational expense
- **Performance**: ROUGE-1 F1: 0.321, BLEU-4: 0.081
- **Use Case**: Research, novel approaches

## 📈 Performance Analysis

### Key Insights from Results:

1. **Highest Basic/Advanced ROUGE-1 F1 in this run**: LexRank scored 0.400.
2. **Highest Basic/Advanced notebook BLEU in this run**: K-Clustering scored 0.861; BLEU implementations vary between notebooks.
3. **Pegasus results**: Use the separate shared-evaluator table above; do not compare its smoothed BLEU-4 directly with unsmoothed notebook BLEU.

### Performance Trends:
- **Transformer models** generally outperform traditional methods
- **Abstractive models** show higher BLEU scores but lower ROUGE scores
- **Extractive models** are more consistent but less creative
- **Graph-based methods** provide good baseline performance

## 📈 Performance Comparison

| Collection | Tabular comparison |
|---|---|
| Basic and advanced notebooks | See the model comparison table above. |
| Pegasus checkpoints | See the dedicated Pegasus evaluation table above. |

The original comparison images were removed as requested. Their score information is preserved in the tables above and the original-source archive.

## 🛠️ Technical Requirements

### Core Dependencies
```bash
python -m pip install -r requirements-models.txt
```

### Additional Libraries
- **spacy**: For advanced NLP processing
- **gensim**: For topic modeling
- **sumy**: For multi-document summarization
- **bert-extractive-summarizer**: For BERT-based summarization

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: Optional but recommended for transformer models
- **Storage**: At least 5GB free space for models
- **Python**: 3.9 or higher

## How to Install All Models

Use Python 3.9 or newer. `requirements-models.txt` installs the core notebook dependencies plus PyTorch, Transformers, SentencePiece, the BERT summarizer, and protobuf.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name text-summarization-models --display-name "Python (Text Summarization Models)"
.\.venv\Scripts\python.exe run_all_notebooks.py
```

### macOS or Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-models.txt
python -m ipykernel install --prefix .venv --name text-summarization-models --display-name "Python (Text Summarization Models)"
python run_all_notebooks.py
```

The runner registers the project kernel automatically if it is missing. Each pretrained notebook downloads its checkpoint through Transformers on first use. Hugging Face files are cached under `.model_cache/`; set `HF_HOME` to use another cache location. NLTK resources are stored under `.nltk_data/` and downloaded by the notebooks as needed. Missing cache files are downloaded again automatically.

No model weights are included in Git. Check each upstream model card and license for usage terms; weights differ in size and license, and the local Hugging Face cache is intentionally ignored by `.gitignore`.

## 📚 Educational Value

This repository serves as a comprehensive learning resource for:

1. **Understanding Different Approaches**: From basic statistical methods to advanced transformer models
2. **Performance Comparison**: Real-world evaluation using standard metrics
3. **Implementation Examples**: Complete working code for each approach
4. **Research References**: Academic papers and research background for each model
5. **Practical Applications**: Real-world use cases and considerations

### Learning Path Recommendations:

#### **Beginner Level**
1. Start with NLTK and TF-IDF models
2. Understand basic NLP concepts
3. Learn about evaluation metrics

#### **Intermediate Level**
1. Explore graph-based methods (TextRank, LexRank)
2. Study matrix decomposition (LSA)
3. Understand clustering approaches

#### **Advanced Level**
1. Dive into transformer models (BERT, BART, T5)
2. Experiment with Pegasus models
3. Implement custom approaches

## 🎯 Use Cases & Model Selection Guide

### **For Simple Documents**
- **NLTK**: Educational purposes, basic understanding
- **TF-IDF**: Keyword extraction, document classification
- **SumBasic**: Single document summarization

### **For Complex Documents**
- **BERT**: Context-heavy documents
- **BART**: High-quality abstractive summaries
- **T5**: Production systems, multilingual

### **For Multi-Document**
- **LSA**: Semantic analysis, multi-document
- **LexRank**: Long documents, systematic approach
- **Multi-Document Pegasus**: Advanced multi-document scenarios

### **For Real-time Applications**
- **DistilBART**: Fast inference, good quality
- **TextRank**: General purpose, consistent
- **TF-IDF**: Simple, fast processing

### **For Research Purposes**
- **GPT-2**: Creative summarization, research
- **Advanced transformer models**: State-of-the-art approaches
- **Graph-Based Summary**: Novel methodologies

## 🔬 Research Contributions

This project includes:
- **30 curated notebooks: 18 basic/advanced methods and 12 Pegasus checkpoints**
- **Comprehensive performance evaluation**
- **Educational implementations**
- **Research paper references**
- **Practical considerations and trade-offs**

### Research Areas Covered:
- **Extractive vs Abstractive Summarization**
- **Statistical vs Neural Approaches**
- **Graph-based vs Matrix-based Methods**
- **Traditional NLP vs Deep Learning**
- **Single vs Multi-document Summarization**

## 🚨 Common Issues & Troubleshooting

### Installation Issues
```bash
# If you encounter CUDA issues
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For memory issues with large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Model Loading Issues
- Ensure sufficient RAM for large models
- Use CPU versions for memory-constrained environments
- Check model file paths and permissions

### Performance Issues
- Use smaller models for faster inference
- Consider batch processing for multiple documents
- Implement caching for repeated summarizations

## 👥 Contributing

Contributions are welcome! Areas for improvement:
- Additional model implementations
- Performance optimizations
- Documentation improvements
- New evaluation metrics
- Domain-specific adaptations

### Contribution Guidelines:
1. Fork the repository
2. Create a feature branch
3. Add your implementation with proper documentation
4. Include performance metrics
5. Submit a pull request

## 🙏 Acknowledgements

This repository was created by me as part of an Academic Capstone Project at Vellore Institute of Technology. Special thanks to **Prof. Durgesh Kumar** and the faculty members for their guidance and support.

### Research Acknowledgments:
- **Hugging Face** for transformer model implementations
- **NLTK** team for natural language processing tools
- **Academic community** for research papers and methodologies

## 📄 License

This project is for educational purposes. Please respect the licenses of individual model implementations and research papers referenced.

### License Information:
- **Educational Use**: Free for educational and research purposes
- **Commercial Use**: Check individual model licenses
- **Attribution**: Please cite the original research papers

## 📞 Support & Contact

For questions, issues, or contributions:
- **Issues**: Use GitHub issues for bug reports
- **Discussions**: Use GitHub discussions for questions
- **Contributions**: Submit pull requests for improvements

---

**Note**: This repository is designed for educational purposes and research. For production use, please ensure proper licensing and consider the specific requirements of your use case.

### Version Information
- **Last Updated**: October 30, 2026
- **Model inventory**: 31 in the original README; 30 notebooks in the current curated layout
- **Python Version**: 3.9+
- **License**: Educational Use


## Shared Article, Notebook Layout, and Execution (Added October 30, 2026)

All notebooks use the shared global article in `article.txt` and target summary in `reference_summary.txt`. The article is a paraphrased, attributed digest of NOAA NCEI's [global 2025 climate assessment](https://www.ncei.noaa.gov/news/global-climate-202513). Shared preprocessing and evaluation helpers are in `summarization_utils.py`.

The notebook layout is **18 + 12 = 30**: 18 notebooks in `Basic and Advanced Text Summarisation Models/` and 12 checkpoint-specific Pegasus notebooks in `Pegasus Checkpoint Models/`.

The latest full run executed all 30 notebooks with no cell errors, inference fallbacks, or missing metrics. Results and metric-method labels are saved in `evaluation_results.csv`.

### Updated run results

The Pegasus table separates current shared-article metrics from the original notebook scores. The original scores used the COVID-19 sample and source-article comparison, so they are not directly comparable. Gigaword input is capped to the checkpoint's positional limit, and `protobuf` is included for Pegasus tokenization.

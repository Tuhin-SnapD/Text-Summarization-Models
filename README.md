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
# Python 3.7+ required
python --version

# Install core dependencies
pip install -r requirements.txt
```

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
cd "Basic to Advance Text Summarisation Models"

# Run any Jupyter notebook
jupyter notebook BERT.ipynb
```

## 📊 Model Performance Comparison Table

| Model | Type | ROUGE F1 | BLEU | Pros | Cons | Best For |
|-------|------|----------|------|------|------|----------|
| **GPT-2** | Abstractive | 0.797 | 0.423 | High recall, contextual | Low control, expensive | Research, complex docs |
| **LSA** | Extractive | 0.602 | 0.869 | Semantic relationships | Needs large data | Multi-document |
| **TextRank** | Extractive | 0.586 | 0.694 | Unbiased, consistent | Limited context | General purpose |
| **T5** | Abstractive | 0.573 | 0.683 | High accuracy, multilingual | Resource intensive | Production systems |
| **NLTK** | Extractive | 0.589 | 0.869 | Simple, language-independent | Limited accuracy | Beginners |
| **BERT** | Extractive | 0.545 | 0.677 | Context understanding | Extractive only | Context-heavy docs |
| **BART** | Abstractive | 0.402 | 0.905 | State-of-the-art | Large model size | High-quality summaries |
| **LexRank** | Extractive | 0.357 | 0.651 | Handles complex texts | Limited coverage | Long documents |
| **SumBasic** | Extractive | 0.589 | 0.621 | Simple implementation | Limited coverage | Single documents |
| **Pegasus XSum** | Abstractive | 0.085 | 0.462 | Extreme news summaries | Very short outputs | BBC-style news |
| **Pegasus CNN/DailyMail** | Abstractive | 0.517 | 0.796 | News article summaries | News-domain focus | Daily news |
| **Pegasus Newsroom** | Abstractive | 0.690 | 0.863 | Diverse news summaries | News-domain focus | Newsroom articles |
| **Pegasus MultiNews** | Abstractive | 0.557 | 0.510 | Multi-document summaries | Needs grouped sources | News collections |
| **Pegasus Gigaword** | Abstractive | 0.142 | 0.437 | Concise headline generation | Short-output bias | News headlines |
| **Pegasus WikiHow** | Abstractive | 0.209 | 0.420 | How-to content | Instructional domain | How-to articles |
| **Pegasus Reddit-TIFU** | Abstractive | 0.311 | 0.887 | Informal narrative summaries | Reddit-domain style | User stories |
| **Pegasus BigPatent** | Abstractive | 0.261 | 0.790 | Long patent documents | Technical language | Patent abstracts |
| **Pegasus ArXiv** | Abstractive | 0.291 | 0.074 | Research paper summaries | Long technical input | Scientific papers |
| **Pegasus PubMed** | Abstractive | 0.349 | 0.150 | Biomedical summaries | Biomedical domain | Medical literature |
| **Pegasus AESLC** | Abstractive | 0.233 | 0.917 | Email-oriented summaries | Email-domain focus | Email subject lines |
| **Pegasus BillSum** | Abstractive | 0.449 | 0.924 | Legislative summaries | Legal language | Bills and legislation |

## 🏗️ Model Categories

### 1. Basic Statistical Models
These models use fundamental NLP techniques and statistical approaches:

#### **NLTK-based Summarization** ([`NLTK.ipynb`](<Basic to Advance Text Summarisation Models/NLTK.ipynb>))
- **Approach**: Frequency-based sentence scoring using NLTK library
- **Pros**: Simple implementation, language-independent, well-documented
- **Cons**: Limited accuracy for complex texts, resource-intensive
- **Performance**: ROUGE F1: 0.589, BLEU: 0.869
- **Use Case**: Educational purposes, simple documents

#### **TF-IDF Summarization** ([`TFIDF.ipynb`](<Basic to Advance Text Summarisation Models/TFIDF.ipynb>))
- **Approach**: Term Frequency-Inverse Document Frequency weighting
- **Pros**: Effective for identifying important terms, widely used
- **Cons**: May miss context, struggles with technical jargon
- **Performance**: Evaluated using standard TF-IDF scoring
- **Use Case**: Keyword extraction, document classification

#### **SumBasic Algorithm** ([`SumBasic.ipynb`](<Basic to Advance Text Summarisation Models/SumBasic.ipynb>))
- **Approach**: Probabilistic sentence selection based on word frequency
- **Pros**: Simple implementation, good for single documents
- **Cons**: Limited coverage, may lack coherence
- **Performance**: ROUGE F1: 0.589, BLEU: 0.621
- **Use Case**: Single document summarization

### 2. Graph-Based Models
These models represent text as graphs and use centrality algorithms:

#### **TextRank** ([`TextRank.ipynb`](<Basic to Advance Text Summarisation Models/TextRank.ipynb>))
- **Approach**: Graph-based ranking using sentence similarity
- **Pros**: Automatic, unbiased, consistent results
- **Cons**: Limited context understanding, may miss nuances
- **Performance**: ROUGE F1: 0.586, BLEU: 0.694
- **Use Case**: General purpose summarization

#### **LexRank** ([`LexRank.ipynb`](<Basic to Advance Text Summarisation Models/LexRank.ipynb>))
- **Approach**: Graph-based lexical centrality using cosine similarity
- **Pros**: Handles complex texts, good performance on benchmarks
- **Cons**: Limited coverage, extractive only
- **Performance**: ROUGE F1: 0.357, BLEU: 0.651
- **Use Case**: Long documents, multi-document summarization

#### **Connected Dominating Set** ([`Connected_Dominating_Set.ipynb`](<Basic to Advance Text Summarisation Models/Connected_Dominating_Set.ipynb>))
- **Approach**: Graph theory-based sentence selection
- **Pros**: Theoretical foundation, systematic approach
- **Cons**: Computational complexity, may not capture semantics
- **Use Case**: Research applications

### 3. Matrix Decomposition Models
These models use mathematical techniques to identify important content:

#### **LSA (Latent Semantic Analysis)** ([`LSA.ipynb`](<Basic to Advance Text Summarisation Models/LSA.ipynb>))
- **Approach**: Singular Value Decomposition for semantic relationships
- **Pros**: Captures semantic relationships, good for multi-document
- **Cons**: Requires large training data, limited coverage
- **Performance**: ROUGE F1: 0.602, BLEU: 0.869
- **Use Case**: Multi-document summarization, semantic analysis

#### **Sentence Ranking** (overlapping with the TF-IDF sentence-scoring notebook and retained in the original-source archive)
- **Approach**: Matrix-based sentence importance scoring
- **Pros**: Systematic ranking, handles large documents
- **Cons**: May miss context, computational expense
- **Use Case**: Large document processing

### 4. Clustering-Based Models
These models group similar content together:

#### **K-Clustering** ([`K_Clustering.ipynb`](<Basic to Advance Text Summarisation Models/K_Clustering.ipynb>))
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

#### **Luhn's Model** ([`Luhn's_Model.ipynb`](<Basic to Advance Text Summarisation Models/Luhn's_Model.ipynb>))
- **Approach**: Classic statistical approach using word frequency
- **Pros**: Historical significance, simple implementation
- **Cons**: Basic approach, limited effectiveness
- **Use Case**: Educational purposes, historical context

#### **Compressive Document Summarization** ([`Compressive_Document_Summarization.ipynb`](<Basic to Advance Text Summarisation Models/Compressive_Document_Summarization.ipynb>))
- **Approach**: Sentence compression and selection
- **Pros**: More concise summaries, preserves key information
- **Cons**: Complex implementation, may lose context
- **Use Case**: Length-constrained summaries

### 6. Transformer-Based Models
These are state-of-the-art deep learning models:

#### **BERT Extractive Summarization** ([`BERT.ipynb`](<Basic to Advance Text Summarisation Models/BERT.ipynb>))
- **Approach**: Pre-trained BERT model for sentence extraction
- **Pros**: High accuracy, captures context well
- **Cons**: Limited to extractive, depends on pre-trained models
- **Performance**: ROUGE F1: 0.545, BLEU: 0.677
- **Use Case**: Context-heavy documents

#### **BART** ([`BART.ipynb`](<Basic to Advance Text Summarisation Models/BART.ipynb>))
- **Approach**: Bidirectional and Auto-Regressive Transformer
- **Pros**: State-of-the-art performance, high accuracy
- **Cons**: Resource-intensive, large model size
- **Performance**: ROUGE F1: 0.402, BLEU: 0.905
- **Use Case**: High-quality abstractive summaries

#### **T5** ([`T5.ipynb`](<Basic to Advance Text Summarisation Models/T5.ipynb>))
- **Approach**: Text-To-Text Transfer Transformer
- **Pros**: High accuracy, customizable, multilingual
- **Cons**: Resource-intensive, technical complexity
- **Performance**: ROUGE F1: 0.573, BLEU: 0.683
- **Use Case**: Production systems, multilingual applications

#### **GPT-2** ([`GPT_2.ipynb`](<Basic to Advance Text Summarisation Models/GPT_2.ipynb>))
- **Approach**: Generative Pre-trained Transformer 2
- **Pros**: Large language model, contextual understanding
- **Cons**: Lack of control, high computational costs
- **Performance**: ROUGE F1: 0.797, BLEU: 0.423
- **Use Case**: Research, creative summarization

#### **DistilBART** ([`Distilbart.ipynb`](<Basic to Advance Text Summarisation Models/Distilbart.ipynb>))
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

#### **SUMY** ([`SUMY.ipynb`](<Basic to Advance Text Summarisation Models/SUMY.ipynb>))
- **Approach**: Multi-document summarization toolkit
- **Pros**: Specialized for multi-document scenarios
- **Cons**: Limited to specific use cases
- **Use Case**: Multi-document summarization

#### **AI Transformer Pipeline** ([`AI_Transformer_pipeline.ipynb`](<Basic to Advance Text Summarisation Models/AI_Transformer_pipeline.ipynb>))
- **Approach**: Custom transformer-based pipeline
- **Pros**: Customizable architecture, modern approach
- **Cons**: Complex implementation, requires expertise
- **Use Case**: Custom applications, research

## 🚀 Advanced Architecture Models

### Pegasus Models (`Models Used in Proposed Architecture/`)

The repository includes **12 different Pegasus model implementations**:

#### **Single Document Pegasus** ([`Pegasus_XSum.ipynb`](<Models Used in Proposed Architecture/Pegasus_XSum.ipynb>))
- **Approach**: Single document abstractive summarization
- **Pros**: High-quality summaries, generalization capability
- **Cons**: Large computational requirements, limited control
- **Performance**: ROUGE F1: 0.086, BLEU: 0
- **Use Case**: Single document abstractive summarization

#### **Multi-Document Pegasus** ([`Pegasus_MultiNews.ipynb`](<Models Used in Proposed Architecture/Pegasus_MultiNews.ipynb>))
- **Approach**: Multi-document summarization
- **Pros**: Handles multiple documents, comprehensive coverage
- **Cons**: Complex processing, resource-intensive
- **Use Case**: Multi-document scenarios

#### **All 12 Pegasus Models** ([12 checkpoint-specific notebooks](<Models Used in Proposed Architecture/>))
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

#### **Graph-Based Summary** ([`Graph_Based_Summary.ipynb`](<Basic to Advance Text Summarisation Models/Graph_Based_Summary.ipynb>))
- **Approach**: Novel graph-based approach using LCS, TF-IDF, and matrix methods
- **Pros**: Combines multiple techniques, systematic approach
- **Cons**: Complex implementation, computational expense
- **Performance**: ROUGE F1: 0.417, BLEU: 0.901
- **Use Case**: Research, novel approaches

## 📈 Performance Analysis

### Key Insights from Results:

1. **Best Overall Performance**: GPT-2 achieved the highest ROUGE F1 score (0.797)
2. **Best BLEU Score**: BART achieved the highest BLEU score (0.905)
3. **Most Balanced**: T5 shows good balance between ROUGE and BLEU scores
4. **Efficiency vs Accuracy**: DistilBART offers a good trade-off for real-time applications

### Performance Trends:
- **Transformer models** generally outperform traditional methods
- **Abstractive models** show higher BLEU scores but lower ROUGE scores
- **Extractive models** are more consistent but less creative
- **Graph-based methods** provide good baseline performance

## 📈 Performance Comparison

| Collection | Tabular comparison |
|---|---|
| Existing model notebooks | See the updated shared-article results table below. |
| Pegasus checkpoints | See the 12-checkpoint score table below. |

The original comparison images were removed as requested. Their replacement is the updated results table in the added section below.

## 🛠️ Technical Requirements

### Core Dependencies
```bash
pip install transformers
pip install torch
pip install nltk
pip install rouge
pip install scikit-learn
pip install numpy
pip install pandas
pip install networkx
pip install sentencepiece
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
- **Python**: 3.7 or higher

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
- **31 different summarization approaches**
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
- **Python Version**: 3.7+
- **License**: Educational Use


## Shared Article, Notebook Layout, and Execution (Added October 30, 2026)

All notebooks use the shared global article in `article.txt` and the target summary in `reference_summary.txt`. The article is a paraphrased, attributed digest of NOAA NCEI's [global 2025 climate assessment](https://www.ncei.noaa.gov/news/global-climate-202513). Shared preprocessing and ROUGE/BLEU evaluation helpers are in `summarization_utils.py`.

The notebook layout is **18 + 12 = 30**: 18 notebooks in `Basic to Advance Text Summarisation Models/` and 12 checkpoint-specific Pegasus notebooks in `Models Used in Proposed Architecture/`. The complete original notebook and README sources (with the obsolete score images omitted) are preserved in `original_source_backup.zip`; the prior generated 30-notebook state is preserved in `current_generated_snapshot.zip`.

Install the dependencies in `requirements.txt`, then run `python run_all_notebooks.py` to execute all 30 notebooks in the local Jupyter kernel and save their outputs. In this Windows environment, all 30 notebooks completed with no cell errors. Twelve classical notebooks ran their selected methods, six transformer notebooks used the documented TextRank fallback because Windows Application Control blocked the PyTorch DLL, and the 12 Pegasus notebooks reported that inference was unavailable before checkpoint downloads began.

### Updated run results

The historical values in the original comparison table above are retained for context; they are not new scores from the shared NOAA article. Transformer fallback metrics are baseline results and must not be presented as pretrained-model scores. The Pegasus scores are now included in the main Model Performance Comparison Table above. They come from the original COVID-19 sample and were compared against the source article rather than a human-written reference, so they are not comparable with the shared NOAA article evaluation. Current Pegasus inference was attempted in Jupyter, but Windows Application Control blocked the local PyTorch runtime before generation.

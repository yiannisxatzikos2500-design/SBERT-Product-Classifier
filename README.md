# Goods vs Services Classification (Multilingual)

This repository contains the MSc Business Data Science (AAU) 1st-semester project on automatically classifying short business-related texts as **goods** or **services**, and predicting **macro-categories**. The project compares lexical baselines (TF-IDF) with sentence embedding approaches (SBERT and SetFit) and emphasizes **external validation** using an EUIPO-like dataset.

## Data

The main dataset is constructed by integrating multiple public sources (e-commerce product texts and service reviews) and harmonizing them into a common schema with:

- `name`
- `goods_or_services`
- `category`
- `about_product`
- `review`

> Note: If full datasets cannot be shared, this repository provides scripts to rebuild the final dataset structure or to reproduce the experiments using an anonymized sample.

## Data sources and provenance

All datasets used in this project are publicly available and were obtained from open data repositories. Each dataset was selected to represent either goods-oriented or service-oriented textual descriptions.

### Goods-oriented datasets

**Amazon Sales Dataset**  
Product names and short product descriptions from an e-commerce context, used to represent physical goods.  
Source:  
https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

**E-commerce Text Classification Dataset**  
Short product descriptions labeled by category, used to enrich the goods class with concise and structured product texts.  
Source:  
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

---

### Service-oriented datasets

**Airline Reviews Dataset**  
Customer reviews describing air travel services, including aspects such as flight experience, staff interaction, and punctuality.  
Source:
https://www.kaggle.com/datasets/juhibhojani/airline-reviews


**Hotel Reviews Dataset**  
User-generated reviews of hotel services, focusing on accommodation quality, amenities, and customer experience.  
Source:  
https://www.kaggle.com/datasets/anu0012/hotel-review

**Restaurant Reviews Dataset**  
Textual reviews describing dining services, food quality, and overall customer satisfaction.  
Source:  
https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews

---

### External validation dataset

**EUIPO-like trademark descriptions (hand-labeled)**  
A small external validation dataset inspired by EUIPO trademark descriptions of goods and services.  
This dataset was manually curated for evaluation purposes and is used exclusively for out-of-domain validation.  
Source inspiration:  
https://www.euipo.europa.eu/en/trade-marks

---

### Notes on data usage

- No personal data is intentionally collected or processed.
- All datasets are used strictly for academic research purposes.
- When redistribution of full datasets is not permitted, this repository provides code and documentation to reconstruct the data processing pipeline.
- Only derived and processed data strictly necessary for reproducibility are stored in the repository.


## Methods

1. **Preprocessing**
   - build a unified text field from `name`, `about_product`, and `review`
   - clean text (lowercasing, removing URLs/emails, keeping letters and spaces)

2. **Binary classification (goods vs services)**
   - TF-IDF + Logistic Regression (baseline)
   - TF-IDF + Logistic Regression (keywords removed robustness check)
   - SBERT embeddings + Logistic Regression
   - SBERT embeddings + Logistic Regression (augmented service training set)
   - SetFit (few-shot fine-tuning)

3. **Multi-class classification (macro categories)**
   - category prediction using the same text pipeline

4. **Evaluation**
   - internal validation (train/val/test split with stratification)
   - external validation (EUIPO-like hand-labeled set)
   - metrics: accuracy, macro-F1, weighted-F1, confusion matrices

## Results (key external validation table)

External validation (EUIPO-like):

| model | accuracy | f1_macro | f1_weighted |
|---|---:|---:|---:|
| TF-IDF + LogReg | 0.473684 | 0.321429 | 0.304511 |
| SBERT + LogReg (base) | 0.526316 | 0.424242 | 0.411483 |
| SBERT + LogReg (augmented) | 0.736842 | 0.724638 | 0.721587 |
| SetFit style (SBERT + LogReg) | 0.526316 | 0.424242 | 0.411483 |
| SetFit (real, Trainer, few-shot) | 0.789474 | 0.784091 | 0.782297 |

## Run the experiments

The data processing and model training pipeline is implemented using Jupyter notebooks.

Run the notebooks in the following order:

1. `01_build_dataset.ipynb`
2. `02_goods_services_classification_pipeline.ipynb`

## Run the Streamlit app

```bash
streamlit run 03_main.py
```


## Ethics and governance

No personal data is intentionally collected.
All data sources are public datasets.
Only data strictly necessary for reproducibility is stored in the repository.
When redistribution is not possible, the repository provides code to reconstruct the pipeline.

## Authors

Alvaro Buendía
Ioannis Chatzikos

## Supervisor

Milad Abbasiharofteh
Aalborg University Business School












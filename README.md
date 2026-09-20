# Long term shifts in North Atlantic seabird diet composition

A one day machine learning demonstration built while applying for the AWI/MarDATA PhD position in Data Science and Bioinformatics for Marine and Arctic Biodiversity. It is a portfolio proof of concept, not a submitted or peer reviewed study, see Limitations below.

**Write up:** [What 84 years of seabird diets reveal about a hidden ocean shift](https://simplyecologist.com/machine-learning-in-ecology-seabird-diets/), a plain language walkthrough of the method and the results.

## What this shows

The posting asks for experience applying supervised algorithms (including for environmental predictions) and unsupervised algorithms to biodiversity data, Python with data science libraries, sequence and dataset harmonization skills, and interest in interpreting patterns from long term Arctic and marine monitoring data. This project runs that full workflow end to end on a real, open, 84 year seabird monitoring dataset:

1. Harmonizes a raw diet records table into a documented, Darwin Core style occurrence schema, keeping the original WoRMS AphiaID taxonomic identifiers so the table stays joinable to other AphiaID or GBIF backbone indexed sources.
2. Builds a colony year by prey taxon diet composition matrix, the diet ecology equivalent of a site by species community matrix.
3. Runs unsupervised clustering (Bray Curtis dissimilarity, average linkage, visualised with NMDS) to look for structure across 84 years of monitoring.
4. Trains two supervised random forest classifiers: one predicting predator species identity from diet composition, and one predicting whether a sample is from before or after 1990 using nothing but diet composition, as a stand in for detecting an environmental or ecosystem regime signal from biological monitoring data, with feature importance used to surface likely indicator prey taxa.

## Data

[seabirddietDB](https://github.com/annakrystalli/seabirddietDB) by Anna Krystalli, Agnes Olin, James Grecian and Ruedi Nager (2019), MIT licensed. It collates 2857 published and grey literature records of prey consumed by the ten most common seabird species breeding in the British Isles, 1933 to 2017, across 64 colonies including long running monitoring sites such as Isle of May and Fair Isle. The raw `.rda` file is included as `data_raw_seabirddiet.rda` with attribution; see `LICENSE_DATA.md` for the original license text.

## Results

- Species identity classifier: 85 percent held out accuracy, 87 percent 5 fold cross validated accuracy, predicting which of eight seabird species a diet sample belongs to from composition alone. Sandeels (*Ammodytes marinus*, family Ammodytidae) are the dominant signal, consistent with known dietary specialisation in this system.
- Era classifier (pre versus post 1990, no year or location given to the model): 86 percent held out accuracy against a 69 percent majority class baseline, ROC AUC 0.86. The top predictive prey taxa (clupeids, gadids, sandeels) match the fish groups at the centre of well documented North Sea community changes since the late 20th century.
- Unsupervised Bray Curtis clustering does not cleanly recover either species or era (adjusted Rand index near zero for both). Read together with the supervised results, this suggests the multi decade dietary shift is a continuous, multivariate gradient rather than a sharp regime boundary, exactly the kind of pattern a supervised model can still detect even when clustering cannot draw a clean line. Reported as found rather than smoothed over.

Full numbers, the confusion matrix, feature importances and all figures are in the `outputs_*` and `figures_*` files, and are walked through with narrative in `seabird_diet_ml_analysis.ipynb`. This repository was uploaded as flat files through the browser for speed; running `analysis.py` locally regenerates the same content into proper `outputs/` and `figures/` folders.

## Limitations

- Northeast Atlantic colonies, not the Arctic sites named in the posting. The method transfers directly; it has not been run on Arctic data here.
- Only 2011 of 2857 raw records had a usable frequency of occurrence value; rows without one were dropped rather than imputed.
- The 1990 threshold is a simplification for the demonstration, not fitted to a specific documented shift year, and about a third of samples come from one very well sampled colony (Isle of May).
- The NMDS ordination has a moderate stress value (about 0.22), a useful but not highly precise two dimensional summary of the Bray Curtis distances.
- No environmental covariates (sea surface temperature, ice extent, bathymetry) are used. The era classifier is a biological proxy signal, not an explicit environment to biology model.

## What real Arctic monitoring work would add

Pulling actual Arctic seabird or benthic occurrence data from GBIF, OBIS or PANGAEA and harmonizing it against a dataset like this one using shared AphiaID or GBIF backbone keys, adding real environmental covariates instead of a manual era split, and comparing this Bray Curtis plus random forest baseline against transformer based sequence models for tasks like eDNA community composition, where sequence data rather than tabular frequencies is the primary input.

## Reproducing this

```
pip install -r requirements.txt
mkdir -p data_raw && mv data_raw_seabirddiet.rda data_raw/seabirddiet.rda && mv data_raw_DESCRIPTION data_raw/DESCRIPTION
python analysis.py            # regenerates outputs/ and figures/ from data_raw/seabirddiet.rda
python build_notebook.py      # rebuilds the notebook structure from analysis.py
jupyter nbconvert --to notebook --execute --inplace seabird_diet_ml_analysis.ipynb   # runs it and embeds outputs
```

## Repository layout

```
data_raw_seabirddiet.rda, data_raw_DESCRIPTION   original data file plus its package description
analysis.py                full pipeline: load, harmonize, composition matrix, clustering, classifiers
build_notebook.py          builds the notebook from the same functions in analysis.py
seabird_diet_ml_analysis.ipynb   narrated walkthrough with embedded figures and results
figures_*.png              all generated plots (analysis.py recreates these under figures/ when run locally)
outputs_*.csv / .txt       harmonized table, composition matrix, cluster assignments, classifier reports
```

## Licence

Code in this repository is MIT licensed, see `LICENSE`.

The harmonized occurrence table `outputs_harmonized_occurrence_table.csv` is a derived work released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse it freely, including commercially, with credit:

> Frey, E. (2026). Harmonized seabird diet occurrence table, British Isles 1933 to 2017. Derived from seabirddietDB. https://github.com/elifrey/seabird-diet-ml ORCID 0009-0002-7482-7986

The underlying records come from seabirddietDB by Krystalli, Olin, Grecian and Nager (2019), used with attribution. See `LICENSE_DATA.md` for the full terms.

"""
Long term shifts in North Atlantic seabird diet composition:
an unsupervised and supervised machine learning demonstration.

Data source: seabirddietDB (Krystalli, Olin, Grecian & Nager, 2019),
a database of seabird prey/diet records collated from published and
grey literature for the ten most common seabird species breeding in
the British Isles. MIT licensed. https://github.com/annakrystalli/seabirddietDB

This script:
  1. Loads and harmonizes the raw diet records into a Darwin Core
     style occurrence/measurement table (with WoRMS AphiaIDs preserved
     as the interoperable taxonomic identifier).
  2. Builds a colony year x prey taxon diet composition matrix, the
     community ecology equivalent of a site x species matrix.
  3. Runs unsupervised clustering (KMeans on a PCA reduced ordination)
     to look for structure in diet composition over 80+ years of
     monitoring, and checks whether the clusters track known regime
     shift periods in the North Sea / Northeast Atlantic.
  4. Trains supervised classifiers: (a) predator species identity from
     diet composition, to test dietary niche separation between
     species, and (b) a pre/post 1990 "era" label from diet composition
     alone, as a stand in for predicting environmental/regime state
     from biological monitoring data, with feature importance used to
     surface likely indicator prey taxa.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyreadr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

RANDOM_STATE = 42
FIG_DIR = "figures"
OUT_DIR = "outputs"

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 1. Load and harmonize
# ---------------------------------------------------------------------------

def load_raw():
    res = pyreadr.read_r("data_raw/seabirddiet.rda")
    df = res["seabirddiet"].copy()
    return df


def harmonize(df):
    """Bring the raw diet table into a tidy, documented occurrence/measurement
    schema. Column names and semantics follow Darwin Core where a direct
    analogue exists (decimalLatitude/decimalLongitude, scientificName), and
    the original WoRMS AphiaID identifiers are kept so records could be
    cross linked to OBIS/WoRMS or any other AphiaID indexed source."""

    out = pd.DataFrame({
        "recordID": df["id"],
        "year": df["year"].astype("Int64"),
        "decimalLatitude": df["latitude"],
        "decimalLongitude": df["longitude"],
        "locality": df["location"],
        "predatorScientificName": df["pred_valid_name"],
        "predatorAphiaID": df["pred_valid_aphia_id"],
        "predatorCommonName": df["pred_common_name"],
        "breedingStatus": df["pred_breeding_status"],
        "preyScientificName": df["prey_valid_name"].fillna(df["prey_taxon"]),
        "preyAphiaID": df["prey_valid_aphia_id"],
        "frequencyOfOccurrence": df["freq_occ"],
        "sampleSize": df["sample_size"],
        "referenceID": df["ref_ids"],
    })
    out = out.dropna(subset=["year", "predatorScientificName", "preyScientificName"])
    out = out[out["frequencyOfOccurrence"].notna()]
    out["year"] = out["year"].astype(int)
    return out


# ---------------------------------------------------------------------------
# 2. Community composition matrix
# ---------------------------------------------------------------------------

def build_composition_matrix(occ, min_prey_per_sample=3, top_n_prey=40):
    occ = occ.copy()
    occ["sample_unit"] = (
        occ["predatorScientificName"] + " | " + occ["locality"] + " | " + occ["year"].astype(str)
    )

    top_prey = occ["preyScientificName"].value_counts().head(top_n_prey).index
    occ["preyGroup"] = np.where(
        occ["preyScientificName"].isin(top_prey), occ["preyScientificName"], "Other/rare prey"
    )

    wide = occ.pivot_table(
        index="sample_unit",
        columns="preyGroup",
        values="frequencyOfOccurrence",
        aggfunc="mean",
        fill_value=0.0,
    )

    prey_count = occ.groupby("sample_unit")["preyGroup"].nunique()
    keep = prey_count[prey_count >= min_prey_per_sample].index
    wide = wide.loc[wide.index.intersection(keep)]

    meta = (
        occ.drop_duplicates("sample_unit")
        .set_index("sample_unit")[
            ["predatorScientificName", "predatorCommonName", "locality", "year",
             "decimalLatitude", "decimalLongitude"]
        ]
        .loc[wide.index]
    )
    return wide, meta


# ---------------------------------------------------------------------------
# 3. Unsupervised: ordination + clustering
# ---------------------------------------------------------------------------

def run_unsupervised(wide, meta):
    # Community ecology composition data (frequencies bounded 0-1, many zeros)
    # is not well served by Euclidean distance on standardized values: rare
    # taxa get inflated and the geometry stops reflecting compositional
    # similarity. Bray Curtis dissimilarity is the standard choice for this
    # kind of data, and average linkage hierarchical clustering (UPGMA) on
    # that dissimilarity is the conventional community ecology approach
    # (broadly equivalent to what the vegan package does in R).
    from sklearn.manifold import MDS

    D = squareform(pdist(wide.values, metric="braycurtis"))

    # Average linkage silhouette is maximized by splitting off tiny outlier
    # groups (down to singletons), which is not an interpretable regime
    # structure. We therefore only consider solutions where every cluster
    # has at least a handful of samples, and pick the best silhouette among
    # those (reported alongside the unconstrained scan for transparency).
    MIN_CLUSTER_SIZE = 4
    sil_scores = {}
    sil_scores_all = {}
    labels_by_k = {}
    Z = linkage(pdist(wide.values, metric="braycurtis"), method="average")
    for k in range(2, 12):
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels_k)) < 2:
            continue
        sil = silhouette_score(D, labels_k, metric="precomputed")
        sil_scores_all[k] = sil
        min_size = pd.Series(labels_k).value_counts().min()
        if min_size >= MIN_CLUSTER_SIZE:
            sil_scores[k] = sil
            labels_by_k[k] = labels_k
    best_k = max(sil_scores, key=sil_scores.get)
    cluster_labels = labels_by_k[best_k]

    mds = MDS(n_components=2, dissimilarity="precomputed", metric=False,
              random_state=RANDOM_STATE, n_init=4, normalized_stress="auto")
    coords = mds.fit_transform(D)
    stress = mds.stress_

    meta = meta.copy()
    meta["cluster"] = cluster_labels
    meta["pc1"] = coords[:, 0]
    meta["pc2"] = coords[:, 1]
    evr = [np.nan, np.nan]  # not meaningful for NMDS, kept for label reuse below

    # Does cluster membership track a pre/post ~1990 regime split?
    meta["era"] = np.where(meta["year"] < 1990, "before 1990", "1990 or later")
    era_vs_cluster = pd.crosstab(meta["cluster"], meta["era"])
    ari_era = adjusted_rand_score(
        (meta["era"] == "1990 or later").astype(int), meta["cluster"]
    )

    # Figure 1: silhouette by k
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(list(sil_scores_all.keys()), list(sil_scores_all.values()), marker="o",
            color="lightgrey", label="all solutions (incl. tiny outlier clusters)")
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o",
            color="C0", label=f"min cluster size >= {MIN_CLUSTER_SIZE}")
    ax.scatter([best_k], [sil_scores[best_k]], color="crimson", zorder=5, label=f"chosen k={best_k}")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("silhouette score (Bray Curtis)")
    ax.set_title("Cluster quality by k, average linkage on Bray Curtis dissimilarity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_silhouette_by_k.png")
    plt.close(fig)

    # Figure 2: NMDS scatter coloured by cluster
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(meta["pc1"], meta["pc2"], c=meta["cluster"], cmap="tab10", s=22, alpha=0.8)
    ax.set_xlabel("NMDS1")
    ax.set_ylabel("NMDS2")
    ax.set_title(f"Diet composition NMDS (stress={stress:.3f}), coloured by cluster (k={best_k})")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_pca_by_cluster.png")
    plt.close(fig)

    # Figure 3: NMDS scatter coloured by year (temporal gradient)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(meta["pc1"], meta["pc2"], c=meta["year"], cmap="viridis", s=22, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("year")
    ax.set_xlabel("NMDS1")
    ax.set_ylabel("NMDS2")
    ax.set_title("Diet composition NMDS, coloured by year")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_pca_by_year.png")
    plt.close(fig)

    # Figure 4: NMDS scatter coloured by predator species
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    species = meta["predatorCommonName"].astype("category")
    for cat in species.cat.categories:
        m = species == cat
        ax.scatter(meta.loc[m, "pc1"], meta.loc[m, "pc2"], s=22, alpha=0.8, label=cat)
    ax.set_xlabel("NMDS1")
    ax.set_ylabel("NMDS2")
    ax.set_title("Diet composition NMDS, coloured by predator species")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_pca_by_species.png")
    plt.close(fig)

    cluster_vs_species = pd.crosstab(meta["cluster"], meta["predatorCommonName"])
    ari_species = adjusted_rand_score(meta["predatorCommonName"], meta["cluster"])

    meta.to_csv(f"{OUT_DIR}/unsupervised_cluster_assignments.csv")
    era_vs_cluster.to_csv(f"{OUT_DIR}/cluster_vs_era_crosstab.csv")
    cluster_vs_species.to_csv(f"{OUT_DIR}/cluster_vs_species_crosstab.csv")

    return meta, best_k, sil_scores, era_vs_cluster, ari_era, cluster_vs_species, ari_species


# ---------------------------------------------------------------------------
# 4a. Supervised: predator species identity from diet
# ---------------------------------------------------------------------------

def run_species_classifier(wide, meta):
    y = meta["predatorScientificName"].values
    counts = pd.Series(y).value_counts()
    keep_species = counts[counts >= 15].index
    mask = pd.Series(y).isin(keep_species).values
    X = wide.values[mask]
    y = y[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=RANDOM_STATE, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced"),
        X, y_enc, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    )

    fi = pd.Series(clf.feature_importances_, index=wide.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=90, fontsize=7)
    ax.set_yticklabels(le.classes_, fontsize=7)
    ax.set_xlabel("predicted species")
    ax.set_ylabel("true species")
    ax.set_title(f"Species identity from diet composition\nheld out accuracy = {acc:.2f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_species_confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 6))
    fi.head(15).iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("random forest feature importance")
    ax.set_title("Top prey taxa distinguishing predator species")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_species_feature_importance.png")
    plt.close(fig)

    with open(f"{OUT_DIR}/species_classifier_report.txt", "w") as f:
        f.write(f"Held out accuracy: {acc:.3f}\n")
        f.write(f"5 fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n\n")
        f.write(report)

    fi.to_csv(f"{OUT_DIR}/species_feature_importance.csv")
    return acc, cv_scores, fi


# ---------------------------------------------------------------------------
# 4b. Supervised: pre/post 1990 era from diet composition (environmental proxy)
# ---------------------------------------------------------------------------

def run_era_classifier(wide, meta, threshold_year=1990):
    meta = meta.copy()
    meta["era"] = (meta["year"] >= threshold_year).astype(int)  # 1 = 1990 or later

    counts = meta["era"].value_counts()
    if counts.min() < 15:
        return None

    X = wide.values
    y = meta["era"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    baseline = max(np.mean(y_test), 1 - np.mean(y_test))

    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, class_weight="balanced"),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="roc_auc"
    )

    fi = pd.Series(clf.feature_importances_, index=wide.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(5, 6))
    fi.head(15).iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel("random forest feature importance")
    ax.set_title(f"Prey taxa most predictive of\npre vs post {threshold_year} diet era")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_era_feature_importance.png")
    plt.close(fig)

    with open(f"{OUT_DIR}/era_classifier_report.txt", "w") as f:
        f.write(f"Task: predict whether a diet sample is from {threshold_year} or later, "
                f"using only its prey composition (no year or location given to the model).\n")
        f.write(f"Held out accuracy: {acc:.3f} (majority class baseline: {baseline:.3f})\n")
        f.write(f"Held out ROC AUC: {auc:.3f}\n")
        f.write(f"5 fold CV ROC AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n\n")
        f.write("Top indicator prey taxa (feature importance):\n")
        f.write(fi.head(15).to_string())

    fi.to_csv(f"{OUT_DIR}/era_feature_importance.csv")
    return acc, auc, cv_scores, fi, baseline


# ---------------------------------------------------------------------------
# 5. Descriptive figure: sandeel share of diet over time (context/EDA)
# ---------------------------------------------------------------------------

def sandeel_trend_figure(occ):
    is_sandeel = occ["preyScientificName"].str.contains(
        "Ammodyt", case=False, na=False
    )
    trend = (
        occ.assign(is_sandeel=is_sandeel)
        .groupby("year")
        .apply(lambda g: np.average(g["is_sandeel"], weights=g["frequencyOfOccurrence"].clip(lower=0.001)))
    )
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(trend.index, trend.values, marker="o", ms=3, lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel("sandeel weighted share of diet records")
    ax.set_title("Sandeel (Ammodytidae) presence in seabird diet records over time")
    ax.axvline(1990, ls="--", color="grey", lw=1)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/00_sandeel_trend.png")
    plt.close(fig)


def main():
    df = load_raw()
    occ = harmonize(df)
    occ.to_csv(f"{OUT_DIR}/harmonized_occurrence_table.csv", index=False)
    print(f"Harmonized occurrence table: {occ.shape[0]} rows, {occ.shape[1]} columns")

    sandeel_trend_figure(occ)

    wide, meta = build_composition_matrix(occ)
    wide.to_csv(f"{OUT_DIR}/diet_composition_matrix.csv")
    print(f"Composition matrix: {wide.shape[0]} colony year samples x {wide.shape[1]} prey groups")

    meta, best_k, sil_scores, era_vs_cluster, ari_era, cluster_vs_species, ari_species = run_unsupervised(wide, meta)
    print(f"Best k by silhouette (min cluster size enforced): {best_k}")
    print("Cluster vs era crosstab:")
    print(era_vs_cluster)
    print(f"Adjusted Rand index (cluster vs pre/post 1990): {ari_era:.3f}")
    print("Cluster vs predator species crosstab:")
    print(cluster_vs_species)
    print(f"Adjusted Rand index (cluster vs predator species): {ari_species:.3f}")

    acc, cv_scores, fi_species = run_species_classifier(wide, meta)
    print(f"Species classifier held out accuracy: {acc:.3f}, CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    era_result = run_era_classifier(wide, meta)
    if era_result:
        acc_e, auc_e, cv_e, fi_era, baseline = era_result
        print(f"Era classifier held out accuracy: {acc_e:.3f} (baseline {baseline:.3f}), AUC: {auc_e:.3f}")
        print(f"Era classifier CV AUC: {cv_e.mean():.3f} +/- {cv_e.std():.3f}")

    print("\nDone. Figures in figures/, tables in outputs/.")


if __name__ == "__main__":
    main()

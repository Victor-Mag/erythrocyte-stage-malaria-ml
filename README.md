<h1>Antimalarial Activity Prediction – Baseline QSAR Model</h1>

<h2>Project Overview</h2>

<p>
This repository contains a <strong>baseline machine learning model</strong> for predicting
<strong>antimalarial activity in the blood stage</strong> using classical QSAR techniques.
</p>

<p>
The current version (<strong>v1.0 – baseline</strong>) is intentionally designed as a
<strong>transparent and reproducible reference point</strong>, rather than a final or
production-ready virtual screening solution.
</p>

<hr/>

<h2>Current Model Summary (v1.0 – Baseline)</h2>

<ul>
  <li><strong>Dataset size:</strong> ~3000 molecules</li>
  <li><strong>Class ratio:</strong> ~0.4 active / 0.6 inactive</li>
  <li><strong>Descriptors:</strong> Morgan (ECFP) fingerprints</li>
  <li><strong>Algorithm:</strong> XGBoost classifier</li>
  <li><strong>Train/test split:</strong> Random split</li>
  <li><strong>Optimization focus:</strong> Recall of the active class</li>
</ul>

<h3>Performance (Random Split)</h3>

<ul>
  <li>Accuracy: ~0.86</li>
  <li>Active class precision: ~0.79</li>
  <li>Active class recall: ~0.71</li>
</ul>

<p>
Overall, the model behaves as a <strong>conservative classifier</strong>, favoring precision
over aggressive recovery of active compounds.
</p>

<hr/>

<h2>Critical Limitations</h2>

<h3>1. Optimistic Evaluation Due to Random Split</h3>

<p>
The use of a <strong>random train/test split</strong> likely inflates performance estimates due to:
</p>

<ul>
  <li>Structural similarity between training and test compounds</li>
  <li>Scaffold overlap when using circular fingerprints</li>
</ul>

<p>
Consequently, reported metrics should be interpreted as
<strong>upper-bound performance estimates</strong>, not true generalization to novel chemistry.
</p>

<h3>2. Limited Generalization to Novel Chemical Scaffolds</h3>

<p>
Morgan fingerprints primarily encode <strong>local substructural information</strong> and may fail to:
</p>

<ul>
  <li>Generalize to unseen scaffolds</li>
  <li>Capture mechanism-level or physicochemical drivers of activity</li>
</ul>

<p>
This limitation is especially relevant for antimalarial discovery, where
<strong>scaffold hopping is essential</strong>.
</p>

<h3>3. Trade-off Between Recall and False Positives</h3>

<p>
While recall optimization for the active class was explored, higher recall values led to
a substantial increase in false positives.
</p>

<p>
This behavior reflects a <strong>structural ambiguity in chemical space</strong>,
rather than insufficient hyperparameter tuning.
</p>

<h3>4. Classification-Oriented Metrics</h3>

<p>
Standard classification metrics (Accuracy, F1-score) are not ideal for virtual screening,
where <strong>early enrichment</strong> is more relevant than global classification performance.
</p>

<hr/>

<h2>Intended Use of the Current Model</h2>

<p>
The <strong>v1.0 baseline</strong> model is suitable for:
</p>

<ul>
  <li>Conservative prioritization of compounds within a known chemical space</li>
  <li>Serving as a methodological reference for future model development</li>
</ul>

<p>
It is <strong>not intended</strong> for:
</p>

<ul>
  <li>De novo hit discovery</li>
  <li>High-confidence prediction on novel scaffolds</li>
  <li>Production-level decision-making</li>
</ul>

<hr/>

<h2>Planned Improvements</h2>

<h3>Methodological Improvements</h3>

<ul>
  <li>Scaffold-based splitting (Bemis–Murcko)</li>
  <li>Evaluation using enrichment metrics (EF1%, EF5%, BEDROC, PR-AUC)</li>
  <li>Learning-to-rank approaches for virtual screening</li>
</ul>

<h3>Descriptor Enhancement</h3>

<ul>
  <li>Combination of Morgan fingerprints with physicochemical descriptors</li>
  <li>Exploration of pharmacophore-based or hybrid representations</li>
</ul>

<h3>Modeling Strategies</h3>

<ul>
  <li>Comparison between conservative and high-recall models</li>
  <li>Two-stage (cascade) screening approaches</li>
  <li>Threshold optimization driven by screening objectives</li>
</ul>

<hr/>

<h2>Versioning</h2>

<ul>
  <li>
    <strong>v1.0 – Baseline:</strong>
    Random split, Morgan fingerprints, XGBoost classifier
  </li>
</ul>

<p>
All future versions will be explicitly compared against this baseline to ensure
<strong>methodological transparency and reproducibility</strong>.
</p>

<hr/>

<h2>Final Note</h2>

<p>
This project prioritizes <strong>scientific honesty over metric inflation</strong>.
Limitations are explicitly documented to support meaningful interpretation and
continuous improvement.
</p>

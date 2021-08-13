# Lupus Time-Series Readme

# Creator: Dylan Smith dsmith167@fordham.edu



**Base Repository (Pre-processing) [Lupus/]:**

Notes: Current Code requires all csvs to fall in shared folder with the following.

ST1.0\_timesteps.py

	Purpose: Derive statistics about distance between data points

ST1.5\_med\_preprocessing.py

	Purpose: Assign each medication prescription instance a group among pertinent medication groups to Lupus study (or assign to other)

ST2\_transform\_0.1.\_series.py

	Purpose: Transform base data to proper length (3 years of data and 1 year of encoded label data) and synthesize into demographic and encounter derived features

ST3\_lupus\_combine.py

	Purpose: Combine into a cohesive data set, given trimmed and formatted csvs for encounter, lab tests, and medications

ST4A\_time\_series\_impute.py

	Purpose: Given cohesive data set, extrapolate/interpolate forward, and impute the rest of missing data, to output model-ready dataset 1

ST4B\_time\_series\_extrapolate.py

	Purpose: Given cohesive data set, extrapolate in two directions and interpolate, and then remove instances with missing entries, to output model-ready dataset 2


ST\_1\_LSTM.py (Serves an analogous purpose to ST4A or ST4B except for LSTM Model, utilizing ST4B Output and Base Dataset)

	Purpose: Take extrapolated and interpolated data and add on functionality for Labels defined at every time step (0,6,...36 months) and with different target lengths (12,9,6,3 months)


**Sub Folder (Project Testing) [Lupus/project_test/]:**

Notes: Current Code requires all csvs in outer folder. Additionally, &quot;feat\_importance&quot;, &quot;output&quot;, &quot;ROC curves&quot;, and &quot;saved\_results&quot; are sub folders that need to be established prior to running.

project\_functions.py

	Purpose: Hold functions to be used in project\_test\_dict, for cleanliness and convenience

project\_test\_dict.py

	Purpose: Interactively run a single or the complete set of machine learning models, either from scratch or from previously created data, and export outputs, feature importance, roc auc curves, and saved data. Added Functionality : Incorporate optional lag amounts (0 through 5) and target lengths (12,9,6,3 months)


lstm\_functions.py

	Purpose: Hold functions to be used in lstm\_test\_search, for cleanliness and convenience

lstm\_test\_search.py

	Purpose: Interactively run a single or set of lstm models and export outputs. Optional inclusion of flat variables or lstm/simple nn toggles; interactive choice of lag amounts (0 through 5), target lengths (12,9,6,3 months), testing size within cross-validation (.05,.1,.2), and nested-CV decision metric ('f1', 'f0', 'acc1', 'acc0', 'acc', 'auc', 'ppr', 'npr', 'diagI')


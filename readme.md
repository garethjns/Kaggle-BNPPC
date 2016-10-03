# Kaggle BNP Paribas Cardif
Data and description: [https://www.kaggle.com/c/bnp-paribas-cardif-claims-management][1]

Objective: Accelerate claims management process

Position: 490/2926 (top 17%)

## Methods

- Tree ensembles (R)
	* Fork of [https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select/code][2]
	* Feature engineering:
		* Counting missing data, error codes
	* Preprocessing:
		* Zero and near-zero var features removed (caret)
		* Highly correlated features removed (caret)
		* Linear feature combinations removed (caret)
		* Scaling (caret)
	* Model training:
		* Boosted tree ensemble (XGBoost)
		* Extra trees ensemble (extraTrees)
	
	 
[1]: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
[2]: https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select/code
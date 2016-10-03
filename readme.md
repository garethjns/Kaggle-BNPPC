# Kaggle BNP Paribas Cardif
[https://www.kaggle.com/c/bnp-paribas-cardif-claims-management][1]

Objective: Accelerate claims management process

Data: ~230000 rows, 132 features

Position: 490/2926 (top 17%)

## Methods

- Tree ensembles (R)
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
	* Fork of [https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select/code][2]

	 
[1]: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
[2]: https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select/code
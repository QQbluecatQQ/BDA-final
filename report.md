# Final Project–Big Data

> 系級: 精準健康碩一
> 學號: R13K47008
> 姓名: 俞政佑
> GitHub: <https://github.com/QQbluecatQQ/BDA-final>

## STEP1: Visualization

1. public dataset
	![[public_pairplot.png]]

2. private dataset
	![[private_pairplot.png]]


在最一開始的步驟中, 我決定先將 data 進行可視化, 查看 data 每一個 feature 之間的關係, 以及每一個 feature 的分佈情形. 這裡使用了 `seaborn.pairplot` 來繪製 pairplot, 以便於觀察各個 feature 之間的關聯性


* analysis:
	1. public data: `特徵 2 vs 特徵 1`、`特徵 2 vs 特徵 3`、`特徵 3 vs 特徵 4` 呈現放射狀結構, 有算是滿明顯的分群狀況 ,可能代表某種結構性的相關.
	2. private data: 除了在 public data 提到的 `特徵 2 vs 特徵 1`、`特徵 2 vs 特徵 3`、`特徵 3 vs 特徵 4` 有肉眼可見的分群外, `特徵 4 vs 特徵 5 `、`特徵 5 vs 特徵 6`、`特徵 6 vs 特徵 7` 也有明顯的分群狀況, 可能代表某種結構性的相關.

## STEP2: Preprocess

在 clustering 前, 我先對資料進行 preprocessing, 以提高資料品質並為後續分析提供標準化輸入. 我首先移除缺失值並篩選出具有變異性的特徵欄位, 排除對分群無貢獻的欄位(移除 `id`). 接著, 根據指定的方法（IQR 或 Z-score）移除異常值. IQR 方法基於四分位距（IQR = Q3 - Q1）定義異常值範圍, `IQR_THRESHOLD`（預設 1.5）控制檢測的嚴格程度, 較大的閾值保留更多資料點, 適合分佈較分散的資料, 較小的閾值則更嚴格, 適用於需要精確移除極端值的場景. Z-score 方法計算資料點偏離均值的標準差倍數, `ZSCORE_THRESHOLD`（常見值為 3）決定保留資料的比例, 適合近似正態分佈的資料. 根據 STEP1 的 pairplot 顯示部分特徵呈現明顯的分群與實驗結果, 我選擇 IQR 方法並設 `IQR_THRESHOLD` 為 12 (在固定的 clustering algorithm 下, 這樣的設定在 public data 上表現最佳.) 最後, 透過 StandardScaler 標準化特徵, 統一尺度以避免差異影響分群結果. 此流程確保資料乾淨且一致, 為後續分群分析提供可靠基礎.


## STEP3: Clustering
 

在此部分, 我嘗試了多種常見的 clustering 方法, 包括 `kmeans`, `kmeans++`, `minibatch`, `kmedoids`, `gmm`, `spectral`. `dbscan`, `agglo`, `birch`, `optics`. 我評估了三項指標 `Public data score`、`Silhouette Score`、`Calinski-Harabasz Score` 以衡量分群的品質與穩定性. `Public data score` 反映分群結果的整體準確性, `Silhouette Score` 衡量群內凝聚度和群間分離度, `Calinski-Harabasz Score` 則評估群組間分散度與群內緊密度的比例. 根據實驗結果, 我選擇了 `K-means（init='random'）`, 其在 public dataset 上的得分為 0.9032, 僅次於 Agglomerative Clustering  0.9062, 且具有最高的 Silhouette Score 0.6098 和 Calinski-Harabasz Score 36334.9479. 以下說明 K-means 算法、其適用性、高維資料處理能力.

  

* Introduction to K-means:
	我使用了 K-means 算法, 實現為 `KMeans(n_clusters=15, init='random', random_state=args.random_state)`. K-means 是一種基於距離的分群方法, 通過迭代優化將資料點分配到 15 個群組, 使每個群組內資料點到質心（centroid）的距離平方和最小化. 採用 `init='random'` 初始化方法, 隨機選擇初始質心, 並通過固定 random_state 確保結果可重現.

  

* Why it is suitable for this dataset
	最直觀的理由是 K-means 在三項指標中綜合表現最佳：`Public data score` 0.9032 接近最高分 Agglomerative Clustering 的 0.9062, `Silhouette Score` 0.6098 和 `Calinski-Harabasz Score` 36334.9479 均為最高, 顯示其群內凝聚度和群間分離度優異, 群組間分散度大且群內緊密. 從 STEP1 的 pairplot 可知, dataset 中特徵呈現放射狀結構, 並有明顯的分群趨勢. K-means 擅長處理具有清晰邊界的球形分群, 與這些結構化關係相符. STEP2 的前處理（IQR 方法, IQR_THRESHOLD=12）移除異常值並標準化特徵, 減少了 K-means 對異常值和尺度變化的敏感性, 使其更能發揮優勢. 相較於其他算法（如 GMM 的 0.7563 或 DBSCAN 的 0.3831）, K-means 的高效性和穩定性使其成為本資料集的理想選擇.

  

* How it handles high-dimensional data
	public dataset 包含 4 個特徵, 屬於相對低維, K-means 的表現如預期優異, 實驗結果顯示其 Silhouette Score  0.6098 和 Calinski-Harabasz Score 36334.9479 均為最高, Public data score 0.9032 也接近最佳. 然而, 當特徵維度增至 6 維時, K-means 可能面臨高維資料的挑戰, 例如「維度災難」, 即歐氏距離度量因點間距離趨於相似而失去區分性, 影響分群效果. 儘管如此, 6 維仍非極高維, 且實驗顯示 Silhouette Score 和 Calinski-Harabasz Score 雖略有下降, 但仍表現尚可, Public data score 遠超其他適用於高維資料的算法（如 GMM 的 0.7563）. 我嘗試使用 PCA 降維, 但結果在 public dataset 上反而下降, 且因 private dataset 缺乏真實標籤, 無法驗證降維效果. 因此, 結合 STEP2 的前處理（標準化和特徵篩選保留 4 維有效特徵）,我認為 K-means 在本資料集的維度範圍內仍能有效分群, 適用性良好.

  
  

## STEP4: Unsupervised Evaluation

因為本次作業是無監督學習, 我使用了 Silhouette Score 和 Calinski-Harabasz Score 來評估分群結果的品質, 並以助教提供的 Public Data Score 作為 public dataset 的外部評估指標.在此部分, 我介紹這兩個指標的定義、計算方式及其意義, 並說明其在評估分群方法中的作用.

  

* Silhouette Score：

	Silhouette Score 通過計算每個資料點的群內平均距離（$a(i)$）與最近其他群組的平均距離（$b(i)$）來衡量分群品質, 公式為 $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, 範圍在 $[-1, 1]$.分數接近 1 表示群內凝聚度高且群間分離度好, 分群品質優異.K-means（init='random'）的 Silhouette Score 為 0.6098, 為最高, 顯示其分群結構清晰, 與 STEP1 pairplot 的線性放射狀分群（如 特徵 2 vs 特徵 1）一致, 優於其他方法（如 GMM 的 0.3714).

  

* Calinski-Harabasz Score：

	Calinski-Harabasz Score 通過群組間分散度（$\text{SS}_B$）與群組內分散度（$\text{SS}_W$）的比率評估分群品質, 公式為 $CH = \frac{\text{SS}_B / (k-1)}{\text{SS}_W / (n-k)}$.分數越高表示群組間分離度高、群內緊密.K-means（init='random'）的 Calinski-Harabasz Score 為 36334.9479, 為最高, 顯示群組間分散度大, 群內凝聚度高, 驗證其分群效果優於其他方法（如 GMM 的 11120.2392）

  

## STEP5: Experiment Results

  
 

1. public dataset
	
	| Method | Silhouette Score | Calinski-Harabasz Score | Public Data Score |
	| ------ | ---------------- | ----------------------- | ----------------- |
	| kmeans       | 0.6098           | 36334.9479              | 0.9032      |
	| kmeans++     | 0.5663           | 35868.0185              | 0.8239      |
	| minibatch    | 0.5662           | 35177.0865              | 0.8285      |
	| kmedoids     | 0.5297           | 18884.0217              | 0.8721            |
	| gmm          | 0.3714           | 11120.2392              | 0.7563            |
	| spectral     | 0.4751           | 12708.8440              | 0.7659            |
	| dbscan       | 0.5821           | 270.3420                | 0.3831            |
	| agglo        | 0.5855           | 34666.3260              | 0.9062            |
	| birch        | 0.3830           | 5913.1799               | 0.5076            |
	| optics       | -0.5721          | 13.5861                 | 0.2828            |

  

2. private dataset

	| Method       | Silhouette Score | Calinski-Harabasz Score |
	|--------------|------------------|-------------------------|
	| kmeans       | 0.5241           | 94131.3844              |
	| kmeans++     | 0.5404           | 94569.0271              |
	| minibatch    | 0.5164           | 89404.2491              |
	| kmedoids     | -                | -                       |
	| gmm          | 0.2454           | 30407.6454              |
	| spectral     | 0.4096           | 39608.0735              |
	| dbscan       | -0.2651          | 279.6048                |
	| agglo        | -                | -                       |
	| birch        | 0.3986           | 22429.4383              |
	| optics       | -0.6450          | 12.0238                 |


## STEP6: Result Visualization


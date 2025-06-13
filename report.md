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


## Conclusion & Analysis

* 分群 pipeline: 
	本次分群作業從 data visualization 出發, 透過 pairplot 初步觀察特徵之間的分布與潛在分群結構. 接著進行 data preprocessing, 包含缺失值移除、異常值處理（使用 IQR, threshold 設為 12）與標準化（StandardScaler）, 確保資料分布一致且不受極端值干擾. 完成資料清洗後, 嘗試多種分群演算法（共 10 種）, 最後以內部指標（Silhouette Score、Calinski-Harabasz Score）及外部指標（Public data score）綜合比較, 選定最佳模型, 並進行 PCA 與 UMAP 的 2D 可視化以輔助評估分群結構的合理性. 

* Preprocessing 的有效性:
	在 preprocessing 階段, 我使用了 IQR 方法來移除異常值, 並將 `IQR_THRESHOLD` 設為 12. 這個方法使 public dataset score 從 0.88 提升至 0.90, 顯示出 preprocessing 在提升分群品質方面的有效性. IQR 方法能夠有效地去除極端值, 減少對 K-means 分群結果的負面影響, 並且在特徵標準化後, 資料分布更為均勻, 有助於 K-means 算法的收斂與分群效果.

* Model 選擇:
	經實驗比較 , `K-means（init='random'）` 在 public dataset 上的整體表現最為穩定且分數最高（或次高）, 在 Silhouette Score（0.6098）與 Calinski-Harabasz Score（36334.95）上優於所有其他模型, 在 Public data score 上也僅略低於 Agglomerative。其餘模型如 GMM、DBSCAN、OPTICS 雖各有特點, 但受限於資料特性（偏向球狀分布、維度不高）與參數敏感性, 整體分群品質與穩定性皆不如 K-means。因此, 綜合效率、表現與解釋性, K-means 被選為最終模型。

* 最終結果分析:

	1. Public dataset: 畢竟不論是 model 還是參數的調整, 都是針對 public dataset 進行的, 所以在 public dataset 上分群的表現還不錯. Public dataset 經降維後的 PCA 與 UMAP 結果皆顯示出明顯的分群分布, 群體邊界清晰, 說明 K-means 對此資料集的適應性極佳. 同時助教提供的 Public Data Score 為 0.9032 相較於其他方法可以說是相當高的分數, 代表 K-means 在此資料集上成功捕捉到潛在的分群結構. 在評估指標方面, K-means 的 Silhouette Score 為 0.6098 和 Calinski-Harabasz Score 達 36334 代表了群內緊密且群間明確, 至少是實驗各種方法中表現最好的. 整體而言, K-means 成功揭示了原始資料中潛在的分群結構. 

	2. Private dataset: 相較於 public dataset, Private dataset 的 feature 維度提升至 6 並要分 23 個群, 這對 K-means 來說分群挑戰加劇. 儘管無法直接得知 private data score, 但從 Silhouette Score（0.5241）和 Calinski-Harabasz Score（94131.3844）來看, 分群品質仍然最眾多實驗方法中表現較佳. 這顯示 K-means 在高維資料上仍能保持一定的分群效果, 雖然可能不如在 public dataset 上那麼明顯. UMAP 投影下群與群之間略有重疊, 說明了高維資料的複雜性與 K-means 在此情境下的局限性. 但整體仍能觀察出合理的聚類輪廓.

	3. 統整: public dataset 在我的 preprocessing 與 K-means 分群下擁有還不錯的表現, 但 private dataset 可預期的因為增加維度, 使的 K-means 分群的效果不如 public dataset. 但整體而言, K-means 在兩個資料集上都能夠捕捉到潛在的分群結構, 並且在評估指標上相對其他方式表現較佳.

## Some notes (一些用了但沒有效果的實驗)

1. data 取 log
2. 對 feature 進行加權
3. 使用 PCA 或 KernelPCA
4. 先對某些維度進行初步 clustering, 再做後續擴展



# CSS-HGAT-SSP

To tackle the challenge of spatiotemporal data sparsity in disaster scenarios, we propose a method for flood risk prediction that leverages the principle of geographic similarity (<b>CSS-HGAT-SSP</b>). By aggregating information from distant neighbors, the approach enhances predictive accuracy and resilience.

## 1. Data Source

All the data utilized in this study is publicly accessible and available for download online. To promote research collaboration and knowledge sharing, the sources of all datasets are listed below.
A similar description of the following data can be found in our previously published paper, "Integrated expression and analysis of urban flood disaster events from the perspective of multi-spatial semantic fusion," in the International Journal of Applied Earth Observation and Geoinformation <sup>[1]</sup>.

| Category |	Source |	Resolution |	Url |
| :--- | :--- | :--- | --- |
| Precipitation distribution | CHIRPS | 30m | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| DEM |	FABDEM | 30m |	https://data.bris.ac.uk/data/dataset/25wfy0f9ukoge2gs7a5mqpq2j7 |
| Landuse | ESRI Global Land Use Data | 30m | https://livingatlas.arcgis.com/landcoverexplorer |
| Climate | Chinese Academy of Sciences | 1km | https://www.resdc.cn/DOI/DOI.aspx?DOIID=96 |
| Night light index | EOG Nighttime Light | 15" | https://eogdata.mines.edu/nighttime_light/annual/v21/2021/ |
| Population density | WorldPop | 100m | https://hub.worldpop.org/geodata/summary?id=49730 |
| Social media data | Weibo | — | https://weibo.com |
| Flood inundation | ChinaGEOSS Data Sharing Network | 5m |	https://noda.ac.cn/datasharing/datasetDetails/62bedede0ef34854c4c2787f |   

[1] Wang, Shunli, Rui Li, and Huayi Wu. "Integrated expression and analysis of urban flood disaster events from the perspective of multi-spatial semantic fusion." International Journal of Applied Earth Observation and Geoinformation 132 (2024): 104032. [https://doi.org/10.1016/j.jag.2024.104032]

## 2. Project Structure

Our proposed CSS-HGAT-SSP method's entire project consists of the following five folders.

| Folder                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_data              | Including the initial data directly used in the experiment, which are files processed from the raw data. |
| ./experimental_results           | Including the output result files from each experimental step. |
| ./program_embedding              | Contains scripts for generating and evaluating embeddings using various algorithms, including CSS-HGAT, DeepWalk, GAE, GCN, GraphSAGE, and Node2Vec.|
| ./program_prediction_parameter   | Contains code for the selection of the number of nearest neighbor nodes and the ablation study of the risk prediction module.|
| ./program_spatial_prediction     | Contains code for spatial flood risk prediction using spatial relationships and neighborhood aggregation mechanisms.|
| ./program_temporal_prediction    | Contains code for temporal flood risk prediction based on sequential patterns and time-series modeling techniques.|

## 3. Experimental Steps

### 3.1 Data preprocessing

After collecting the necessary experimental data, data preprocessing is conducted as the first step. For raster imagery data, the following operations are performed using ArcGIS Pro software (available for download at https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview): <br>(1) Create a vector grid within Zhengzhou city with a 500-meter side length. <br>(2) Standardize the coordinate system to the Krasovsky_1940_Albers projection.<br>(3) Use the "Zonal Statistics" tool to unify the raster scale to 500 meters.<br>(4) Extract multidimensional attribute values for each unit using the "Extract Multi Values to Points" tool.

The process of extracting flood disaster impact information based on Weibo data is detailed in our previously published paper, "Fine-grained flood disaster information extraction incorporating multiple semantic features," in the International Journal of Digital Earth <sup>[2]</sup>. The preprocessed data is saved in the "experimental_data" directory, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_data/Grid500_AllCity_Edges.csv           | Connectivity relationships between all grid units in Zhengzhou City. |
| ./experimental_data/Grid500_AllCity_Nodes.csv           | Attribute information of all grid units in Zhengzhou City. |
| ./experimental_data/Grid500_Selected_FloodArea.csv      | Disaster-affected area of grid units overlapping with flood disaster submerged zones. |

[2] Wang, Shunli, Rui Li, and Huayi Wu. "Fine-grained flood disaster information extraction incorporating multiple semantic features." International Journal of Digital Earth 18.1 (2025): 2448221.
 [https://doi.org/10.1016/j.jag.2024.104032]

### 3.2 Environmental representation of geographical units

We integrate multi-order neighborhoods into the GAT framework for geographic unit environmental characterization, with the resulting representations stored in the "./experimental_results/(1)_neighborhood_selection" folder. These results are compared with those obtained using mainstream methods such as GAE, GCN, and Node2Vec, with the representations from different methods are stored in the "./experimental_results/(2)_embeddings" folder. The corresponding code is located in the "./program_embedding" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./program_embedding/(1)_neighborhood_order_selection_CSS-HGAT.py           | Evaluate the effect of aggregating representations from different orders of neighborhoods based on <b>our method</b>.|
| ./program_embedding/(2)_environmental_representation_of_DeepWalk.py        | Calculating the environmental representation of grid cells based on the <b>DeepWalk</b> method.|
| ./program_embedding/(2)_environmental_representation_of_Node2Vec.py        | Calculating the environmental representation of grid cells based on the <b>Node2Vec</b> method.|
| ./program_embedding/(2)_environmental_representation_of_GAE.py             | Calculating the environmental representation of grid cells based on the <b>GAE</b> method.|
| ./program_embedding/(2)_environmental_representation_of_GCN.py             | Calculating the environmental representation of grid cells based on the <b>GCN</b> method.|
| ./program_embedding/(2)_environmental_representation_of_GraphSAGE.py       | Calculating the environmental representation of grid cells based on the <b>GraphSAGE</b> method.|
| ./program_embedding/(2)_evaluation_of_environmental_representation.py      | Evaluating environmental representations obtained by different methods.|
| ./program_embedding/(3)_comparison_of_clustering_results.py                | Comparing the clustering performance based on different environmental representations.|
| ./program_embedding/(3)_distribution_of_silhouette_coefficient.py          | Calculating the distribution of silhouette coefficients for clustering results based on different representations.|

The results of this process are saved in two folders: "./experimental_results/(1)_neighborhood_selection" and "./experimental_results/(2)_embeddings".

| Folder                            | Included Files                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_results/(1)_neighborhood_selection           | GAT_embeddings_predictions_1.csv, GAT_embeddings_predictions_2.csv, GAT_embeddings_predictions_3.csv, GAT_embeddings_predictions_4.csv, GAT_embeddings_predictions_5.csv, GAT_embeddings_predictions_6.csv, GAT_embeddings_predictions_7.csv, GAT_embeddings_predictions_8.csv, GAT_embeddings_predictions_9.csv, GAT_embeddings_predictions_10.csv|
| ./experimental_results/(2)_embeddings        | DeepWalk_embeddings.csv, Node2Vec_embeddings.csv, GAE_embeddings.csv, GCN_embeddings.csv, GraphSAGE_embeddings.csv, GAT_embeddings_predictions_7.csv|


### 3.3 Flood disaster risk prediction

We construct a semantic spatial correlation graph and establish local and global features. Through multi-stage label propagation, we enable flood risk prediction that accounts for changes in geographic unit attributes.

#### 3.3.1 Prediction_parameter settings

The experimental parameter settings include the selection of the number of nearest neighbor nodes and the ablation of the risk prediction module, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./program_prediction_parameterg/(1)_construction_of_semantic_graph.py           | Clustering evaluation with semantic graphs of varying nearest neighbors: the proposed CSS-HGAT-SSP method.|
| ./program_prediction_parameter/(2)_module_ablation_CSS-HGAT-Label.py           | Directly evaluates the model using the predicted label information obtained during training.|
| ./program_prediction_parameter/(2)_module_ablation_CSS-HGAT-Cluster.py         | Evaluates predictions based on the learned environmental representations of geographic units using multi-stage label propagation.|

#### 3.3.2 Spatial prediction of disaster risks

Based on the semantic similarity graph, this method integrates both local and global features of nodes to achieve spatial prediction of disaster risks. By extracting local features from nodes and combining them with environmental representations of the nodes, a feature representation that incorporates both local and global information is obtained. Further, multi-stage label propagation is employed to cluster nodes and propagate labels, utilizing the graph's structural information to refine label assignment. The corresponding code is located in the "./program_spatial_prediction" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./program_spatial_prediction/spatial_prediction_SVM.py           | Calculate the experimental results of the <b>semantic-based method SVM</b>.|
| ./program_spatial_prediction/spatial_prediction_MLP.py           | Calculate the experimental results of the <b>semantic-based method MLP</b>.|
| ./program_spatial_prediction/spatial_prediction_PageRank.py      | Calculate the experimental results of the <b>structure-based method PageRank</b>.|
| ./program_spatial_prediction/spatial_prediction_DeepWalk.py      | Calculate the experimental results of the <b>structure-based method DeepWalk</b>.|
| ./program_spatial_prediction/spatial_prediction_GCN_GraphSAGE_CSS-HGAT-SSP.py         | Calculate the experimental results of methods that combine semantic and structural information (<b>GCN, GraphSAGE, and our proposed CSS-HGAT-SSP</b>).|

The results of constructing semantic spatial association graphs based on different numbers of nearest neighbor nodes for clustering are saved in the "./experimental_results/(3)_knn_results" folder.

#### 3.3.2 Temporal prediction of disaster risks

To account for the varying influence of representations from different preceding days on the disaster status of a geographic unit on the next day, we introduce a self-attention mechanism to dynamically adjust the weights of each time step. By leveraging the Transformer model, we capture the dependencies between different time steps and integrate the environmental representations of geographic units over multiple days, allowing us to predict the geographic unit's label for the next time step. The corresponding code is located in the "./program_temporal_prediction" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./program_temporal_prediction/(1)_temporal_prediction_daily_representation.py           | Obtain unit representations for each day.|
| ./program_temporal_prediction/(2)_temporal_prediction_CSS-HGAT-Cluster.py               | Disaster risk temporal prediction results directly based on the environmental representation of each date (CSS-HGAT-Cluster)|
| ./program_temporal_prediction/(2)_temporal_prediction_CSS-HGAT-SSP.py                   | Disaster risk temporal prediction results based on CSS-HGAT-SSP (our method).|
| ./program_temporal_prediction/(2)_temporal_prediction_results_plotting.py               | Visualize the experimental comparison results of disaster risk temporal prediction as a chart.|
| ./program_temporal_prediction/(3)_temporal_prediction_save_prediction.py                | Predict the disaster risk for July 21 based on the representations from the previous four days and save the results to a file.|

The results of this process are saved in three folders: "./experimental_results/(4)_temporal_embeddings", "./experimental_results/(5)_temporal_model" and "./experimental_results/(6)_temporal_prediction".

| Folder                            | Included Files                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_results/(4)_temporal_embeddings        | embeddings_day_15.csv, embeddings_day_16.csv, embeddings_day_17.csv, embeddings_day_18.csv, embeddings_day_19.csv, embeddings_day_20.csv, embeddings_day_21.csv, embeddings_day_22.csv, embeddings_day_23.csv, embeddings_day_24.csv, embeddings_day_25.csv|
| ./experimental_results/(5)_temporal_model             | CSS-HGAT-SSP_best_model_1_days.pth, CSS-HGAT-SSP_best_model_2_days.pth,CSS-HGAT-SSP_best_model_3_days.pth, CSS-HGAT-SSP_best_model_4_days.pth, CSS-HGAT-SSP_best_model_5_days.pth|
| ./experimental_results/(6)_temporal_prediction        | predictions_21.csv|

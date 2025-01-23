# CSS-HGAT-SSP

To tackle the challenge of spatiotemporal data sparsity in disaster scenarios, we propose a method for flood risk prediction that leverages the principle of geographic similarity (<b>CSS-HGAT-SSP</b>). By aggregating information from distant neighbors, the approach enhances predictive accuracy and resilience.

The code for the proposed method includes four steps: **neighborhood order selection, risk spatial prediction based on semantic spatial association graphs, daily geographic unit representation generation, and risk temporal prediction**, which correspond to the four code files: *1_neighborhood_order_selection_CSS-HGAT.py, 2_construction_of_semantic_graph.py, 3_temporal_prediction_daily_representation.py, 4_temporal_prediction_CSS-HGAT-SSP.py*, respectively.

## 1. Data Source

All the data utilized in this study is publicly accessible and available for download online. To promote research collaboration and knowledge sharing, the sources of all datasets are listed below.
A similar description of the following data can be found in the previously published paper, "Integrated expression and analysis of urban flood disaster events from the perspective of multi-spatial semantic fusion," in the International Journal of Applied Earth Observation and Geoinformation.

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

## 2. Project Structure

Our proposed CSS-HGAT-SSP method's entire project consists of the following folders.

| Folder                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_data              | Including the initial data directly used in the experiment, which are files processed from the raw data. |
| ./experimental_results           | Including the output result files from each experimental step. |
| ./method_comparison/program_embedding              | Contains scripts for generating and evaluating embeddings using various algorithms, including CSS-HGAT, DeepWalk, GAE, GCN, GraphSAGE, and Node2Vec.|
| ./method_comparison/program_prediction_parameter   | Contains code for the selection of the number of nearest neighbor nodes and the ablation study of the risk prediction module.|
| ./method_comparison/program_spatial_prediction     | Contains code for spatial flood risk prediction using spatial relationships and neighborhood aggregation mechanisms.|
| ./method_comparison/program_temporal_prediction    | Contains code for temporal flood risk prediction based on sequential patterns and time-series modeling techniques.|

## 3. Experimental Steps

### Step 1. Data processing

After collecting the necessary experimental data, data preprocessing is conducted as the first step.

#### Step 1.1 Raster data processing

For raster data, processing is performed using Python-based ArcGIS Pro functionalities. *In the uploaded <b>0_rasterDataProcess.py</b>, functions for projection and clipping, as well as multi-value extraction to points, are implemented based on the arcpy module*. Download and install ArcGIS Pro from the following link: https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview. Subsequently, unify the coordinate system of the raster data to the Krasovsky_1940_Albers projection and extract the corresponding attribute values within each 500-meter grid.

<b>(1) Clipping and projection</b>
```
def clipProject (clip_folder,project_folder,fname):
    """
        Clips a raster to the specified extent and then projects it to a new coordinate system.

        Parameters:
        clip_folder (str): The folder path where the clipped raster will be saved.
        project_folder (str): The folder path where the projected raster will be saved.
        fname (str): The filename of the input raster.

        Returns:
        str: The file path of the projected raster.
    """
    # Clip the raster
    clipath = clip_folder + "ZhengzhouClip_" + fname
    arcpy.Clip_management(
        fname, "112 34 115 35", #The bounding rectangle extent of Zhengzhou City.
        clipath, "#", "#", "NONE")

    # Project the clipped raster.
    prjpath = project_folder + "ZhengzhouProject_" + fname
    arcpy.ProjectRaster_management(clipath, prjpath,
                                   "Krasovsky_1940_Albers.prj", "BILINEAR", "500", #Projected to the Krasovsky_1940_Albers coordinate system.
                                   "GCS_WGS_1984_to_GCS_Krasovsky_1940", "#", "#")

    return 0
```

<b>(2) Extract raster values to points</b>
```
def valuesToPoints (project_folder,fname):
    """
       Extracts raster values to points from a specified raster file.

       Parameters:
       project_folder (str): The folder path where the projected raster is located.
       fname (str): The filename of the input raster.

       Returns:
       int: Returns 0 upon successful execution.
    """
    # Define the point features to which raster values will be extracted.
    inPointFeature = "./experimental_data/Grids_in_Zhengzhou/Grid500_label.shp"

    # Define the input raster and field name.
    inTif = project_folder + "ZhengzhouProject_" + fname
    inField = fname[:-4]

    # Append the raster and field to the extract list.
    inList.append([inTif, inField])
    ExtractMultiValuesToPoints(inPointFeatures, inList, "BILINEAR")

    # Export the extracted values to a CSV file with specified fields.
    # export_ASCII = inTif[:-4] "_500.csv"
    # arcpy.ExportXYv_stats(input_features,["FID", "ID", inField], "COMMA", export_ASCII, "ADD_FIELD_NAMES")

    return 0
```

#### Step 1.2 Weibo data preprocessing

The process of extracting flood disaster impact information based on Weibo data is detailed in the published paper, "Fine-grained flood disaster information extraction incorporating multiple semantic features," in the International Journal of Digital Earth.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_data/Grid500_AllCity_Edges.csv           | Connectivity relationships between all grid units in Zhengzhou City. |
| ./experimental_data/Grid500_AllCity_Nodes.csv           | Attribute information of all grid units in Zhengzhou City. |
| ./experimental_data/Grid500_Selected_FloodArea.csv      | Disaster-affected area of grid units overlapping with flood disaster submerged zones. |
| ./experimental_data/Bound_of_Zhengzhou/Zhengzhou_Bound.shp      | Shapefile representing the administrative boundary of Zhengzhou City. |
| ./experimental_data/Grids_in_Zhengzhou/Zhengzhou_Grid500.shp      | Shapefile representing the 500-meter grid system covering Zhengzhou City. |
| ./experimental_data/Grids_in_Zhengzhou/Grid500_label.shp      | Shapefile representing the center points of each 500-meter grid within Zhengzhou City. |


### Step 2. Neighborhood order selection (Environmental representation of geographical units)

<b>(1) Selecting the optimal neighborhood for environmental representation</b>

*We integrate multi-order neighborhoods into the GAT framework for geographic unit environmental characterization, with the uploaded <b>0_rasterDataProcess.py</b>.*
The resulting representations of different neighborhood order stored in the "./experimental_results/(1)_neighborhood_selection" folder. 

<b>(2) Comparative verification</b>

These results are compared with those obtained using mainstream methods such as <b>DeepWalk, Node2Vec, GAE, GCN, and GraphSAGE</b>, with the representations from different methods are stored in the "./experimental_results/(2)_embeddings" folder. The corresponding code is located in the "./program_embedding" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./method_comparison/program_embedding/(1)_neighborhood_order_selection_CSS-HGAT.py           | Evaluate the effect of aggregating representations from different orders of neighborhoods based on <b>our method</b>.|
| ./method_comparison/program_embedding/(2)_environmental_representation_of_DeepWalk.py        | Calculating the environmental representation of grid cells based on the <b>DeepWalk</b> method.|
| ./method_comparison/program_embedding/(2)_environmental_representation_of_Node2Vec.py        | Calculating the environmental representation of grid cells based on the <b>Node2Vec</b> method.|
| ./method_comparison/program_embedding/(2)_environmental_representation_of_GAE.py             | Calculating the environmental representation of grid cells based on the <b>GAE</b> method.|
| ./method_comparison/program_embedding/(2)_environmental_representation_of_GCN.py             | Calculating the environmental representation of grid cells based on the <b>GCN</b> method.|
| ./method_comparison/program_embedding/(2)_environmental_representation_of_GraphSAGE.py       | Calculating the environmental representation of grid cells based on the <b>GraphSAGE</b> method.|
| ./method_comparison/program_embedding/(2)_evaluation_of_environmental_representation.py      | Evaluating environmental representations obtained by different methods.|
| ./method_comparison/program_embedding/(3)_comparison_of_clustering_results.py                | Comparing the clustering performance based on different environmental representations.|
| ./method_comparison/program_embedding/(3)_distribution_of_silhouette_coefficient.py          | Calculating the distribution of silhouette coefficients for clustering results based on different representations.|

The results of this process are saved in two folders: "./experimental_results/(1)_neighborhood_selection" and "./experimental_results/(2)_embeddings".

| Folder                            | Included Files                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_results/(1)_neighborhood_selection           | GAT_embeddings_predictions_1.csv, GAT_embeddings_predictions_2.csv, GAT_embeddings_predictions_3.csv, GAT_embeddings_predictions_4.csv, GAT_embeddings_predictions_5.csv, GAT_embeddings_predictions_6.csv, GAT_embeddings_predictions_7.csv, GAT_embeddings_predictions_8.csv, GAT_embeddings_predictions_9.csv, GAT_embeddings_predictions_10.csv|
| ./experimental_results/(2)_embeddings        | DeepWalk_embeddings.csv, Node2Vec_embeddings.csv, GAE_embeddings.csv, GCN_embeddings.csv, GraphSAGE_embeddings.csv, GAT_embeddings_predictions_7.csv|


### Step 3. Risk spatial prediction based on semantic association graph

We construct a semantic spatial correlation graph and establish local and global features. Through multi-stage label propagation, we enable flood risk prediction that accounts for changes in geographic unit attributes.

#### Step 3.1 Construction of the semantic association graph

*Semantic spatial association graphs are constructed by selecting 1 to 10 nearest neighbor nodes respectively, to explore the impact of different numbers of nearest neighbor nodes on spatial prediction of disaster risk, as shown in the uploaded code <b>2_construction_of_semantic_graph.py</b>.*

#### Step 3.2 Ablation of the risk prediction module
A comparison of CSS-HGAT-SSP with two other methods is conducted using the following three metrics: weighted precision, weighted recall, and micro ROC-AUC, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./pmethod_comparison/program_prediction_parameterg/(1)_construction_of_semantic_graph.py           | Clustering evaluation with semantic graphs of varying nearest neighbors: the proposed CSS-HGAT-SSP method.|
| ./method_comparison/program_prediction_parameter/(2)_module_ablation_CSS-HGAT-Label.py           | Directly evaluates the model using the predicted label information obtained during training.|
| ./method_comparison/program_prediction_parameter/(2)_module_ablation_CSS-HGAT-Cluster.py         | Evaluates predictions based on the learned environmental representations of geographic units using multi-stage label propagation.|

#### Step 3.3 Comparison of spatial prediction results for disaster risk

These spatial prediction results for disaster risk are compared with those obtained using mainstream methods such as <b>SVM, MLP, PageRank, DeepWalk, GCN, and GraphSAGE</b>. The corresponding code is located in the "./method_comparison/program_spatial_prediction" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./method_comparison/program_spatial_prediction/spatial_prediction_SVM.py           | Calculate the experimental results of the <b>semantic-based method SVM</b>.|
| ./method_comparison/program_spatial_prediction/spatial_prediction_MLP.py           | Calculate the experimental results of the <b>semantic-based method MLP</b>.|
| ./method_comparison/program_spatial_prediction/spatial_prediction_PageRank.py      | Calculate the experimental results of the <b>structure-based method PageRank</b>.|
| ./method_comparison/program_spatial_prediction/spatial_prediction_DeepWalk.py      | Calculate the experimental results of the <b>structure-based method DeepWalk</b>.|
| ./method_comparison/program_spatial_prediction/spatial_prediction_GCN_GraphSAGE_CSS-HGAT-SSP.py         | Calculate the experimental results of methods that combine semantic and structural information (<b>GCN, GraphSAGE, and our proposed CSS-HGAT-SSP</b>).|

The results of constructing semantic spatial association graphs based on different numbers of nearest neighbor nodes for clustering are saved in the "./experimental_results/(3)_knn_results" folder.

### Step 4. Temporal prediction of disaster risks

#### Step 4.1 Daily geographic unit representation generation

*Based on the aforementioned experimental results, an appropriate number of neighboring nodes is selected to generate daily unit environmental representations between July 15 and July 25. The experimental process is executed according to the uploaded code file <b>3_temporal_prediction_daily_representation.py</b>.*

#### Step 4.2 Risk temporal prediction

*By leveraging the Transformer model, we capture the dependencies between different time steps and integrate the environmental representations of geographic units over multiple days, allowing us to predict the geographic unit's label for the next time step, as shown in the uploaded code <b>4_temporal_prediction_CSS-HGAT-SSP.py</b>.* 

#### Step 4.3 Comparison of temporal prediction results for disaster risk
we compare the temporal prediction results of CSS-HGAT-Cluster based directly on the environmental representation of each date and CSS-HGAT-SSP based on the semantic association graph. The corresponding code is located in the "./method_comparison/program_temporal_prediction" folder, as detailed below.

| File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./method_comparison/program_temporal_prediction/(1)_temporal_prediction_daily_representation.py           | Obtain unit representations for each day.|
| ./method_comparison/program_temporal_prediction/(2)_temporal_prediction_CSS-HGAT-Cluster.py               | Disaster risk temporal prediction results directly based on the environmental representation of each date (CSS-HGAT-Cluster)|
| ./method_comparison/program_temporal_prediction/(2)_temporal_prediction_CSS-HGAT-SSP.py                   | Disaster risk temporal prediction results based on CSS-HGAT-SSP (our method).|
| ./method_comparison/program_temporal_prediction/(2)_temporal_prediction_results_plotting.py               | Visualize the experimental comparison results of disaster risk temporal prediction as a chart.|
| ./method_comparison/program_temporal_prediction/(3)_temporal_prediction_save_prediction.py                | Predict the disaster risk for July 21 based on the representations from the previous four days and save the results to a file.|

The results of this process are saved in three folders: "./experimental_results/(4)_temporal_embeddings", "./experimental_results/(5)_temporal_model" and "./experimental_results/(6)_temporal_prediction".

| Folder                            | Included Files                                   |
| :------------------------------------- | :---------------------------------------- |
| ./experimental_results/(4)_temporal_embeddings        | embeddings_day_15.csv, embeddings_day_16.csv, embeddings_day_17.csv, embeddings_day_18.csv, embeddings_day_19.csv, embeddings_day_20.csv, embeddings_day_21.csv, embeddings_day_22.csv, embeddings_day_23.csv, embeddings_day_24.csv, embeddings_day_25.csv|
| ./experimental_results/(5)_temporal_model             | CSS-HGAT-SSP_best_model_1_days.pth, CSS-HGAT-SSP_best_model_2_days.pth,CSS-HGAT-SSP_best_model_3_days.pth, CSS-HGAT-SSP_best_model_4_days.pth, CSS-HGAT-SSP_best_model_5_days.pth|
| ./experimental_results/(6)_temporal_prediction        | predictions_21.csv|

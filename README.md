# CSS-HGAT-SSP
flood risk prediction based on higher-order attention adaptive weighted aggregation

## 2. Structure

Due to commercial and legal restrictions in China, we are unable to share certain data. Instead, we provide mock data to demonstrate how the codes work.

| Folder/File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./mock_data/{SZ/LD}_data/              | Mock data for Shenzhen/London.            |
| ./mock_data/{SZ/LD}_data_process.ipynb | Code for processing Shenzhen/London data. |
| {SZ/LD}_train.py                       | Code for training the HGCN model.         |

## Data source
| Category |	Source |	Resolution |	Url |
| --- | --- | --- | --- |
| Precipitation distribution |	CHIRPS |	30m |	https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| DEM |	FABDEM |	30m |	https://data.bris.ac.uk/data/dataset/25wfy0f9ukoge2gs7a5mqpq2j7 |
| Landuse |	ESRI Global Land Use Data |	30m |	https://livingatlas.arcgis.com/landcoverexplorer |
| Climate |	Chinese Academy of Sciences |	1km |	https://www.resdc.cn/DOI/DOI.aspx?DOIID=96 |
| Night light index |	EOG Nighttime Light |	15" |	https://eogdata.mines.edu/nighttime_light/annual/v21/2021/ |
| Population density |	landscan-global-2021 |	30" |	https://landscan.ornl.gov/ |
| Social media data |	Weibo |	- |	https://weibo.com |
| Flood inundation |	ChinaGEOSS Data Sharing Network |	5m |	https://noda.ac.cn/datasharing/datasetDetails/62bedede0ef34854c4c2787f |

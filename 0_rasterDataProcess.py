'''Preprocessing raster data using Python-based ArcGIS Pro functionalities.'''


import arcpy
from arcpy.sa import *

arcpy.CheckOutExtension("Spatial")
arcpy.env.workspace = "./experimental_data"

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



filelist = [ ] #List of raster files, saved in the "./experimental_data" directory after downloading.

for fname in filelist:
    clip_folder = arcpy.env.workspace + "/clip"
    project_folder = arcpy.env.workspace + "/project"
    clipProject(clip_folder, project_folder, fname)
    valuesToPoints(project_folder, fname)

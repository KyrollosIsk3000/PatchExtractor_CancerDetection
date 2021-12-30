### This program is for extracting patches from CT slices that the authors of the Lung-PET-CT-Dx dataset have assigned bounding boxes to. The requirements of the patch extraction algorithm are listed below.

print("Importing packages and script files ...")
### We are going to start by importing the modules we need. NOTE: Listing the packages in alphabetical order makes it easier to find them in the list if needed.
# Standard libraries ...
from datetime import datetime # We are going to use the local date and time to name some outputs.
import os
from PIL import Image # ... for dealing with images.
import random # ... for random number generation.
import shutil # ... for doing things similar to those doable using the os package.

# External modules ...
import numpy as np
import pydicom # ... for working with DICOM files.
import pandas as pd

# Script files ...
import get_data_from_XML # The authors' get_data_from_XML.py file.
import getUID # The authors' getUID.py file.
import utils # The authors' utils.py file.

print("Finished importing packages and script files.")

### Define global variables for extracting patches.
PATCH_WIDTH = 64 # pixels.
PATCH_HEIGHT = 64 # pixels.
NUMBER_OF_PIXELS = PATCH_WIDTH * PATCH_HEIGHT


### Define global variables for sorting extracted patches.
NUMBER_OF_PIXELS_2cm = 30

# Train-Development-Holout split proportions.
PROPORTION_OF_TRAINING_PATCHES = 0.70
PROPORTION_OF_DEVELOPMENT_PATCHES = 0.15
PROPORTION_OF_HOLDOUT_PATCHES = 0.15

try:
    assert (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES + PROPORTION_OF_HOLDOUT_PATCHES) == 1.0

except AssertionError:
    print("Error: The training-development-holdout proportions do not add up to 1. Please try again.")
    exit()


### Define global values.
TIMESTAMP = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) # This timestamp will be used for naming some outputs.


### Define global directories.
DIRECTORY_ANNOTATION_FILES = "AnnotationFiles/"
DIRECTORY_PATCHES = "Patches/"
DIRECTORY_PATCHES_TRAINING = DIRECTORY_PATCHES + "Training_" + TIMESTAMP + "/"
DIRECTORY_PATCHES_DEVELOPMENT = DIRECTORY_PATCHES + "Development_" + TIMESTAMP + "/"
DIRECTORY_PATCHES_HOLDOUT = DIRECTORY_PATCHES + "Holdout_" + TIMESTAMP + "/"

DATA_FILE_NAME = "PatchExtractionSession_Info_" + TIMESTAMP + ".txt"

def make_SubdirectoriesForSubsets(SubsetDirectory):
    os.mkdir(SubsetDirectory + "Noncancerous/")
    os.mkdir(SubsetDirectory + "Cancerous/")


def remove_OldDirectories():
    REPLACE_DIRECTORIES_ANSWER = input("WARNING: Do you want to delete the patches that already exist, if any, and replace the directories? (y/n) ")
        
    if REPLACE_DIRECTORIES_ANSWER in set(["y", "Y"]):
        shutil.rmtree(DIRECTORY_PATCHES) # Remove all contents in the specified directory, but not the directory itself. [https://docs.python.org/3/library/shutil.html#shutil.rmtree]
        
        print("The old directories have been deleted.")
    
    elif REPLACE_DIRECTORIES_ANSWER in set(["n", "N"]):
            print("Ok. The patches will not be deleted. Exiting the program ...")
            exit()
        
    else:
        print("ERROR: Invalid answer. Exiting the program ...")
        exit()


def make_Directories():
    print("Making the new directories ...")
        
    os.mkdir(DIRECTORY_PATCHES_TRAINING)
    make_SubdirectoriesForSubsets(DIRECTORY_PATCHES_TRAINING)

    os.mkdir(DIRECTORY_PATCHES_DEVELOPMENT)
    make_SubdirectoriesForSubsets(DIRECTORY_PATCHES_DEVELOPMENT)

    os.mkdir(DIRECTORY_PATCHES_HOLDOUT)
    make_SubdirectoriesForSubsets(DIRECTORY_PATCHES_HOLDOUT)


if __name__ == "__main__":    
    try:
        remove_OldDirectories()
        make_Directories()

    except:
        print("ERROR: The main directory seems to not be present. Making this directory ...")
        os.mkdir(DIRECTORY_PATCHES)
        make_Directories()
        # REPLACE_DIRECTORIES_ANSWER = input("WARNING: Do you want to delete the patches that already exist, if any, and replace the directories? (y/n) ")
        
        # if REPLACE_DIRECTORIES_ANSWER in set(["y", "Y"]):
        #     shutil.rmtree(DIRECTORY_PATCHES) # Remove all contents in the specified directory, but not the directory itself. [https://docs.python.org/3/library/shutil.html#shutil.rmtree]
            
        #     print("The old directories have been deleted. Making the new directories ...")
        #     make_Directories()
        
        # elif REPLACE_DIRECTORIES_ANSWER in set(["n", "N"]):
        #     print("Ok. The patches will not be deleted. Exiting the program ...")
        #     exit()
        
        # else:
        #     print("ERROR: Invalid answer. Exiting the program ...")
        #     exit()


    ### Toggle debug mode on or off.
    DEBUG_MODE = input("Run the program in debug mode? (y/n) ")
    if DEBUG_MODE in set(["y", "Y"]):
        DEBUG_MODE_ANSWER = True
    elif DEBUG_MODE in set(["n", "N"]):
        DEBUG_MODE_ANSWER = False


    ### Define admin user inputs
    TestOrNot_PatchExtraction = input("Run the patch extraction test? (y/n) ")
    if TestOrNot_PatchExtraction in set(["y", "Y"]):
        TestOrNot_PatchExtraction_Answer = True

        Test_Patches_Locations_DF = pd.read_csv("Test1_PatchExtraction_Info.csv") # Import the locations & other data of the test patches for testing the accuracy of the patch extraction algorithm.

        ### Define global variables.
        NUMBER_OF_PATIENTS_TO_SELECT = 1
        NUMBER_OF_PATCHES_TO_EXTRACT = Test_Patches_Locations_DF.shape[0]


    elif TestOrNot_PatchExtraction in set(["n", "N"]):
        TestOrNot_PatchExtraction_Answer = False

        ### Define global variables.
        NUMBER_OF_PATIENTS_TO_SELECT = int(float(input("How many patients to select? (Minimum of 3) ")))
        try:
            assert NUMBER_OF_PATIENTS_TO_SELECT >= 3
        except:
            while NUMBER_OF_PATIENTS_TO_SELECT < 3:
                NUMBER_OF_PATIENTS_TO_SELECT = int(float(input("TRY AGAIN: How many patients to select? (Minimum of 3) ")))

        NUMBER_OF_PATCHES_TO_EXTRACT = int(float(input("How many patches to extract in total? (NOTE: Odd number is rounded up to even number.) ")))

        GROUP_BY_PATIENT = input("Would you like to group the patches by patient? (This is useful for trying to mimic k-fold cross-validation.) (y/n) ")
        if GROUP_BY_PATIENT in set(["y", "Y"]):
            GROUP_BY_PATIENT_ANSWER = True
        elif GROUP_BY_PATIENT in set(["n", "N"]):
            GROUP_BY_PATIENT_ANSWER = False


class Preparer:
    def __init__(self, NumberOfPatientsToSelect, NumberOfPatchesToExtract):
        self.NumberOfPatientsToSelect = NumberOfPatientsToSelect
        self.NumberOfPatchesToExtract = NumberOfPatchesToExtract

    ### Make a database with the directories of all the patients I want the patch extractor to consider extracting patches from. Patients not in this dictionary will be omitted from my project.
    def initialise_Databases(self): # Gather the directories of all the patients into a database.
        print("Initialising the patient data structures for the patch extraction ...")
        
        self.DatabaseOfAllPatients = []
        # Patient directories will be appended to this list in the functions that add directories to good-quality CT slices to consideration.

        self.DirectoriesOfGoodQualityCTSlices = [] # The creation of this list must be executed before the .append() method is called on it. Python would not have the list defined when it calls the method if the creation of the list is to be executed in the same function in which the .append() method is called on the list.

        # Initialise dictionaries for holding the Training, Development & Holdout subsets of each set of info.
        self.DictionariesOfSelectedPatients_Dict = {}
        self.DatabasesOfSelectedPatients_Dict = {}
        self.AnnotationDirectoryLists_Dict = {}

        print("The patient data structures have been initialised.")
        print()


    def addTo_DatabaseOfAllPatients(self, PatientDirectory):
        self.DatabaseOfAllPatients.append(PatientDirectory)


    def addTo_DirectoriesOfGoodQualityCTSlices(self, DirectoryOfCTSlices):
        self.DirectoriesOfGoodQualityCTSlices.append(DirectoryOfCTSlices)

        ### Now add the Patient Directory to the DatabaseOfAllPatients list.
        ## Make a way for me to not have to copy-paste an additional directory for each set of good-quality CT slices.
        self.BackslashCounter = 0
        self.PatientDirectory = ""

        # Extract the Patient Directory from the directory to the good-quality CT slices.
        for character_in_directory in DirectoryOfCTSlices:
            self.PatientDirectory = self.PatientDirectory + character_in_directory # Construct the Patient Directory.

            if character_in_directory == "/": # Checking for the "/" character AFTER it has been added to PatientDirectory means it will be included in PatientDirectory as its last character.
                self.BackslashCounter = self.BackslashCounter + 1
            
            if self.BackslashCounter == 2: # This is the pattern observed in the Patient Directories.
                break
        
        self.addTo_DatabaseOfAllPatients(self.PatientDirectory) # Add the extracted Patient Directory to the DatabaseOfAllPatients list.


    def specify_DirectoriesOfGoodQualityCTSlices(self): # Point to where the good-quality CT images for the patients I am considering in the project are.
        print("Locating the directories of the good-quality CT slices of the selected patients ...")
        
        # NOTE: To *add more patients for consideration*, add their directories here.
        self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/") # ... for Patient A0001.
        
        if TestOrNot_PatchExtraction_Answer == False: # Add all of the other patients I want to consider to the list if I am *not* running the patch extraction test.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0002/04-25-2007-ThoraxAThoraxRoutine-Adult-34834/3.000000-ThoraxRoutine  8.0.0  B40f-10983/") # ... for Patient A0002.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0003/07-07-2006-ThoraxAThoraxRoutine Adult-24087/3.000000-ThoraxRoutine  10.0  B40f-30728/") # ... for Patient A0003.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0004/02-15-2006-ThoraxAThoraxRoutine Adult-61611/3.000000-ThoraxRoutine  8.0.0  B40f-82719/") # ... for Patient A0004.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0005/02-24-2006-ThoraxAThoraxRoutine Adult-73190/3.000000-ThoraxRoutine  10.0  B40f-90548/") # ... for Patient A0005.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0006/04-20-2006-C-SP_Chest-19185/6.000000-5mm-16798/") # ... for Patient A0006. # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0007/12-06-2006-ThoraxAThoraxRoutine Adult-81744/3.000000-ThoraxRoutine  8.0.0  B40f-72091/") # ... for Patient A0007.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0008/07-01-2004-Chest-76304/2.000000-5mm-84471/") # ... for Patient A0008.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0009/07-31-2004-ThoraxAThoraxRoutine Adult-43867/3.000000-ThoraxRoutine  10.0  B40f-21396/")
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0010/01-20-2005-Chest-25859/3.000000-5mm-90492/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0011/02-15-2009-lungc-82781/3.000000-5mm Lung SS50-12455/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0012/08-22-2008-Chest-14151/3.000000-Recon 2 5mm-96796/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0013/02-12-2009-lungc-76661/3.000000-5mm Lung SS50-48906/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0014/09-24-2008-LUNG-83121/3.000000-5mm Lung SS50-27468/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0015/03-06-2009-Chest-02300/2.000000-5mm-06593/")
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0016/12-25-2008-CHEST-97536/3.000000-Recon 2 5mm-58376/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0018/07-24-2008-lungc-69037/3.000000-5mm Lung SS50-65517/") # Temporarily removed from the list because the FileNotFoundError was raised when, for e.g., 1-095.dcm was chosen in a directory with poor-quality CT images but a file with the same name did *not* exist in the directory with the good-quality CT images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0019/10-18-2008-lungc-39748/3.000000-d phase lung-86179/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0022/08-01-2009-Chest-72437/2.000000-5mm-82416/")
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0023/05-22-2009-LUNG-75312/3.000000-5mm Lung SS50-00560/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0024/10-03-2008-lung-47910/3.000000-Recon 2 5mm-25055/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0027/08-20-2008-Chest-83904/3.000000-5mm-30826/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0028/04-25-2009-Chest-75454/3.000000-Recon 2 5mm-61503/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0029/07-22-2009-Chest-54447/2.000000-5mm-93472/")
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0031/06-17-2009-chc-02948/3.000000-5mm Lung SS50-89984/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0032/09-17-2008-lungc-87185/3.000000-5mm Lung SS50-17574/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0036/08-12-2009-lungc-35555/3.000000-5mm Lung SS50-25105/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0037/07-29-2009-lung-25917/3.000000-5mm Lung SS50-77238/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0039/07-01-2009-lungc-82123/3.000000-5mm Lung SS50-43511/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0040/06-18-2009-lungc-16792/3.000000-5mm Lung SS50-41983/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0042/05-03-2009-lungc-77255/3.000000-5mm Lung SS50-78580/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0045/07-17-2008-LUNG-93360/3.000000-Recon 2 5mm-58897/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0049/01-29-2009-lungc-89792/3.000000-5mm Lung SS50-67526/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0050/01-29-2009-LUNG-78453/3.000000-Recon 2 5mm-99531/") # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0051 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0052 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0053 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0054 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0055 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0056 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0057/01-06-2009-Chest-40170/2.000000-5mm-16020/") # Temporarily removed due to a Key Error being raised for '1.3.6.1.4.1.14519.5.2.1.6655.2359.102888975070755246278836376242'.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0059/08-13-2009-ch-68144/3.000000-Recon 2 5mm-48628/")
            # Patient A0061 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0062 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0063/11-29-2008-Chest-76119/3.000000-Recon 2 5mm-07790/") # Temporarily removed from the list to avoid the Key Error from being raised when focusing on the "Lung_Dx-A0063/11-29-2008-Chest-76119/" directory.
            # Patient A0064 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0065 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0066 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0068 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0069 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0070 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0071 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0072 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0073 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0074 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0075 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0077 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0078 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0080 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0081 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0082 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0083 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0084 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0085/07-19-2010-lung-38482/3.000000-5mm Lung-66818/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0086/07-26-2010-ch-16992/3.000000-5mm Lung SS50-89263/")
            # Patient A0087 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0088 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0089 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0090 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0091 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0092/07-28-2010-Chest-89190/3.000000-Recon 2 5mm-50918/")
            # Patient A0093 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0094 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0095 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0096 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0097/02-13-2010-chest-15956/3.000000-Recon 2 5mm-14655/")
            # Patient A0098 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0099 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0100 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0101 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0102 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0103 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0104 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0105 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0106 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0107 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0108 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0109 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0110 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0111 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0112/05-24-2011-lung-18939/3.000000-5mm Lung SS50-23969/") # Temporarily removed from the list to avoid a Key Error.
            # Patient A0113 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0114 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0115 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0116/05-20-2011-chest-46964/3.000000-Recon 2 5mm-29584/")
            # Patient A0117 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0118 # Temporarily removed from the list to avoid a Key Error.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0119/04-24-2011-chest-72805/102.000000-5mm-15192/")
            # Patient A0120 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0121 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0123 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0124/07-23-2010-lung-86797/3.000000-Recon 2 5mm-40508/")
            # Patient A0125 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0126 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0127 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0128 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0129 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0130 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0131 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0132/04-30-2011-Chest  3D IMR-47106/205.000000-FBP-61810/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0133/02-18-2011-Chest-74926/2.000000-5mm-62219/")
            # Patient A0134 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0135 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0136 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0137/03-13-2011-Chest-83131/5.000000-Recon 2 5mm-57816/")
            # Patient A0138 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0139 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0140 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0141 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0142 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0143 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0144 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0145 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0146 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0147 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0148 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0149/05-18-2011-LUNG-09390/3.000000-Recon 2 5mm-35402/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0150/05-20-2011-LUNGC-58460/3.000000-5mm Lung SS50-25639/")
            # Patient A0152 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0153 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0154/08-28-2010-Chest-55785/2.000000-5mm-12477/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0155/03-03-2010-Chest-98689/2.000000-5mm-99218/")
            # Patient A0156 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0157 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0159 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0160 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0161 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0162/03-11-2011-LUNG-95510/3.000000-5mm Lung SS50-27755/")
            # Patient A0163 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0164/04-12-2010-PET01PTheadlung Adult-08984/8.000000-Thorax  1.0  B31f-52757/")
            # Patient A0165 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0166 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0167 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0168 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0169 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0170 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0171 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0173 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0174 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0175 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0176 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0177 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0178 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0179 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0180 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0181 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0182/07-12-2009-PET07PTheadlung Adult-05523/8.000000-Thorax  1.0  B31f-31674/")
            # Patient A0183 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0184 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0185 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0187/08-06-2009-PET07PTheadlung Adult-59218/8.000000-Thorax  1.0  B31f-54495/")
            # Patient A0188 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0189 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0190 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0191 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0192 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0193 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0194/03-04-2011-PET01PTheadlung Adult-01183/9.000000-Thorax  1.0  B31f-99191/")
            # Patient A0195 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0196 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0197 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0198 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0199 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0200 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0201 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0202 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0203 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0204 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0205/10-22-2009-PET03WholebodyFirstHead Adult-27784/8.000000-Thorax  1.0  B31f-69374/")
            # Patient A0206 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0208 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0210 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0211 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0212 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0213 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0214 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0216 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0217/05-05-2010-PET01PTheadlung Adult-80947/8.000000-Thorax  1.0  B31f-22555/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0218/11-17-2010-PET01PTheadlung Adult-43342/9.000000-Thorax  1.0  B31f-34123/")
            # Patient A0220 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0221 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0222 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0223 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0224 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0225 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0226 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0227 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0228 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0229 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0230 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0231 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0232 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0233 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0234 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0235 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0236 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0237 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0238 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0239 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0240 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0241 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0242 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0243 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0244 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0246 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0247 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0248 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0249 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0250 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0251 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0252 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0253 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0254 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0255 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0256/12-10-2010-PET01PTheadlung Adult-01751/9.000000-Thorax  1.0  B31f-83142/")
            # Patient A0257 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0258/11-11-2010-PET01PTheadlung Adult-02764/9.000000-Thorax  1.0  B31f-95235/")
            # Patient A0259 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0260/12-02-2010-PET01PTheadlung Adult-92650/9.000000-Thorax  1.0  B31f-06249/")
            self.addTo_DirectoriesOfGoodQualityCTSlices("Lung_Dx-A0261/02-23-2011-PET01PTheadlung Adult-75442/9.000000-Thorax  1.0  B31f-01106/")
            # Patient A0262 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0263 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0264 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.
            # Patient A0265 # Temporarily removed from the list to avoid the FileNotFoundError from being raised in the case where a DICOM file in the directory of poor-quality images is chosen but not all DICOM files in that directory are matched with corresponding files with the same name in the directory of good-quality images.

        self.MaximumNumberOfPatientsToConsider = len(self.DirectoriesOfGoodQualityCTSlices)
        
        if self.NumberOfPatientsToSelect > self.MaximumNumberOfPatientsToConsider: # Avoid an error due to expecting more patients than how many are in the list for consideration.
            self.NumberOfPatientsToSelect = self.MaximumNumberOfPatientsToConsider
            print("WARNING: You expected to consider more patients than how many were in the list for consideration. Now considering the max. number of patients, i.e., {} patients ...".format(self.MaximumNumberOfPatientsToConsider))
        
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.DirectoriesOfGoodQualityCTSlices =", self.DirectoriesOfGoodQualityCTSlices)
            # print()
            ####

        print("The directories of the good-quality CT slices have been located.")
        print()


    ### Randomly select a set of patients to extract patches from. Make sure that each patient is only in one subset, i.e., in only one of either the Training, Development or Holdout subsets.
    def determine_NumbersOfPatientsBySubset(self, NumberOfPatientsToSelect):
        # The 3 assignments below take care of the Training:Development:Holdout split proportions.
        self.NumberOfPatients_Training = int(NumberOfPatientsToSelect * PROPORTION_OF_TRAINING_PATCHES)
        self.NumberOfPatients_Holdout = int(NumberOfPatientsToSelect * PROPORTION_OF_HOLDOUT_PATCHES)
        self.NumberOfPatients_Development = NumberOfPatientsToSelect - self.NumberOfPatients_Training - self.NumberOfPatients_Holdout # int() rounds a number down to the nearest integer, so the Development subset will always be larger than the Holdout subset if the same proportions have been assigned to them.
        
        self.PlacesInTheSubsetsToAllocateToPatients_List = ([DIRECTORY_PATCHES_TRAINING] * self.NumberOfPatients_Training) + ([DIRECTORY_PATCHES_DEVELOPMENT] * self.NumberOfPatients_Development) + ([DIRECTORY_PATCHES_HOLDOUT] * self.NumberOfPatients_Holdout)
        assert (len(self.PlacesInTheSubsetsToAllocateToPatients_List) == NumberOfPatientsToSelect) # Make sure there are enough places for the selected patients but no more places than the number of selected patients.
    
    
    def allocate_PatientToSubset(self): # Randomly assign the subset places in PlacesInTheSubsetsToAllocateToPatients_List to the selected patients.
        if TestOrNot_PatchExtraction_Answer == False:
            # Choose a subset for the patient.
            self.RandomIndex_SubsetPlace = random.randint(0, len(self.PlacesInTheSubsetsToAllocateToPatients_List) - 1)
            self.SubsetForPatient = self.PlacesInTheSubsetsToAllocateToPatients_List[self.RandomIndex_SubsetPlace]

            # Each place in the PlacesInTheSubsetsToAllocateToPatients_List shall be taken by only one patient. Remove the selected place b/c a patient has now taken it.
            self.PlacesInTheSubsetsToAllocateToPatients_List.remove(self.PlacesInTheSubsetsToAllocateToPatients_List[self.RandomIndex_SubsetPlace])

        elif TestOrNot_PatchExtraction_Answer == True:
            self.SubsetForPatient = "Patches_FromFiji/TestPatches/PatchExtraction_Test/"
        
        return self.SubsetForPatient


    # def allocate_PatientToSubset(self):
    #     if TestOrNot_PatchExtraction_Answer == False:
    #         # Generate a random number.
    #         # If it is < PROPORTION_OF_TRAINING_PATCHES, allocate to Training subset.
    #         # If it is >= PROPORTION_OF_TRAINING_PATCHES but < (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES), allocate to Development subset.
    #         # If it is >= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES) but <= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES + PROPORTION_OF_HOLDOUT_PATCHES), allocate to Holdout subset.
    #         self.RandomNumber_AllocatePatientToSubset = random.uniform(a = 0.0, b = 1.0)
            
    #         if self.RandomNumber_AllocatePatientToSubset < PROPORTION_OF_TRAINING_PATCHES:
    #             self.SubsetForPatient = DIRECTORY_PATCHES_TRAINING
            
    #         elif (self.RandomNumber_AllocatePatientToSubset >= PROPORTION_OF_TRAINING_PATCHES) and (self.RandomNumber_AllocatePatientToSubset < (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES)):
    #             self.SubsetForPatient = DIRECTORY_PATCHES_DEVELOPMENT
            
    #         elif (self.RandomNumber_AllocatePatientToSubset >= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES)) and (self.RandomNumber_AllocatePatientToSubset <= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES + PROPORTION_OF_HOLDOUT_PATCHES)):
    #             self.SubsetForPatient = DIRECTORY_PATCHES_HOLDOUT

    #     elif TestOrNot_PatchExtraction_Answer == True:
    #         self.SubsetForPatient = "Patches_FromFiji/TestPatches/PatchExtraction_Test/"
        
    #     return self.SubsetForPatient
    

    def make_DatabaseOfSelectedPatients(self):
        print("Choosing patients for patch extraction ...")
        
        self.DatabaseOfSelectedPatients = [] # Start a new list. This list will contain ALL selected patients before they are to be allocated into the 3 subsets. NOTE: The filled version of this list is the output of this method.
        self.NumberOfPatientsToSelectFrom = len(self.DatabaseOfAllPatients)

        self.determine_NumbersOfPatientsBySubset(self.NumberOfPatientsToSelect)

        for patient in range(0, self.NumberOfPatientsToSelect):
            self.SubsetForPatient = self.allocate_PatientToSubset() # Randomly select a subset to allocate the patient to.
            
            self.PatientToSelect = random.randint(a = 0, b = self.NumberOfPatientsToSelectFrom - 1) # Generate a random *index* for a patient to select. Select a patient at random.
            self.DatabaseOfSelectedPatients.append((self.DatabaseOfAllPatients[self.PatientToSelect], self.SubsetForPatient)) # Record the selected patient's *directory* & data subset into the set of patients to later extract patches from.
            
            # Prepare for the next time a patient is to be selected.
            self.DatabaseOfAllPatients.remove(self.DatabaseOfAllPatients[self.PatientToSelect]) # ... so a patient is not selected more than once.
            self.NumberOfPatientsToSelectFrom = len(self.DatabaseOfAllPatients) # ... to avoid a random index from being too high for the modified DatabaseOfAllPatients.

            if self.NumberOfPatientsToSelectFrom == 0: # A precaution against bugs.
                break

        print("Patients have now been chosen.")
        print()
            
            
    def move_PatientFromSubsetToSubset(self, FromSubset, ToSubset):
        ToSubset.append(FromSubset.pop()) # Remove the last patient from the FromSubset & put them into the ToSubset. The pop() method returns what it removes from the list.

        return FromSubset, ToSubset # ... for updating the lists.

    
    def initialise_DatabasesForPatientsBySubset(self):
        self.DatabaseOfSelectedPatients_Training = []
        self.DatabaseOfSelectedPatients_Development = []
        self.DatabaseOfSelectedPatients_Holdout = []
    
    
    def separate_PatientsBySubset(self): # This method includes code that makes sure the Training-Development_Holdout split is achieved successfully.
        print("Separating patients by subset ...")
        
        self.initialise_DatabasesForPatientsBySubset()

        for patient_directory, patient_subset_directory in self.DatabaseOfSelectedPatients:
            if patient_subset_directory == DIRECTORY_PATCHES_TRAINING:
                self.DatabaseOfSelectedPatients_Training.append((patient_directory, patient_subset_directory))

            elif patient_subset_directory == DIRECTORY_PATCHES_DEVELOPMENT:
                self.DatabaseOfSelectedPatients_Development.append((patient_directory, patient_subset_directory))
            
            elif patient_subset_directory == DIRECTORY_PATCHES_HOLDOUT:
                self.DatabaseOfSelectedPatients_Holdout.append((patient_directory, patient_subset_directory))
        
        
        ## Make sure there is at least 1 patient in each subset.
        # Identify which subset has no patients.
        self.List_Check_HasNoPatients = [0, 0, 0] # A list for keeping track of which & how many subsets have no patients. The order of values is [Training, Development, Holdout].

        self.NumberOfPatients_Training = len(self.DatabaseOfSelectedPatients_Training)
        self.NumberOfPatients_Development = len(self.DatabaseOfSelectedPatients_Development)
        self.NumberOfPatients_Holdout = len(self.DatabaseOfSelectedPatients_Holdout)

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("Before moving patients between subsets ...")
            # print("len(self.DatabaseOfSelectedPatients_Training) =", self.NumberOfPatients_Training)
            # print("len(self.DatabaseOfSelectedPatients_Development) =", self.NumberOfPatients_Development)
            # print("len(self.DatabaseOfSelectedPatients_Holdout) =", self.NumberOfPatients_Holdout)
            # print()
            ####

        if self.NumberOfPatients_Training == 0:
            print("The Training subset has no patients.")
            self.List_Check_HasNoPatients[0] = 1
        
        if self.NumberOfPatients_Development == 0:
            print("The Development subset has no patients.")
            self.List_Check_HasNoPatients[1] = 1
        
        if self.NumberOfPatients_Holdout == 0:
            print("The Holdout subset has no patients.")
            self.List_Check_HasNoPatients[2] = 1
        
        if sum(self.List_Check_HasNoPatients) > 0: # The primary aim of moving patients from a subset to another is to proceed with the program, i.e., each subset must have at least 1 patient in it. The split is according to user discretion.
            print("Moving patients from subset to subset so that each subset has at least 1 patient ...")

            # Constraints:
                # The Training subset must have the most patients for training the model as best as possible. However, exceptions may occur according to user discretion.
                # The Development subset must have enough patients for testing the trained model.
            
            # Make sure the Training subset has patients in it. This subset is the most important one.
            while ((self.NumberOfPatients_Training == 0) or (self.NumberOfPatients_Development == 0) or (self.NumberOfPatients_Holdout == 0)):
                try:
                    assert self.NumberOfPatients_Training > 0
                
                except AssertionError: # A minimum of 3 patients must be chosen for patch extraction. This means that, at this minimum number, either one subset has 1 patient & the other has 2, or one subset has 3 patients & the other also has 0.
                    # In this case, take a patient from the subset with the most patients & put them into the Training subset.
                    if self.NumberOfPatients_Development > self.NumberOfPatients_Holdout: # This IF statement will be True even if there are no patients in the Holdout subset. There would be at least 2 patients in the Development subset.
                        self.DatabaseOfSelectedPatients_Development, self.DatabaseOfSelectedPatients_Training = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Development, ToSubset = self.DatabaseOfSelectedPatients_Training)

                    elif self.NumberOfPatients_Holdout >= self.NumberOfPatients_Development: # This IF statement will be True even if there are no patients in the Development subset. There would be at least 2 patients in the Holdout subset in this case. Also, the Development subset is more important than the Holdout subset, so there should ideally be more patients in the Development subset than in the Holdout subset. The minimum number of patients in each of these subsets must be 2 for the "==" case in this IF statement to be executed & be True.
                        self.DatabaseOfSelectedPatients_Holdout, self.DatabaseOfSelectedPatients_Training = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Holdout, ToSubset = self.DatabaseOfSelectedPatients_Training)
                    
                    
                try:
                    assert self.NumberOfPatients_Development > 0
                
                except AssertionError: # A minimum of 3 patients must be chosen for patch extraction. This means that, at this minimum number, either one subset has 1 patient & the other has 2, or one subset has 3 patients & the other also has 0.
                    # In this case, take a patient from the subset with the most patients & put them into the Development subset.
                    if self.NumberOfPatients_Training > self.NumberOfPatients_Holdout: # This IF statement will be True even if there are no patients in the Holdout subset. There would be at least 2 patients in the Training subset.
                        self.DatabaseOfSelectedPatients_Training, self.DatabaseOfSelectedPatients_Development = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Training, ToSubset = self.DatabaseOfSelectedPatients_Development)

                    elif self.NumberOfPatients_Holdout >= self.NumberOfPatients_Training: # This IF statement will be True even if there are no patients in the Development subset. There would be at least 2 patients in the Holdout subset in this case. Also, the Development subset is more important than the Holdout subset, so there should ideally be more patients in the Development subset than in the Holdout subset. The minimum number of patients in each of these subsets must be 2 for the "==" case in this IF statement to be executed & be True.
                        self.DatabaseOfSelectedPatients_Holdout, self.DatabaseOfSelectedPatients_Development = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Holdout, ToSubset = self.DatabaseOfSelectedPatients_Development)
                    
                    
                try:
                    assert self.NumberOfPatients_Holdout > 0
                
                except AssertionError: # A minimum of 3 patients must be chosen for patch extraction. This means that, at this minimum number, either one subset has 1 patient & the other has 2, or one subset has 3 patients & the other also has 0.
                    # In this case, take a patient from the subset with the most patients & put them into the Holdout subset.
                    if self.NumberOfPatients_Training > self.NumberOfPatients_Development: # This IF statement will be True even if there are no patients in the Holdout subset. There would be at least 2 patients in the Training subset.
                        self.DatabaseOfSelectedPatients_Training, self.DatabaseOfSelectedPatients_Holdout = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Training, ToSubset = self.DatabaseOfSelectedPatients_Holdout)

                    elif self.NumberOfPatients_Development >= self.NumberOfPatients_Training: # This IF statement will be True even if there are no patients in the Training subset. There would be at least 2 patients in the Development subset in this case. Also, the Development subset is more important than the Holdout subset, so there should ideally be more patients in the Development subset than in the Holdout subset. The minimum number of patients in each of these subsets must be 2 for the "==" case in this IF statement to be executed & be True.
                        self.DatabaseOfSelectedPatients_Development, self.DatabaseOfSelectedPatients_Holdout = self.move_PatientFromSubsetToSubset(FromSubset = self.DatabaseOfSelectedPatients_Development, ToSubset = self.DatabaseOfSelectedPatients_Holdout)

                self.NumberOfPatients_Training = len(self.DatabaseOfSelectedPatients_Training)
                self.NumberOfPatients_Development = len(self.DatabaseOfSelectedPatients_Development)
                self.NumberOfPatients_Holdout = len(self.DatabaseOfSelectedPatients_Holdout)
            
            # Now all 3 subsets should have at least 1 patient each.
            
            
        # Check that there is at least 1 patient in each subset.
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("After moving patients between subsets ...")
            # print("len(self.DatabaseOfSelectedPatients_Training) =", self.NumberOfPatients_Training)
            # print("len(self.DatabaseOfSelectedPatients_Development) =", self.NumberOfPatients_Development)
            # print("len(self.DatabaseOfSelectedPatients_Holdout) =", self.NumberOfPatients_Holdout)
            # print()
            ####

        assert self.NumberOfPatients_Training > 0
        assert self.NumberOfPatients_Development > 0
        assert self.NumberOfPatients_Holdout > 0

        assert (self.NumberOfPatients_Training + self.NumberOfPatients_Development + self.NumberOfPatients_Holdout == self.NumberOfPatientsToSelect) # Make sure no selected patient is omitted from any subset.
        # assert ((self.NumberOfPatients_Training > self.NumberOfPatients_Development) and (self.NumberOfPatients_Training > self.NumberOfPatients_Holdout)) # The Training subset must have the most patients for training the model as best as possible.

        print("There is/are {} patient(s) in the Training subset.".format(self.NumberOfPatients_Training))
        print("There is/are {} patient(s) in the Development subset.".format(self.NumberOfPatients_Development))
        print("There is/are {} patient(s) in the Holdout subset.".format(self.NumberOfPatients_Holdout))
        print("The Training : Development : Holdout split is {0:0.3f}%".format((self.NumberOfPatients_Training / self.NumberOfPatientsToSelect) * 1e2), ": {0:0.3f}%".format((self.NumberOfPatients_Development / self.NumberOfPatientsToSelect) * 1e2), ": {0:0.3f}%.".format((self.NumberOfPatients_Holdout / self.NumberOfPatientsToSelect) * 1e2))
        self.SplitSuitability_Answer = input("Is this Training : Development : Holdout split suitable? (y/n) ")

        if self.SplitSuitability_Answer in set(["y", "Y"]):
            print("Ok. Extracting patches ...") # Proceed with the program.
        
        elif self.SplitSuitability_Answer in set(["n", "N"]):
            print("Ok. You said the split was not suitable. Exiting the program ...")
            print("Please modify the number of patients to select &/or the Training : Development : Holdout split proportions & rerun the program.")
            exit()
        
        else:
            print("ERROR: Invalid answer. Exiting the program ...")
            exit()

        print("The patients have now been separated by subset.")
        print()
    
    
    ### Construct a dictionary with the SOP Instance UIDs, i.e., XML annotation file names, as keys and tuples of the paths to the DICOM files & the DICOM filenames as values. At this stage, I no longer care about which patient the patches came from, so I may dissociate the DICOM files from the patients. 
    def make_DictionariesOfSelectedPatients(self):
        print("Putting the patients into one master dictionary for patch extraction ...")
        
        self.DictionaryOfSelectedPatients_Training = {} # Start a new dictionary.
        self.DictionaryOfSelectedPatients_Development = {} # Start a new dictionary.
        self.DictionaryOfSelectedPatients_Holdout = {} # Start a new dictionary.
        # Each dictionary: 1 key-value pair per Annotation File directory.
            # Key: Annotation File directory relative to "AnnotationFiles/", i.e., Patient Name (minus "Lung_Dx-"). E.g., "A0001".
            # Value:
                # Tuple:
                    # Dictionary: 1 key-value pair per SOP Instance UID.
                        # Key: SOP Instance UID (XML annotation filename). NOTE: These SOP Instance UIDs do *not* necessarily belong to the good-quality CT slices.
                        # Value: (Path to the DICOM file, DICOM filename)
                    # Data subset allocated to the patient.
        
        for patient_directory, patient_subset_directory in self.DatabaseOfSelectedPatients_Training:
            self.PatientDictionary = getUID.getUID_path(patient_directory) # Get a dictionary of that patient which has their SOP Instance UIDs & DICOM file paths & names. The DICOM files with the SOP Instance UIDs here are the ones the authors' visualisation code shows.
            # Dictionary: 1 key-value pair per SOP Instance UID.
                # Key: SOP Instance UID (XML annotation filename). NOTE: These SOP Instance UIDs do *not* necessarily belong to the good-quality CT slices.
                # Value: (Path to the DICOM file, DICOM filename)

            # Add the PatientDictionary to the DictionaryOfSelectedPatients.
            self.DictionaryOfSelectedPatients_Training[patient_directory[8:13]] = (self.PatientDictionary, patient_subset_directory) # Put all patients' dictionaries into one master dictionary. & associate their subsets to them.
        

        for patient_directory, patient_subset_directory in self.DatabaseOfSelectedPatients_Development:
            self.PatientDictionary = getUID.getUID_path(patient_directory) # Get a dictionary of that patient which has their SOP Instance UIDs & DICOM file paths & names. The DICOM files with the SOP Instance UIDs here are the ones the authors' visualisation code shows.
            # Dictionary: 1 key-value pair per SOP Instance UID.
                # Key: SOP Instance UID (XML annotation filename). NOTE: These SOP Instance UIDs do *not* necessarily belong to the good-quality CT slices.
                # Value: (Path to the DICOM file, DICOM filename)

            # Add the PatientDictionary to the DictionaryOfSelectedPatients.
            self.DictionaryOfSelectedPatients_Development[patient_directory[8:13]] = (self.PatientDictionary, patient_subset_directory) # Put all patients' dictionaries into one master dictionary. & associate their subsets to them.


        for patient_directory, patient_subset_directory in self.DatabaseOfSelectedPatients_Holdout:
            self.PatientDictionary = getUID.getUID_path(patient_directory) # Get a dictionary of that patient which has their SOP Instance UIDs & DICOM file paths & names. The DICOM files with the SOP Instance UIDs here are the ones the authors' visualisation code shows.
            # Dictionary: 1 key-value pair per SOP Instance UID.
                # Key: SOP Instance UID (XML annotation filename). NOTE: These SOP Instance UIDs do *not* necessarily belong to the good-quality CT slices.
                # Value: (Path to the DICOM file, DICOM filename)

            # Add the PatientDictionary to the DictionaryOfSelectedPatients.
            self.DictionaryOfSelectedPatients_Holdout[patient_directory[8:13]] = (self.PatientDictionary, patient_subset_directory) # Put all patients' dictionaries into one master dictionary. & associate their subsets to them.

        
    def merge_DictionariesOfSelectedPatients(self):
        self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_TRAINING] = self.DictionaryOfSelectedPatients_Training
        self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_DEVELOPMENT] = self.DictionaryOfSelectedPatients_Development
        self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_HOLDOUT] = self.DictionaryOfSelectedPatients_Holdout

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.DictionariesOfSelectedPatients_Dict =", self.DictionariesOfSelectedPatients_Dict)
            # print()
            ####

        print("The patients are now in one master dictionary for patch extraction.")
        print()
    

    def merge_DatabasesOfSelectedPatients(self):
        self.DatabasesOfSelectedPatients_Dict[DIRECTORY_PATCHES_TRAINING] = self.DatabaseOfSelectedPatients_Training
        self.DatabasesOfSelectedPatients_Dict[DIRECTORY_PATCHES_DEVELOPMENT] = self.DatabaseOfSelectedPatients_Development
        self.DatabasesOfSelectedPatients_Dict[DIRECTORY_PATCHES_HOLDOUT] = self.DatabaseOfSelectedPatients_Holdout

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.DatabasesOfSelectedPatients_Dict =", self.DatabasesOfSelectedPatients_Dict)
            # print()
            ####
    
    
    def make_AnnotationDirectoryListsDictionary(self):
        print("Merging annotation directories into another master dictionary for patch extraction ...")
        
        self.AnnotationDirectoryLists_Dict[DIRECTORY_PATCHES_TRAINING] = list(self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_TRAINING].keys()) # REFERENCE: https://www.geeksforgeeks.org/python-dictionary-keys-method/
        self.AnnotationDirectoryLists_Dict[DIRECTORY_PATCHES_DEVELOPMENT] = list(self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_DEVELOPMENT].keys())
        self.AnnotationDirectoryLists_Dict[DIRECTORY_PATCHES_HOLDOUT] = list(self.DictionariesOfSelectedPatients_Dict[DIRECTORY_PATCHES_HOLDOUT].keys())

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.AnnotationDirectoryLists_Dict =", self.AnnotationDirectoryLists_Dict)
            # print()
            ####
        
        print("The annotation directories are now in their own master dictionary.")
        print()


    def count_NumberOfXMLFilesTheSelectedPatientsHave(self):
        print("Counting the number of XML annotation files the selected patients have ...")
        
        self.NumberOfXMLFilesOfAllSelectedPatients = 0 # Initialise this variable.
        self.XMLFilesOfAllSelectedPatients_List = [] # ... for disregarding the files not named in the DictionariesOfSelectedPatients_Dict dictionary.

        for subset_directory in self.AnnotationDirectoryLists_Dict:
            for patient_name_index in range(0, len(self.AnnotationDirectoryLists_Dict[subset_directory])):
                self.PathToXMLFilesForPatient = os.path.join(DIRECTORY_ANNOTATION_FILES, self.AnnotationDirectoryLists_Dict[subset_directory][patient_name_index]).replace("\\", "/")
                self.XMLFilesForPatient = os.listdir(self.PathToXMLFilesForPatient)
                self.NumberOfXMLFilesForPatient = len(self.XMLFilesForPatient)

                self.NumberOfXMLFilesOfAllSelectedPatients = self.NumberOfXMLFilesOfAllSelectedPatients + self.NumberOfXMLFilesForPatient
                self.XMLFilesOfAllSelectedPatients_List.extend(self.XMLFilesForPatient)
        
        print("There are", self.NumberOfXMLFilesOfAllSelectedPatients, "XML annotation files for the selected patients.")
        print()


    def count_NumberOfGoodQualityCTImagesPatchesCouldBeExtractedFrom(self):
        print("Counting the number of good-quality CT images patches could be extracted from ...")

        ### All of the info I need for this counting is in the DictionariesOfSelectedPatients_Dict dictionary.
        self.DictionariesOfSelectedPatients_ForCounting_Dict = self.DictionariesOfSelectedPatients_Dict # Make a copy of this dictionary in case I modify it for the counting.
        self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = 0 # Initialise this counter.
        self.XMLFilesOfAllSelectedPatients_Dict = {xml_filename : 0 for xml_filename in self.XMLFilesOfAllSelectedPatients_List}

        ### Remove the XML filenames from XMLFilesOfAllSelectedPatients_List that are not in the DictionariesOfSelectedPatients_Dict dictionary.
        # Loop over every SOP Instance UID in the DictionariesOfSelectedPatients_Dict dictionary.
        for subset_directory in self.DictionariesOfSelectedPatients_Dict.keys():
            for patient_name in self.DictionariesOfSelectedPatients_Dict[subset_directory].keys():
                for SOPInstanceUID in self.DictionariesOfSelectedPatients_Dict[subset_directory][patient_name][0].keys():
                    if (SOPInstanceUID + ".xml") in self.XMLFilesOfAllSelectedPatients_List:
                        self.XMLFilesOfAllSelectedPatients_Dict[SOPInstanceUID + ".xml"] = 1

        
        # self.XMLFilesOfAllSelectedPatients_Dict_Copy = self.XMLFilesOfAllSelectedPatients_Dict

        # self.XMLFilenames_ForCounting_List = []
        # self.WhetherOrNotToKeep_List = []
        # for xml_filename, whether_or_not_to_keep in self.XMLFilesOfAllSelectedPatients_Dict.items():
        #     self.XMLFilenames_ForCounting_List.append(xml_filename)
        #     self.WhetherOrNotToKeep_List.append(whether_or_not_to_keep)

        # self.XMLFilenames_ForCounting_List_Copy = self.XMLFilenames_ForCounting_List
        
        # for whether_or_not_to_keep_Index in range(0, len(self.WhetherOrNotToKeep_List)):
        #     if self.WhetherOrNotToKeep_List[whether_or_not_to_keep_Index] == 0:
        #         self.XMLFilenames_ForCounting_List.pop(self.XMLFilenames_ForCounting_List_Copy[whether_or_not_to_keep_Index]) # Use a copy of the list whose length is being modified to avoid the IndexError "list index out of range" error.
        
        # for xml_filename in self.XMLFilesOfAllSelectedPatients_Dict_Copy.keys():
        #     if self.XMLFilesOfAllSelectedPatients_Dict[xml_filename] == 0:
        #         self.XMLFilesOfAllSelectedPatients_Dict_Copy[xml_filename] = self.XMLFilesOfAllSelectedPatients_Dict.pop(xml_filename)

        for xml_filename in self.XMLFilesOfAllSelectedPatients_Dict.keys():
            self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom + self.XMLFilesOfAllSelectedPatients_Dict[xml_filename] # self.XMLFilesOfAllSelectedPatients_Dict[xml_filename] == either 0 or 1; 1 if the IF condition above is True.

        # self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = len(self.XMLFilenames_ForCounting_List)


        # # NOTES:
        #     # If a SOP Instance UID belongs to a poor-quality DICOM file, it is used to lead to a good-quality DICOM file.
        #     # The question then becomes, ‘How many good-quality DICOM files could my patch extractor extract patches from?’ However, perhaps there are good-quality DICOM files that do not have XML annotation files but their corresponding poor-quality DICOM files do.
        #         # Get the SOP Instance UIDs of all of the good-quality DICOM files for all selected patients, or at least the good-quality DICOM files I have specified.
        #         # Determine how many of these SOP Instance UIDs have an XML annotation file belonging to them.

        # ## Get the SOP Instance UIDs of all of the good-quality DICOM files for all selected patients, or at least the good-quality DICOM files I have specified.
        # self.SOPInstanceUIDsOfAllGoodQualityDICOMFiles_ForCounting_List = []

        # for path_to_good_quality_dicom_files_of_patient in self.DirectoriesOfGoodQualityCTSlices:
        #     self.GoodQualityDICOMFiles_ForCounting_List = os.listdir(path_to_good_quality_dicom_files_of_patient)

        #     for good_quality_dicom_file in self.GoodQualityDICOMFiles_ForCounting_List:
        #         self.PathToGoodQualityDICOMFile_ForCounting = os.path.join(path_to_good_quality_dicom_files_of_patient, good_quality_dicom_file).replace("\\", "/")

        #         self.DICOMInfo_ForCounting = pydicom.read_file(self.PathToGoodQualityDICOMFile_ForCounting) # Get the SOP Instance UID of the DICOM file.
        #         self.SOPInstanceUID_ForCounting = self.DICOMInfo_ForCounting.SOPInstanceUID

        #         self.SOPInstanceUIDsOfAllGoodQualityDICOMFiles_ForCounting_List.append(self.SOPInstanceUID_ForCounting)


        # ### Determine how many of these SOP Instance UIDs have an XML annotation file belonging to them.
        # for path_to_good_quality_dicom_files_of_patient in self.DirectoriesOfGoodQualityCTSlices: # ... after SOPInstanceUIDsOfAllGoodQualityDICOMFiles_ForCounting_List has been completed.
        #     # Extract the patient's name from the directory.
        #     self.PatientName_ForCounting = ((path_to_good_quality_dicom_files_of_patient.split("/"))[0].split("-"))[1]
        #     self.PathToXMLFilesForPatient_ForCounting = os.path.join(DIRECTORY_ANNOTATION_FILES, self.PatientName_ForCounting).replace("\\", "/")
        #     self.XMLFilenamesOfPatient_ForCounting_List = os.listdir(self.PathToXMLFilesForPatient_ForCounting) # NOTE: The filenames have the ".xml" extension.

        #     for xml_filename in self.XMLFilenamesOfPatient_ForCounting_List:
        #         if xml_filename.replace(".xml", "") in self.SOPInstanceUIDsOfAllGoodQualityDICOMFiles_ForCounting_List:
        #             self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom + 1


        # ## However, perhaps there are good-quality DICOM files that do not have XML annotation files but their corresponding poor-quality DICOM files do.
        # # Get the paths to the good-quality DICOM files.
        # self.PathsToGoodQualityDICOMFiles_ForCounting_List = []

        # for subset_directory in self.DictionariesOfSelectedPatients_ForCounting_Dict.keys():
        #     for patient_name in self.DictionariesOfSelectedPatients_ForCounting_Dict[subset_directory].keys():
        #         for SOPInstanceUID in self.DictionariesOfSelectedPatients_ForCounting_Dict[subset_directory][patient_name][0].keys():
        #             if SOPInstanceUID in self.SOPInstanceUIDsOfAllGoodQualityDICOMFiles_ForCounting_List:
        #                 self.PathToGoodQualityDICOMFile_ForCounting = self.DictionariesOfSelectedPatients_ForCounting_Dict[subset_directory][patient_name][0][SOPInstanceUID][0]
        #                 self.PathsToGoodQualityDICOMFiles_ForCounting_List.append(self.PathToGoodQualityDICOMFile_ForCounting)

        # # Identify the SOP Instance UIDs in the DictionariesOfSelectedPatients_ForCounting_Dict dictionary belong to poor-quality DICOM files with the same filenames as the good-quality ones in the same patient.
        # for path_to_good_quality_dicom_file in self.PathsToGoodQualityDICOMFiles_ForCounting_List:
        #     self.PathToGoodQualityDICOMFile_ForCounting_Split = path_to_good_quality_dicom_file.replace("\\", "/").split("/")
        #     self.ProbablyDifferentPathElementInPath_GoodQualityDICOMFile = self.PathToGoodQualityDICOMFile_ForCounting_Split[-2] # This part of the path to the DICOM file is what ultimately leads to a good- or poor-quality DICOM file.
        #     self.PathToGoodQualityDICOMFile_ForCounting_Split.remove(self.ProbablyDifferentPathElementInPath_GoodQualityDICOMFile) # Prepare the path for comparison to that of the corresponding poor-quality DICOM file with the same filename.

        #     # Find the path to the DICOM file with the same filename in the 'other' folder in the patient & remove the corresponding SOP Instance UID from the DictionariesOfSelectedPatients_ForCounting_Dict dictionary.
        #     self.PatientName_ForCounting = ((path_to_good_quality_dicom_file.split("/"))[0].split("-"))[1] # Access the contents of this patient in the DictionariesOfSelectedPatients_ForCounting_Dict dictionary.
        #     if self.PatientName_ForCounting in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_TRAINING].keys():
        #         for SOPInstanceUID in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_TRAINING][self.PatientName_ForCounting][0].keys():
        #             self.PathToDICOMFile_ForCounting = self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_TRAINING][self.PatientName_ForCounting][0][SOPInstanceUID][0]
        #             self.PathToDICOMFile_ForCounting_Split = self.PathToDICOMFile_ForCounting.split("/")
        #             self.ProbablyDifferentPathElementInPath = self.PathToDICOMFile_ForCounting_Split[-2]
        #             self.PathToDICOMFile_ForCounting_Split.remove(self.ProbablyDifferentPathElementInPath)
                    
        #             if self.PathToGoodQualityDICOMFile_ForCounting_Split == self.PathToDICOMFile_ForCounting_Split:
        #                 self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom + 1
        #                 break

            
        #     elif self.PatientName_ForCounting in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_DEVELOPMENT].keys():
        #         for SOPInstanceUID in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_DEVELOPMENT][self.PatientName_ForCounting][0].keys():
        #             self.PathToDICOMFile_ForCounting = self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_DEVELOPMENT][self.PatientName_ForCounting][0][SOPInstanceUID][0]
        #             self.PathToDICOMFile_ForCounting_Split = self.PathToDICOMFile_ForCounting.split("/")
        #             self.ProbablyDifferentPathElementInPath = self.PathToDICOMFile_ForCounting_Split[-2]
        #             self.PathToDICOMFile_ForCounting_Split.remove(self.ProbablyDifferentPathElementInPath)
                    
        #             if self.PathToGoodQualityDICOMFile_ForCounting_Split == self.PathToDICOMFile_ForCounting_Split:
        #                 self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom + 1
        #                 break


        #     elif self.PatientName_ForCounting in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_HOLDOUT].keys():
        #         for SOPInstanceUID in self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_HOLDOUT][self.PatientName_ForCounting][0].keys():
        #             self.PathToDICOMFile_ForCounting = self.DictionariesOfSelectedPatients_ForCounting_Dict[DIRECTORY_PATCHES_HOLDOUT][self.PatientName_ForCounting][0][SOPInstanceUID][0]
        #             self.PathToDICOMFile_ForCounting_Split = self.PathToDICOMFile_ForCounting.split("/")
        #             self.ProbablyDifferentPathElementInPath = self.PathToDICOMFile_ForCounting_Split[-2]
        #             self.PathToDICOMFile_ForCounting_Split.remove(self.ProbablyDifferentPathElementInPath)
                    
        #             if self.PathToGoodQualityDICOMFile_ForCounting_Split == self.PathToDICOMFile_ForCounting_Split:
        #                 self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom + 1
        #                 break
        

        print("(??????????) There are", self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom, "good-quality CT images from all the selected patients that patches can be extracted from.")
        print()
    

    def get_NumberOfClasses(self):
        self.ClassList = get_data_from_XML.get_category("category.txt")
        self.NumberOfClasses = len(self.ClassList)
        return self.NumberOfClasses
    

    # def transfer_Objects(self):
    #     return self.AnnotationDirectoryLists_Dict, self.DictionariesOfSelectedPatients_Dict, self.DirectoriesOfGoodQualityCTSlices, self.DatabasesOfSelectedPatients_Dict
    

    def export_PreparationData(self): # Make a record of the details of the patch extraction session.
        # Contents to display in the file:
            # Number of patients in the Training subset.
            # Number of patients in the Development subset.
            # Number of patients in the Holdout subset.
            # Total number of patients who were selected.
            # Training-Development-Test split.
            # Number of patches that were extracted.
        self.InfoFile = open(DIRECTORY_PATCHES + DATA_FILE_NAME , "w")
        self.InfoFile_Contents_Intro = "### Information for the Patch Extraction session started on {} (Year-Month-Day_Hour-Minute-Second) ###\n".format(TIMESTAMP)
        self.InfoFile_Contents_NumberOfTrainingPatients = "Number of Training patients = {}\n".format(self.NumberOfPatients_Training)
        self.InfoFile_Contents_NumberOfDevelopmentPatients = "Number of Development patients = {}\n".format(self.NumberOfPatients_Development)
        self.InfoFile_Contents_NumberOfHoldoutPatients = "Number of Holdout patients = {}\n".format(self.NumberOfPatients_Holdout)
        self.InfoFile_Contents_TotalNumberOfPatientsSelected = "Total number of patients selected = {}\n".format(self.NumberOfPatientsToSelect)
        self.InfoFile_Contents_TrainingDevelopmentHoldoutSplit = "Training : Development : Holdout split = {0:0.3f}%".format((self.NumberOfPatients_Training / self.NumberOfPatientsToSelect) * 1e2) + " : {0:0.3f}%".format((self.NumberOfPatients_Development / self.NumberOfPatientsToSelect) * 1e2) + " : {0:0.3f}%\n".format((self.NumberOfPatients_Holdout / self.NumberOfPatientsToSelect) * 1e2)

        self.InFoFile_Info = self.InfoFile_Contents_Intro + self.InfoFile_Contents_NumberOfTrainingPatients + self.InfoFile_Contents_NumberOfDevelopmentPatients + self.InfoFile_Contents_NumberOfHoldoutPatients + self.InfoFile_Contents_TotalNumberOfPatientsSelected + self.InfoFile_Contents_TrainingDevelopmentHoldoutSplit
        
        self.InfoFile.write(self.InFoFile_Info)
        self.InfoFile.close()
        

### Extract patches from random locations in the CT slices.
class PatchExtractor(Preparer):
    def __init__(self, NumberOfPatientsToSelect, NumberOfPatchesToExtract, AnnotationDirectoryLists_Dict, DictionariesOfSelectedPatients_Dict, DirectoriesOfGoodQualityCTSlices, DatabasesOfSelectedPatients_Dict, NumberOfXMLFilesOfAllSelectedPatients, Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom):
        super().__init__(NumberOfPatientsToSelect, NumberOfPatchesToExtract)
        self.NumberOfPatientsToSelect = NumberOfPatientsToSelect
        self.NumberOfPatchesToExtract = NumberOfPatchesToExtract
        self.NumberOfPatchesExtracted = 0
        self.AnnotationDirectoryLists_Dict = AnnotationDirectoryLists_Dict
        self.DictionariesOfSelectedPatients_Dict = DictionariesOfSelectedPatients_Dict
        
        self.DirectoriesOfGoodQualityCTSlices = DirectoriesOfGoodQualityCTSlices
        
        self.DatabasesOfSelectedPatients_Dict = DatabasesOfSelectedPatients_Dict

        self.NumberOfXMLFilesOfAllSelectedPatients = NumberOfXMLFilesOfAllSelectedPatients
        self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.NumberOfPatientsToSelect =", self.NumberOfPatientsToSelect)
            # print()
            # print("self.NumberOfPatchesToExtract =", self.NumberOfPatchesToExtract)
            # print()
            ####
    
    
    def get_DirectoryToGoodQualityCTSlicesOfPatient(self, SubsetBeingFocusedOn_Directory, DictionaryOfSelectedPatients, RandomPatient): # E.g., RandomPatient = "A0001".
        for patient_directory_to_good_quality_slices in self.DirectoriesOfGoodQualityCTSlices:
            if RandomPatient in patient_directory_to_good_quality_slices:
                return patient_directory_to_good_quality_slices # E.g., = "Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/".
        
        
    def makeSure_SelectedDirectoryIsSpecifiedDirectory(self, SubsetBeingFocusedOn_Directory, DictionaryOfSelectedPatients, PathToRandomCTSlice, RandomPatient):
        # self.SelectedPatientDirectory = self.get_SelectedPatientDirectory(SubsetBeingFocusedOn_Directory, DictionaryOfSelectedPatients, RandomPatient)
        self.PathToGoodQualityCTSlicesOfPatient = self.get_DirectoryToGoodQualityCTSlicesOfPatient(SubsetBeingFocusedOn_Directory, DictionaryOfSelectedPatients, RandomPatient)
        # E.g., = "Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/".

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.PathToGoodQualityCTSlicesOfPatient =", self.PathToGoodQualityCTSlicesOfPatient)
            # print()
            # print("PathToRandomCTSlice =", PathToRandomCTSlice)
            # print()
            ####

        self.SelectedPatient_PathToRandomDICOMFile_ComponentsList = PathToRandomCTSlice.replace("\\", "/").split("/") # E.g., ["Lung_Dx-A0001", "04-04-2007-Chest-07990", "(Other folder)", "1-18.dcm"].
        self.PathToGoodQualityCTSlicesOfPatient_ComponentsList = self.PathToGoodQualityCTSlicesOfPatient.split("/") # E.g., = ["Lung_Dx-A0001", "04-04-2007-Chest-07990", "2.000000-5mm-40805"].

        self.SpecifiedPatientDICOMFolder = self.PathToGoodQualityCTSlicesOfPatient_ComponentsList[2]

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("BEFORE IF: self.SelectedPatient_PathToRandomDICOMFile_ComponentsList =", self.SelectedPatient_PathToRandomDICOMFile_ComponentsList)
            # print()
            ####

        if self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[2] != self.SpecifiedPatientDICOMFolder:
            self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[2] = self.SpecifiedPatientDICOMFolder
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("AFTER IF: self.SelectedPatient_PathToRandomDICOMFile_ComponentsList =", self.SelectedPatient_PathToRandomDICOMFile_ComponentsList)
            # print()
            ####
        
        self.PathToGoodQualityDICOMFile = os.path.join(self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[0], self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[1], self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[2], self.SelectedPatient_PathToRandomDICOMFile_ComponentsList[3]).replace("\\", "/")

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.PathToGoodQualityDICOMFile =", self.PathToGoodQualityDICOMFile)
            # print()
            ####
        
        return self.PathToGoodQualityDICOMFile # E.g., "Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/1-18.dcm".
    
    
    def choose_SubsetToFocusOnForPatchExtraction(self):
        if TestOrNot_PatchExtraction_Answer == False:
            # Generate a random number.
            # If it is < PROPORTION_OF_TRAINING_PATCHES, allocate to Training subset.
            # If it is >= PROPORTION_OF_TRAINING_PATCHES but < (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES), allocate to Development subset.
            # If it is >= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES) but <= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES + PROPORTION_OF_HOLDOUT_PATCHES), allocate to Holdout subset.
            self.RandomNumber_SubsetToFocusOnForPatchExtraction = random.uniform(a = 0.0, b = 1.0)

            if self.RandomNumber_SubsetToFocusOnForPatchExtraction < PROPORTION_OF_TRAINING_PATCHES:
                self.SubsetToFocusOnForPatchExtraction_Directory = DIRECTORY_PATCHES_TRAINING # This value corresponds to one of the keys of the DictionariesOfSelectedPatients_Dict master dictionary.
            
            elif (self.RandomNumber_SubsetToFocusOnForPatchExtraction >= PROPORTION_OF_TRAINING_PATCHES) and (self.RandomNumber_SubsetToFocusOnForPatchExtraction < (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES)):
                self.SubsetToFocusOnForPatchExtraction_Directory = DIRECTORY_PATCHES_DEVELOPMENT # This value corresponds to one of the keys of the DictionariesOfSelectedPatients_Dict master dictionary.
            
            elif (self.RandomNumber_SubsetToFocusOnForPatchExtraction >= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES)) and (self.RandomNumber_SubsetToFocusOnForPatchExtraction <= (PROPORTION_OF_TRAINING_PATCHES + PROPORTION_OF_DEVELOPMENT_PATCHES + PROPORTION_OF_HOLDOUT_PATCHES)):
                self.SubsetToFocusOnForPatchExtraction_Directory = DIRECTORY_PATCHES_HOLDOUT # This value corresponds to one of the keys of the DictionariesOfSelectedPatients_Dict master dictionary.

        elif TestOrNot_PatchExtraction_Answer == True:
            self.SubsetForPatient = "Patches_FromFiji/TestPatches/PatchExtraction_Test/"
        
        return self.SubsetToFocusOnForPatchExtraction_Directory
    
    
    def choose_CTSlice(self, SubsetToFocusOnForPatchExtraction = None): # Choose a CT slice which has an associated XML annotation file at random.
        if SubsetToFocusOnForPatchExtraction is None:
            self.SubsetToFocusOnForPatchExtraction = self.choose_SubsetToFocusOnForPatchExtraction() # This value corresponds to one of the keys of the DictionariesOfSelectedPatients_Dict master dictionary.
        
        else:
            self.SubsetToFocusOnForPatchExtraction = SubsetToFocusOnForPatchExtraction # ... for when I want to extract the next patch from a patient in the same subset as the patient I just extracted a patch from.
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.AnnotationDirectoryLists_Dict[{}] =".format(self.SubsetToFocusOnForPatchExtraction_Directory), self.AnnotationDirectoryLists_Dict[self.SubsetToFocusOnForPatchExtraction_Directory])
            # print()
            ####
        
        if TestOrNot_PatchExtraction_Answer == False:
            self.RandomPatient = self.AnnotationDirectoryLists_Dict[self.SubsetToFocusOnForPatchExtraction_Directory][random.randint(a = 0, b = len(self.AnnotationDirectoryLists_Dict[self.SubsetToFocusOnForPatchExtraction_Directory]) - 1)] # E.g. output: "A0001".

            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                # print("self.SubsetToFocusOnForPatchExtraction_Directory =", self.SubsetToFocusOnForPatchExtraction_Directory)
                # print("self.RandomPatient =", self.RandomPatient)
                # print()
                ####
                
        elif TestOrNot_PatchExtraction_Answer == True:
            # self.RandomPatient = self.AnnotationDirectoryList[0] # Select Patient A0001.
            pass
        
        
        self.Annotations = get_data_from_XML.XML_preprocessor(data_path = (DIRECTORY_ANNOTATION_FILES + self.RandomPatient), num_classes = self.get_NumberOfClasses()).data # Get the XML annotation filenames for the randomly selected patient.
        # Dictionary:
            # Key: SOP Instance UID. NOTE: These SOP Instance UIDs do *not* necessarily belong to the good-quality CT slices. They are the ones that are shown when the authors' visualisation code is run.
            # Value: numpy array equivalent of [[4 elements for bounding box coordinates, 4 elements for class label vector]]

        # Select a random CT slice, i.e., select a random SOP Instance UID.        
        self.Annotations_SOPInstanceUID_List = list(self.Annotations.keys()) # Make a list for the use of random.randint() for randomly selecting a SOP Instance UID.
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.Annotations_SOPInstanceUID_List =", self.Annotations_SOPInstanceUID_List)
            # print()
            ####
       
        self.PathToRandomCTSlice = None
        self.WhileLoopCounter_PathToRandomCTSlice = 0
        self.ErrorList_RandomSOPInstanceUIDs = []
        while self.PathToRandomCTSlice is None: # Keep obtaining SOP Instance UIDs until a chosen one exists in the DictionaryOfSelectedPatients. [https://stackoverflow.com/questions/4606919/in-python-try-until-no-error]
            if DEBUG_MODE_ANSWER == True:
                pass
                self.WhileLoopCounter_PathToRandomCTSlice = self.WhileLoopCounter_PathToRandomCTSlice + 1
                print("Iteration {} of the WHILE loop for obtaining self.PathToRandomCTSlice ...".format(self.WhileLoopCounter_PathToRandomCTSlice))

            try:
                if TestOrNot_PatchExtraction_Answer == False:
                    self.RandomSOPInstanceUID = str((self.Annotations_SOPInstanceUID_List[random.randint(a = 0, b = len(self.Annotations_SOPInstanceUID_List) - 1)]).replace(".xml", "")) # Choose a random SOP Instance UID from the patient. E.g. output: XML filename excluding .xml extension, i.e., a SOP Instance UID.

                    while self.RandomSOPInstanceUID in self.ErrorList_RandomSOPInstanceUIDs: # Do not try using a SOP Instance UID that leads to an error again.
                        self.RandomSOPInstanceUID = str((self.Annotations_SOPInstanceUID_List[random.randint(a = 0, b = len(self.Annotations_SOPInstanceUID_List) - 1)]).replace(".xml", "")) # Choose a random SOP Instance UID from the patient. E.g. output: XML filename excluding .xml extension, i.e., a SOP Instance UID.

                    
                    if DEBUG_MODE_ANSWER == True:
                        pass
                        #### debug
                        # print("self.RandomSOPInstanceUID =", self.RandomSOPInstanceUID)
                        # print("self.RandomPatient =", self.RandomPatient)
                        # print("self.SubsetToFocusOnForPatchExtraction_Directory =", self.SubsetToFocusOnForPatchExtraction_Directory)
                        # print("self.DictionariesOfSelectedPatients_Dict[{}] =".format(self.SubsetToFocusOnForPatchExtraction_Directory), self.DictionariesOfSelectedPatients_Dict[self.SubsetToFocusOnForPatchExtraction_Directory])
                        # print()
                        ####
                    
                elif TestOrNot_PatchExtraction_Answer == True:
                    self.RandomSOPInstanceUID = str(Test_Patches_Locations_DF["SOP Instance UID"][self.NumberOfPatchesExtracted]) # self.NumberOfPatchesExtracted is the same number as the row number of the DataFrame that I want.

                
                # Access the corresponding CT slice & check if it is in the directory I specified for the given patient.
                self.PathToRandomCTSlice = (self.DictionariesOfSelectedPatients_Dict[self.SubsetToFocusOnForPatchExtraction][self.RandomPatient][0][self.RandomSOPInstanceUID][0]).replace("\1", "/1") # Get the path to the randomly selected DICOM file.
                # E.g., = "Lung_Dx-A0001/04-04-2007-Chest-07990/(Other folder)/".

                if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("self.PathToRandomCTSlice =", self.PathToRandomCTSlice)
                    print()
                    ####

                # Check if the path to the randomly selected DICOM file is in the directory I specified for the patient. If it is not, select the DICOM file with the same name in the specified directory & apply the bounding box to it.
                self.PathToGoodQualityDICOMFile = self.makeSure_SelectedDirectoryIsSpecifiedDirectory(self.SubsetToFocusOnForPatchExtraction_Directory, self.DictionariesOfSelectedPatients_Dict[self.SubsetToFocusOnForPatchExtraction_Directory], self.PathToRandomCTSlice, self.RandomPatient) # Check if the path to the randomly selected DICOM file is in the directory I specified for the patient. If it is not, get the path to the good-quality DICOM file in the specified directory.
                # E.g., self.PathToGoodQualityDICOMFile = "Lung_Dx-A0001/04-04-2007-Chest-07990/2.000000-5mm-40805/1-18.dcm".
                
            except:
                if DEBUG_MODE_ANSWER == True:
                    pass
                    # print("ERROR encountered. Going into the next iteration of the WHILE loop ...")
                
                self.ErrorList_RandomSOPInstanceUIDs.append(self.RandomSOPInstanceUID) # This list is a blacklist. Do not try using a SOP Instance UID that leads to an error again.

                if len(self.ErrorList_RandomSOPInstanceUIDs) == len(self.Annotations_SOPInstanceUID_List):
                    print("All of the SOP Instance UIDs of Patient", self.RandomPatient, "have led to a Key Error. Exiting the program ...")
                    print("Please fix the issue &/or remove the patient from consideration.")
                    exit()
  
        
    def get_ImageDimensions(self, PathToGoodQualityDICOMFile):
        self.DICOMDataOfGoodQualityDICOMFile = pydicom.read_file(PathToGoodQualityDICOMFile)
        
        # Get the dimensions of the CT image.
        self.DICOMFile_ImageWidth = self.DICOMDataOfGoodQualityDICOMFile.Columns # ... along the x-axis.
        self.DICOMFile_ImageHeight = self.DICOMDataOfGoodQualityDICOMFile.Rows # ... along the y-axis.

        return self.DICOMFile_ImageWidth, self.DICOMFile_ImageHeight
    
    
    def get_BoundingBoxCoordinates(self, Annotations_SOPInstanceUID_List, Annotations, RandomSOPInstanceUID):
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.Annotations =", self.Annotations)
            # print()
            # print("self.RandomSOPInstanceUID + '.xml' =", self.RandomSOPInstanceUID + ".xml")
            # print()
            ####
        
        
        self.BoundingBoxCoordinatesAndClassLabelVector = Annotations[RandomSOPInstanceUID + ".xml"] # Get the numpy array equivalent of [[4 elements for bounding box coordinates, 4 elements for class label vector]] corresponding to the randomly selected SOP Instance UID.
        self.BoundingBoxCoordinates = self.BoundingBoxCoordinatesAndClassLabelVector[0][0:4]

        self.xMin_BoundingBox = self.BoundingBoxCoordinates[0]
        self.yMin_BoundingBox = self.BoundingBoxCoordinates[1]
        self.xMax_BoundingBox = self.BoundingBoxCoordinates[2]
        self.yMax_BoundingBox = self.BoundingBoxCoordinates[3]

        return self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox
    

    def calculate_CentreCoordinatesOfBoundingBox(self, xMin_BoundingBox, yMin_BoundingBox, xMax_BoundingBox, yMax_BoundingBox):
        self.x_CentrePixel_BoundingBox = int((xMax_BoundingBox + xMin_BoundingBox) // 2)
        self.y_CentrePixel_BoundingBox = int((yMax_BoundingBox + yMin_BoundingBox) // 2)

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("xMin_BoundingBox =", xMin_BoundingBox)
            # print("yMin_BoundingBox =", yMin_BoundingBox)
            # print("xMax_BoundingBox =", xMax_BoundingBox)
            # print("yMax_BoundingBox =", yMax_BoundingBox)
            # print()
            # print("self.x_CentrePixel_BoundingBox =", self.x_CentrePixel_BoundingBox)
            # print("self.y_CentrePixel_BoundingBox =", self.y_CentrePixel_BoundingBox)
            # print()
            ####

        return self.x_CentrePixel_BoundingBox, self.y_CentrePixel_BoundingBox


    def add_MarginToBoundingBox(self): # Add a margin to the bounding box equivalent to approximately 2 cm.
        self.xMin_BoundingBox_Expanded = self.xMin_BoundingBox - NUMBER_OF_PIXELS_2cm
        self.yMin_BoundingBox_Expanded = self.yMin_BoundingBox - NUMBER_OF_PIXELS_2cm
        self.xMax_BoundingBox_Expanded = self.xMax_BoundingBox + NUMBER_OF_PIXELS_2cm
        self.yMax_BoundingBox_Expanded = self.yMax_BoundingBox + NUMBER_OF_PIXELS_2cm
    
    
    def calculate_PatchCornersCoordinates(self):
        self.xMin_Patch = self.x_Patch_TopLeft
        self.yMin_Patch = self.y_Patch_TopLeft
        self.xMax_Patch = self.xMin_Patch + PATCH_WIDTH - 1
        self.yMax_Patch = self.yMin_Patch + PATCH_HEIGHT - 1

        return self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch
    
    
    def check_Patch_CornerInExpandedBoundingBox(self, x_Patch, y_Patch):
        if (x_Patch in range(int(self.xMin_BoundingBox_Expanded), int(self.xMax_BoundingBox_Expanded + 1))) and (y_Patch in range(int(self.yMin_BoundingBox_Expanded), int(self.yMax_BoundingBox_Expanded + 1))):
            return True
        else:
            return False


    def extract_APatch(self):
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.PathToGoodQualityDICOMFile =", self.PathToGoodQualityDICOMFile)
            ####
        
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            print("In extract_APatch(self) ...")
            ####

        ### Extract a patch.
        self.DICOMFile_ImageWidth, self.DICOMFile_ImageHeight = self.get_ImageDimensions(self.PathToGoodQualityDICOMFile) # Get the dimensions of the CT image in the specified directory.

        if TestOrNot_PatchExtraction_Answer == False:
            # Define random (x, y) coordinates for the top-left corner of the patch to extract from the CT image.
            # self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) * self.DICOMFile_ImageWidth # Define a random location for the patch without going outside the dimensions of the CT image.
            # self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT)) * self.DICOMFile_ImageHeight
            self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define a random location for the patch without going outside the dimensions of the CT image.
            self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))
        
        elif TestOrNot_PatchExtraction_Answer == True:
            # Extract a patch listed in the patch extraction test CSV file.
            self.x_Patch_TopLeft = Test_Patches_Locations_DF["x"][self.NumberOfPatchesExtracted]
            self.y_Patch_TopLeft = Test_Patches_Locations_DF["y"][self.NumberOfPatchesExtracted]


        # Extract the patch from the specified DICOM file/image.
        self.GoodQualityDICOMFile_Array, self.frame_num, self.width, self.height, self.ch = utils.loadFile(self.PathToGoodQualityDICOMFile) # Express the CT image as a numpy array. I am interested only in the array. The array has shape (ch, rows (y), columns (x)).
        
        # Extract a patch.
        self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file. For e.g., 1-18.dcm in the "2.0..." folder of Patient A0001 has an array of shape (1, 512, 512).
        self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
        self.Patch_Array = np.array(self.Patch_DF)

        while self.check_Patch_NotLungTissue(): # Make sure the patch is not from outside the body.
            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                print("In extract_APatch(self) ... while self.check_Patch_NotLungTissue(): ...")
                ####
            
            # Choose another location to extract a patch from.
            self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define a random location for the patch without going outside the dimensions of the CT image.
            self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))
            
            # Extract the patch. The DICOM file has already been chosen outside of this WHILE loop.
            self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file. For e.g., 1-18.dcm in the "2.0..." folder of Patient A0001 has an array of shape (1, 512, 512).
            self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
            self.Patch_Array = np.array(self.Patch_DF)

        # Now it is time to sort the patch.

        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            # print("self.GoodQualityDICOMFile_DF =", self.GoodQualityDICOMFile_DF)
            # print()
            # print("self.y_Patch_TopLeft =", self.y_Patch_TopLeft)
            # print("self.x_Patch_TopLeft =", self.x_Patch_TopLeft)
            # print()
            # print("self.Patch_DF =", self.Patch_DF)
            # print()
            # print("self.Patch_Array =", self.Patch_Array)
            # print()
            ####
    

    def extract_APatch_Noncancerous(self):
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            print("In extract_APatch_Noncancerous(self) ...")
            ####
        
        ### Extract a patch.
        self.DICOMFile_ImageWidth, self.DICOMFile_ImageHeight = self.get_ImageDimensions(self.PathToGoodQualityDICOMFile) # Get the dimensions of the CT image in the specified directory. Updated.

        self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox = self.get_BoundingBoxCoordinates(self.Annotations_SOPInstanceUID_List, self.Annotations, self.RandomSOPInstanceUID) # Update the bounding box coordinates.
        self.add_MarginToBoundingBox() # Update these values for the new bounding box.

        if TestOrNot_PatchExtraction_Answer == False:
            # Define updated random (x, y) coordinates for the top-left corner of the patch to extract from the CT image such that the patch is outside of the expanded bounding box.
            self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define a random location for the patch without going outside the dimensions of the CT image.
            self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))

            self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates() # Update these values.

            while (self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMax_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMax_Patch)): # If this WHILE statement is True, the patch is not Noncancerous. Therefore, a new location for the patch must be chosen.
                self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define another random location for the patch with the aim of extracting a Noncancerous patch. Also, do not going outside the dimensions of the CT image.
                self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))

                self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates()

            # Extract the patch from the specified DICOM file/image.
            self.GoodQualityDICOMFile_Array, self.frame_num, self.width, self.height, self.ch = utils.loadFile(self.PathToGoodQualityDICOMFile) # Express the CT image as a numpy array. I am interested only in the array. The array has shape (ch, rows (y), columns (x)).
            
            # Extract a patch.
            self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file.
            self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
            self.Patch_Array = np.array(self.Patch_DF)
            # Now it is time to sort the patch.

            # self.check_Patch_NotLungTissue()

            while self.check_Patch_NotLungTissue(): # Repeat this loop until the patch is of lung tissue.
                if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("In extract_APatch_Noncancerous(self) ... while self.check_Patch_NotLungTissue(): ...")
                    ####
                
                ### Extract another patch in place of the previous patch. Make sure it is actually a Noncancerous patch.
                self.DICOMFile_ImageWidth, self.DICOMFile_ImageHeight = self.get_ImageDimensions(self.PathToGoodQualityDICOMFile) # Get the dimensions of the CT image in the specified directory.

                self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox = self.get_BoundingBoxCoordinates(self.Annotations_SOPInstanceUID_List, self.Annotations, self.RandomSOPInstanceUID) # Update the bounding box coordinates.
                self.add_MarginToBoundingBox()

                # Define updated random (x, y) coordinates for the top-left corner of the patch to extract from the CT image such that the patch is outside of the expanded bounding box.
                self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define a random location for the patch without going outside the dimensions of the CT image.
                self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))

                self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates() # Update these values.

                while (self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMax_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMax_Patch)): # If this WHILE statement is True, the patch is not Noncancerous. Therefore, a new location for the patch must be chosen.
                    self.x_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageWidth - PATCH_WIDTH)) # Define another random location for the patch with the aim of extracting a Noncancerous patch. Also, do not going outside the dimensions of the CT image.
                    self.y_Patch_TopLeft = (random.randint(0, self.DICOMFile_ImageHeight - PATCH_HEIGHT))

                    self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates()

                # Extract the patch from the specified DICOM file/image.
                self.GoodQualityDICOMFile_Array, self.frame_num, self.width, self.height, self.ch = utils.loadFile(self.PathToGoodQualityDICOMFile) # Express the CT image as a numpy array. I am interested only in the array. The array has shape (ch, rows (y), columns (x)).
                
                # Extract a patch.
                self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file.
                self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
                self.Patch_Array = np.array(self.Patch_DF)
                # Now it is time to sort the patch.

                # self.check_Patch_NotLungTissue()

            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                # print("self.GoodQualityDICOMFile_DF =", self.GoodQualityDICOMFile_DF)
                # print()
                # print("self.y_Patch_TopLeft =", self.y_Patch_TopLeft)
                # print("self.x_Patch_TopLeft =", self.x_Patch_TopLeft)
                # print()
                # print("self.Patch_DF =", self.Patch_DF)
                # print()
                # print("self.Patch_Array =", self.Patch_Array)
                # print()
                ####
    

    def extract_APatch_Cancerous(self):
        ### Extract a patch.
        self.DICOMFile_ImageWidth, self.DICOMFile_ImageHeight = self.get_ImageDimensions(self.PathToGoodQualityDICOMFile) # Get the dimensions of the CT image in the specified directory. Updated.

        self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox = self.get_BoundingBoxCoordinates(self.Annotations_SOPInstanceUID_List, self.Annotations, self.RandomSOPInstanceUID) # Update the bounding box coordinates.
        self.x_CentrePixel_BoundingBox, self.y_CentrePixel_BoundingBox = self.calculate_CentreCoordinatesOfBoundingBox(self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox) # Calculate the coordinates of the centre pixel of the updated bounding box.

        if TestOrNot_PatchExtraction_Answer == False:
            # Define updated random (x, y) coordinates for the top-left corner of the patch to extract from the CT image such that the patch has the bounding box's centre pixel in it.
            self.MinX_CancerousPatch = int(self.x_CentrePixel_BoundingBox - PATCH_WIDTH + 1)
            self.MaxX_CancerousPatch = int(self.x_CentrePixel_BoundingBox - 1)
            self.MinY_CancerousPatch = int(self.y_CentrePixel_BoundingBox - PATCH_HEIGHT + 1)
            self.MaxY_CancerousPatch = int(self.y_CentrePixel_BoundingBox - 1)

            self.x_Patch_TopLeft = random.randint(self.MinX_CancerousPatch, self.MaxX_CancerousPatch) # Define a random location for a Cancerous patch without going outside the dimensions of the CT image.
            self.y_Patch_TopLeft = random.randint(self.MinY_CancerousPatch, self.MaxY_CancerousPatch)

            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                print("In extract_APatch_Cancerous(self) ...")
                # print("self.x_CentrePixel_BoundingBox =", self.x_CentrePixel_BoundingBox)
                # print("self.y_CentrePixel_BoundingBox =", self.y_CentrePixel_BoundingBox)
                # print()
                # print("self.MinX_CancerousPatch =", self.MinX_CancerousPatch)
                # print("self.MaxX_CancerousPatch =", self.MaxX_CancerousPatch)
                # print()
                # print("self.MinY_CancerousPatch =", self.MinY_CancerousPatch)
                # print("self.MaxY_CancerousPatch =", self.MaxY_CancerousPatch)
                # print()
                # print("self.x_Patch_TopLeft =", self.x_Patch_TopLeft)
                # print("self.y_Patch_TopLeft =", self.y_Patch_TopLeft)
                # print()
                ####
            
            # Extract the patch from the specified DICOM file/image.
            self.GoodQualityDICOMFile_Array, self.frame_num, self.width, self.height, self.ch = utils.loadFile(self.PathToGoodQualityDICOMFile) # Express the CT image as a numpy array. I am interested only in the array. The array has shape (ch, rows (y), columns (x)).
            
            # Extract a patch.
            self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file.
            self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
            self.Patch_Array = np.array(self.Patch_DF)

            while self.check_Patch_NotLungTissue(): # Make sure the patch is not from outside the body.
                self.x_Patch_TopLeft = random.randint(self.MinX_CancerousPatch, self.MaxX_CancerousPatch) # Define a random location for a Cancerous patch without going outside the dimensions of the CT image.
                self.y_Patch_TopLeft = random.randint(self.MinY_CancerousPatch, self.MaxY_CancerousPatch)

                if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("In extract_APatch_Cancerous(self) ... while self.check_Patch_NotLungTissue(): ...")
                    # print("self.x_CentrePixel_BoundingBox =", self.x_CentrePixel_BoundingBox)
                    # print("self.y_CentrePixel_BoundingBox =", self.y_CentrePixel_BoundingBox)
                    # print()
                    # print("self.MinX_CancerousPatch =", self.MinX_CancerousPatch)
                    # print("self.MaxX_CancerousPatch =", self.MaxX_CancerousPatch)
                    # print()
                    # print("self.MinY_CancerousPatch =", self.MinY_CancerousPatch)
                    # print("self.MaxY_CancerousPatch =", self.MaxY_CancerousPatch)
                    # print()
                    # print("self.x_Patch_TopLeft =", self.x_Patch_TopLeft)
                    # print("self.y_Patch_TopLeft =", self.y_Patch_TopLeft)
                    # print()
                    ####
                
                # Extract the patch from the specified DICOM file/image.
                self.GoodQualityDICOMFile_Array, self.frame_num, self.width, self.height, self.ch = utils.loadFile(self.PathToGoodQualityDICOMFile) # Express the CT image as a numpy array. I am interested only in the array. The array has shape (ch, rows (y), columns (x)).
                
                # Extract a patch.
                self.GoodQualityDICOMFile_DF = pd.DataFrame(self.GoodQualityDICOMFile_Array[0]) # It is possible to extract a patch from a pandas DataFrame. However, I am assuming that the image is in the first channel of the file.
                self.Patch_DF = self.GoodQualityDICOMFile_DF.iloc[self.y_Patch_TopLeft:(self.y_Patch_TopLeft + PATCH_HEIGHT), self.x_Patch_TopLeft:(self.x_Patch_TopLeft + PATCH_WIDTH)] # Here is an extracted patch as a pandas DataFrame.
                self.Patch_Array = np.array(self.Patch_DF)

            # Now it is time to sort the patch.

            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                # print("self.GoodQualityDICOMFile_DF =", self.GoodQualityDICOMFile_DF)
                # print()
                # print("self.x_Patch_TopLeft =", self.x_Patch_TopLeft)
                # print("self.y_Patch_TopLeft =", self.y_Patch_TopLeft)
                # print()
                # print("self.Patch_DF =", self.Patch_DF)
                # print()
                # print("self.Patch_Array =", self.Patch_Array)
                # print()
                ####
        
                
    def check_Patch_NotLungTissue(self): # This method checks the present value of self.Patch_Array whenever it is called.
        # Q) How will Python know if an extracted patch is neither of noncancerous or cancerous lung tissue?
            # Where < 80% of the patch have < -920 HU (or other value) in pixel value. Any patch with > 80% of its pixels having < -920 HU (or other value) pixel value must be discarded
        
        ## Count the number of pixels. Determine how many pixels have < -920 HU (or other value) in pixel value. If those pixels amount to < 80% of the total number of pixels in the patch, the patch is Noncancerous. If > 80%, discard the patch.
        self.Patch_List_Row = list(self.Patch_Array.reshape(NUMBER_OF_PIXELS))
        self.Index_List = list(range(0, NUMBER_OF_PIXELS)) # ... for making a dictionary. A dictionary is faster to iterate through in a FOR loop compared to a list.
        self.Patch_Dict_Row = dict(zip(self.Index_List, self.Patch_List_Row))

        self.Counter_Pixels_ValueLessThan_ThresholdPixelValue = 0
        self.Counter_Pixels_ValueGreaterThanOrEqual_ThresholdPixelValue = 0

        
        # Some of the original images are circular. The pixels outside of their circular border appear to have a pixel value of -3024 HU, around 2000 HU below the apparent minimum pixel value of the actual image. Therefore, patches that have those pixels will be discarded lest they negatively affect the performance of the patch classifier.
        self.MinimumPixelValue_Threshold = -3000
        self.MinimumPixelValueOfPatchArray = self.Patch_Array.min()
        
        if self.MinimumPixelValueOfPatchArray <= self.MinimumPixelValue_Threshold: # < -3000 HU, in case the pixels outside of the circular border have a range of values.
            print("A patch that had a minimum pixel value of", self.MinimumPixelValueOfPatchArray, "HU was discarded.")
            return True # "True" means that the patch is *not* of lung tissue. In this case, it more accurately means that the patch should be discarded.
            # print("WARNING: A patch that had a minimum pixel value of", self.MinimumPixelValueOfPatchArray, "HU was *not* discarded.")
            # return False

        
        # The section below was commented out b/c fiducial markers can happen in the real world. It is better to include them as training data, lest I miss out on valuable training data.
        # self.MaximumPixelValue_Threshold = 4000
        # self.MaximumPixelValueOfPatchArray = self.Patch_Array.max()

        # if self.MaximumPixelValueOfPatchArray > self.MaximumPixelValue_Threshold: # Some CT images have fiducial markers in them & they appear very bright in the images. The markers are usually located outside the patients' bodies & therefore indicate that the patch was not extracted from lung tissue. Also, the fiducial markers are not naturally part of the human body.
        #     print("A patch that had a maximum pixel value of", self.MaximumPixelValueOfPatchArray, "HU was discarded.")
        #     return True
        
        
        # Check if the patch is of the area outside of the body.
        self.ThresholdPixelValue = -920 # Units: HU.
        
        for index in range(0, NUMBER_OF_PIXELS): # Count the number of pixels with values less than -920 HU & those with values greater than or equal to -920 HU.
            if self.Patch_Dict_Row[index] < self.ThresholdPixelValue:
                self.Counter_Pixels_ValueLessThan_ThresholdPixelValue = self.Counter_Pixels_ValueLessThan_ThresholdPixelValue + 1
            
            elif self.Patch_Dict_Row[index] >= self.ThresholdPixelValue:
                self.Counter_Pixels_ValueGreaterThanOrEqual_ThresholdPixelValue = self.Counter_Pixels_ValueGreaterThanOrEqual_ThresholdPixelValue + 1
        
        self.ProportionOfPixels_ValueLessThan_ThresholdPixelValue = self.Counter_Pixels_ValueLessThan_ThresholdPixelValue / NUMBER_OF_PIXELS
        
        
        self.ProportionOfPixels_Threshold = 0.80
        if (self.ProportionOfPixels_ValueLessThan_ThresholdPixelValue >= self.ProportionOfPixels_Threshold): # ... i.e., the patch is *not* of lung tissue.
            print("A patch that had at least", (self.ProportionOfPixels_Threshold * 100), "% of its pixels with a value less than", self.ThresholdPixelValue, "HU was discarded.")
            return True
        else:
            return False
    
    
    def discard_Patch_NotLungTissue(self):
        # self.NumberOfPatchesExtracted = self.NumberOfPatchesExtracted - 1 # Undo the addition to this counter when a NotLungTissue patch was extracted.
        pass
    
    
    def save_PatchToFile(self, DictionaryOfSelectedPatients, Class, RandomPatient, RandomSOPInstanceUID, SubsetToFocusOnForPatchExtraction_Directory, x_Patch_TopLeft, y_Patch_TopLeft):
        self.NameOfGoodQualityDICOMFile = DictionaryOfSelectedPatients[RandomPatient][0][RandomSOPInstanceUID][1].replace(".dcm", "") # E.g., = "1-18".
        # self.SubsetForPatch = self.DictionaryOfSelectedPatients[self.RandomPatient][1] # The patch shall be saved in the subset that was allocated for the patient.
        if GROUP_BY_PATIENT_ANSWER == False:
            Image.fromarray(self.Patch_Array).save(SubsetToFocusOnForPatchExtraction_Directory + Class + RandomPatient + "_" + self.NameOfGoodQualityDICOMFile + "_" + str(x_Patch_TopLeft) + "_" + str(y_Patch_TopLeft) + ".tif") # Save the patch, which is in numpy array form, as a TIF image.
        
        if GROUP_BY_PATIENT_ANSWER == True: # ... for when I am trying to mimic k-fold cross-validation.
            if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("Saving a patch by patient name ...")
                    ####
            
            try:
                Image.fromarray(self.Patch_Array).save(SubsetToFocusOnForPatchExtraction_Directory + Class + RandomPatient + "/" + RandomPatient + "_" + self.NameOfGoodQualityDICOMFile + "_" + str(x_Patch_TopLeft) + "_" + str(y_Patch_TopLeft) + ".tif") # Save the patch, which is in numpy array form, as a TIF image. Put the patch into a folder in the subset that has the patient's name as its name.
            
            except FileNotFoundError:
                os.mkdir(SubsetToFocusOnForPatchExtraction_Directory + Class + RandomPatient + "/") # The patient directory must exist for Image.fromarray() to successfully save the patch in it.
                Image.fromarray(self.Patch_Array).save(SubsetToFocusOnForPatchExtraction_Directory + Class + RandomPatient + "/" + RandomPatient + "_" + self.NameOfGoodQualityDICOMFile + "_" + str(x_Patch_TopLeft) + "_" + str(y_Patch_TopLeft) + ".tif") # Save the patch, which is in numpy array form, as a TIF image. Put the patch into a folder in the subset that has the patient's name as its name.
        
        self.NumberOfPatchesExtracted = self.NumberOfPatchesExtracted + 1 # Increase the counter of extracted patches by 1.
        
        return SubsetToFocusOnForPatchExtraction_Directory
    

    def sort_PatchToClass(self):
        if DEBUG_MODE_ANSWER == True:
            pass
            #### debug
            print("Sorting the extracted patch to its appropriate class ...")
            ####
        
        ### Sort the patch.
        # Apply the bounding box to the good-quality DICOM file.
        self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox = self.get_BoundingBoxCoordinates(self.Annotations_SOPInstanceUID_List, self.Annotations, self.RandomSOPInstanceUID) # Get the bounding box coordinates.
        
        ## Calculate the criteria by which the patch will be sorted.
        self.x_CentrePixel_BoundingBox, self.y_CentrePixel_BoundingBox = self.calculate_CentreCoordinatesOfBoundingBox(self.xMin_BoundingBox, self.yMin_BoundingBox, self.xMax_BoundingBox, self.yMax_BoundingBox) # Determine the location/coordinates of the centre pixel of the bounding box.
        self.add_MarginToBoundingBox()
        
        self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates() # Determine the corner coordinates of the patch.
        # self.xMin_Patch, self.yMin_Patch, self.xMax_Patch, self.yMax_Patch = self.calculate_PatchCornersCoordinates(x_Patch_TopLeft = self.x_Patch_TopLeft, y_Patch_TopLeft = self.y_Patch_TopLeft)

        # Check if the patch is Cancerous.
        if (self.x_CentrePixel_BoundingBox in range(self.xMin_Patch, self.xMax_Patch + 1)) and (self.y_CentrePixel_BoundingBox in range(self.yMin_Patch, self.yMax_Patch + 1)): # If the patch is Cancerous ...
            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                print("The patch was found to be Cancerous.")
                ####
            
            self.SubsetToSaveNextPatchIn_Directory = self.save_PatchToFile(self.DictionariesOfSelectedPatients_Dict[self.SubsetToFocusOnForPatchExtraction_Directory], "Cancerous/", self.RandomPatient, self.RandomSOPInstanceUID, self.SubsetToFocusOnForPatchExtraction_Directory, self.x_Patch_TopLeft, self.y_Patch_TopLeft) # Save the patch in the Cancerous folder of the chosen subset.
            
            # Now extract a Noncancerous patch & put it into the same subset as the Cancerous patch that was just extracted.
            # self.choose_CTSlice_SameSubset(self.SubsetToSaveNextPatchIn) # Choose a CT slice from any patient allocated to the *same subset*.
            self.choose_CTSlice(SubsetToFocusOnForPatchExtraction = self.SubsetToSaveNextPatchIn_Directory) # Choose a CT slice from any patient allocated to the *same subset*.
            self.extract_APatch_Noncancerous()

            # self.save_PatchToFile_SameSubset(self.SubsetToSaveNextPatchIn_Directory, self.DictionaryOfSelectedPatients_Training, "Noncancerous/", self.RandomPatient_Training, self.RandomSOPInstanceUID_Training)
            self.SubsetToSaveNextPatchIn = self.save_PatchToFile(self.DictionariesOfSelectedPatients_Dict[self.SubsetToSaveNextPatchIn_Directory], "Noncancerous/", self.RandomPatient, self.RandomSOPInstanceUID, self.SubsetToSaveNextPatchIn_Directory, self.x_Patch_TopLeft, self.y_Patch_TopLeft) # Save the patch in the Cancerous folder of the chosen subset.
            self.SubsetToSaveNextPatchIn = None # Prepare to extract the next patch from a patient in a random subset. I.e., reset this value.

            
        # Check if the patch is Noncancerous.
        elif not (self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMin_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMax_Patch, self.yMax_Patch) or self.check_Patch_CornerInExpandedBoundingBox(self.xMin_Patch, self.yMax_Patch)):
            if DEBUG_MODE_ANSWER == True:
                pass
                #### debug
                print("The patch was found to possibly be Noncancerous.")
                ####
            
            if self.check_Patch_NotLungTissue(): # ... if the patch is not lung tissue in the first place. Extract another patch in its place.
                if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("Entering IF self.check_Patch_NotLungTissue(): ...")
                    ####                
                
                self.discard_Patch_NotLungTissue()
            
            else:
                if DEBUG_MODE_ANSWER == True:
                    pass
                    #### debug
                    print("Entering ELSE of IF self.check_Patch_NotLungTissue(): ...")
                    ####   

                self.SubsetToSaveNextPatchIn_Directory = self.save_PatchToFile(self.DictionariesOfSelectedPatients_Dict[self.SubsetToFocusOnForPatchExtraction_Directory], "Noncancerous/", self.RandomPatient, self.RandomSOPInstanceUID, self.SubsetToFocusOnForPatchExtraction_Directory, self.x_Patch_TopLeft, self.y_Patch_TopLeft) # Save the patch in the Noncancerous folder of the chosen subset.
                
                # Now extract a Cancerous patch & put it into the same subset as the Noncancerous patch.
                self.choose_CTSlice(SubsetToFocusOnForPatchExtraction = self.SubsetToSaveNextPatchIn_Directory) # Choose a CT slice from any patient allocated to the *same subset*.
                self.extract_APatch_Cancerous()

                # self.save_PatchToFile_SameSubset(self.SubsetToSaveNextPatchIn_Directory, self.DictionaryOfSelectedPatients_Training, "Cancerous/", self.RandomPatient_Training, self.RandomSOPInstanceUID_Training)
                self.SubsetToSaveNextPatchIn = self.save_PatchToFile(self.DictionariesOfSelectedPatients_Dict[self.SubsetToSaveNextPatchIn_Directory], "Cancerous/", self.RandomPatient, self.RandomSOPInstanceUID, self.SubsetToSaveNextPatchIn_Directory, self.x_Patch_TopLeft, self.y_Patch_TopLeft) # Save the patch in the Cancerous folder of the chosen subset.
                self.SubsetToSaveNextPatchIn = None # Prepare to extract the next patch from a patient in a random subset. I.e., reset this value.

    
    # def extract_Patches(self):
    #     try:
    #         assert self.NumberOfPatchesExtracted == self.NumberOfPatchesToExtract
        
    #     except AssertionError:
    #         while self.NumberOfPatchesExtracted < self.NumberOfPatchesToExtract:
    #             self.choose_CTSlice()
    #             self.extract_APatch()
    #             self.sort_PatchToClass()

    #             if (self.NumberOfPatchesExtracted % 50 == 0): # Display the patch extraction progress periodically. NOTE: Patches are extracted 2 by 2, 1 per class.
    #                 print(self.NumberOfPatchesExtracted, "patches have been extracted ...")
            
    #         try: # Make sure that the required number of patches is extracted. It happened once that only 9,998 patches were extracted when I asked for 10,000.
    #             assert self.NumberOfPatchesExtracted == self.NumberOfPatchesToExtract
            
    #         except AssertionError:
    #             while self.NumberOfPatchesExtracted < self.NumberOfPatchesToExtract:
    #                 self.choose_CTSlice()
    #                 self.extract_APatch()
    #                 self.sort_PatchToClass()

    #                 if (self.NumberOfPatchesExtracted % 50 == 0): # Display the patch extraction progress periodically. NOTE: Patches are extracted 2 by 2, 1 per class.
    #                     print(self.NumberOfPatchesExtracted, "patches have been extracted ...")
    
    def extract_Patches(self):
        while self.NumberOfPatchesExtracted < self.NumberOfPatchesToExtract:
            self.choose_CTSlice()
            self.extract_APatch()
            self.sort_PatchToClass()

            if (self.NumberOfPatchesExtracted % 50 == 0): # Display the patch extraction progress periodically. NOTE: Patches are extracted 2 by 2, 1 per class.
                print()
                print(self.NumberOfPatchesExtracted, "patches have been extracted ...")
                print()
    

    # def extract_ReplacementPatch(self):
    #     pass
    
    
    # def deleteAndReplace_DuplicatePatches(self): # Each patch has a set of coordinates for its top-left corner. Compare those coordinates with those of the patches which have already been extracted to identify duplicates. Doing this is simpler & probably cheaper in computer resources than comparing the patches themselves.
    #     pass

    #     self.extract_ReplacementPatch()
    

    def checkThatTheNumberOfExtractedPatchesIsAsRequired(self): # Check that the number of patches that were extracted & saved into a subset is truly equal to the number of patches I asked for.
        self.SubsetDirectories_List = [DIRECTORY_PATCHES_TRAINING, DIRECTORY_PATCHES_DEVELOPMENT, DIRECTORY_PATCHES_HOLDOUT]
        self.Patches_Directories_List = []

        for subset_directory in self.SubsetDirectories_List:
            for class_folder in os.listdir(subset_directory):
                self.ClassDirectory = os.path.join(subset_directory, class_folder)

                if GROUP_BY_PATIENT_ANSWER == True:
                    for patient_folder in os.listdir(self.ClassDirectory):
                        self.PatientDirectory_InSubsetInClass = os.path.join(self.ClassDirectory, patient_folder)

                        for patch_file in os.listdir(self.PatientDirectory_InSubsetInClass):
                            self.Patches_Directories_List.append(os.path.join(self.PatientDirectory_InSubsetInClass, patch_file).replace("\\", "/"))
                
                elif GROUP_BY_PATIENT_ANSWER == False:
                    for patch_file in os.listdir(self.ClassDirectory):
                        self.Patches_Directories_List.append(os.path.join(self.ClassDirectory, patch_file).replace("\\", "/"))
        
        self.NumberOfPatchesFound = len(self.Patches_Directories_List)
        self.NumberOfPatchesExtracted = self.NumberOfPatchesFound # Correct the number of patches that were extracted to the number of patches that were actually found.

        if self.NumberOfPatchesFound == self.NumberOfPatchesToExtract:
            return True
        
        elif self.NumberOfPatchesFound < self.NumberOfPatchesToExtract: # Extract more patches.
            return False

        elif self.NumberOfPatchesFound > self.NumberOfPatchesToExtract:
            print("WARNING: {} additional patch(es) was/were extracted.".format(self.NumberOfPatchesFound - self.NumberOfPatchesToExtract))
            
            return True
    
    
    def export_appendTo_PreparationData(self): # Use the variables calculated after the patch extraction to add to the record of the details of the patch extraction session.
        self.InfoFile = open(DIRECTORY_PATCHES + DATA_FILE_NAME , "a")
        self.InfoFile_Contents_NumberOfPatchesExtracted = "Number of patches extracted = {}\n".format(self.NumberOfPatchesExtracted)
        self.InfoFile_Contents_NumberOfXMLFilesSelectedPatientsHave = "\nNumber of XML annotation files that the selected patients have = {}\n".format(self.NumberOfXMLFilesOfAllSelectedPatients)
        self.InfoFile_Contents_Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom = "Number of good-quality DICOM files that patches were or could have been extracted from = {}\n".format(self.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom)

        self.InfoFile_AdditionalInfo = self.InfoFile_Contents_NumberOfPatchesExtracted + self.InfoFile_Contents_NumberOfXMLFilesSelectedPatientsHave + self.InfoFile_Contents_Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom

        self.InfoFile.write(self.InfoFile_AdditionalInfo)
        self.InfoFile.close()
    

def main():
    preparer = Preparer(NUMBER_OF_PATIENTS_TO_SELECT, NUMBER_OF_PATCHES_TO_EXTRACT)
    preparer.initialise_Databases()
    preparer.specify_DirectoriesOfGoodQualityCTSlices()
    preparer.make_DatabaseOfSelectedPatients()
    preparer.separate_PatientsBySubset()
    preparer.make_DictionariesOfSelectedPatients()
    preparer.merge_DictionariesOfSelectedPatients()
    preparer.merge_DatabasesOfSelectedPatients()
    preparer.make_AnnotationDirectoryListsDictionary()
    preparer.count_NumberOfXMLFilesTheSelectedPatientsHave()
    preparer.count_NumberOfGoodQualityCTImagesPatchesCouldBeExtractedFrom()

    # AnnotationDirectoryLists_Dict, DictionariesOfSelectedPatients_Dict, DirectoriesOfGoodQualityCTSlices, DatabasesOfSelectedPatients_Dict = preparer.transfer_Objects()
    preparer.export_PreparationData()

    patch_extractor = PatchExtractor(NUMBER_OF_PATIENTS_TO_SELECT, NUMBER_OF_PATCHES_TO_EXTRACT, preparer.AnnotationDirectoryLists_Dict, preparer.DictionariesOfSelectedPatients_Dict, preparer.DirectoriesOfGoodQualityCTSlices, preparer.DatabasesOfSelectedPatients_Dict, preparer.NumberOfXMLFilesOfAllSelectedPatients, preparer.Counter_NumberOfGoodQualityDICOMFilesPatchesCanBeExtractedFrom) # Prepare to use the methods of the PatchExtractor class.
    
    while not patch_extractor.checkThatTheNumberOfExtractedPatchesIsAsRequired(): # Stop extracting patches once the number of patches I asked for have been extracted.
        patch_extractor.extract_Patches()
    
    patch_extractor.export_appendTo_PreparationData()

    print("The patch extraction algorithm has finished extracting {} patches! The patches can be found in the 'Patches/' folder unless an unexpected error has occurred.".format(patch_extractor.NumberOfPatchesExtracted)) # Show a message indicating that the algorithm has finished running.

    
if __name__ == "__main__": # Having this IF statement is good practice for making sure this script file is run only when it is called directly.
    main()
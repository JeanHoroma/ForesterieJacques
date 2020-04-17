# from see import see
import georaster as gr
import sys
import numpy as np
import tkinter
import gdal
from tkinter import filedialog


def lecture_inter(fich):
    with open(fich, 'r') as f:
        # lecture du MNS brut
        larg = np.asscalar(np.fromfile(file=f, dtype=np.int16, count=1).astype(int))
        haut = np.asscalar(np.fromfile(file=f, dtype=np.int16, count=1).astype(int))
        interp = np.fromfile(file=f, dtype=np.int16, count=larg * haut).reshape((haut, larg))
        return interp, larg, haut


def main(nom):
    #  lecture de l'interpretation
    if nom == "":
        tkinter.Tk().withdraw()
        nom = filedialog.askopenfilename(
            title="Fichier contenant l'interpretation ?")  # ,filetypes=(("fichiers shape","*.shp")))
    interp, larg, haut = lecture_inter(nom)
    # lecture de l'image tif georeferencee
    nom_se = nom[:nom.rfind(".")]
    nom_image = nom_se + ".tif"
    # creation de l'image interpretee georeferencee
    info_geo = gr.MultiBandRaster(nom_image, load_data=False)
    im_gen = gr.SingleBandRaster.from_array(interp, info_geo.trans, info_geo.proj.srs, gdal.GDT_Int16)
    im_gen.save_geotiff(nom_se + "_int.tif", dtype=gdal.GDT_Int16)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("")

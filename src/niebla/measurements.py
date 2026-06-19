try:
    from src.niebla.measurements_folder.sfr import sfr
    from src.niebla.measurements_folder.metallicity import metallicity
    from src.niebla.measurements_folder.emissivity import emissivity
    from src.niebla.measurements_folder.ebl import ebl
except ModuleNotFoundError:
    from niebla.measurements_folder.sfr import sfr
    from niebla.measurements_folder.metallicity import metallicity
    from niebla.measurements_folder.emissivity import emissivity
    from niebla.measurements_folder.ebl import ebl

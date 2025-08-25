# IMPORTS -----------------------------------#
import os
import time
import copy
import logging
import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, \
    RegularGridInterpolator

from astropy.io import fits
from astropy import units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM

from src.niebla.safe_log10 import log10_safe
from src.niebla.sfr_models import sfr_model
from src.niebla.metall_models import metall_model
from src.niebla.dust_absorption_models import dust_abs_fraction
from src.niebla.dust_reemission_models import dust_reem

try:
    from hmf import MassFunction
except ModuleNotFoundError:
    print('Module hmf not loaded')

data_path = os.path.join(os.path.split(__file__)[0], 'data/')


class EBL_model(object):
    """
    Class that computes the EBL contribution coming from three sources:
    Single Stellar Populations (SSP), Intra-Halo Light (IHL)
    and axion decay.

    Units of returns
    -----------------
    EBL:
      -> Cubes: nW m**-2 sr**-1
      -> Splines: log10(nW m**-2 sr**-1)
    Emissivity spline: log10(erg s^-1 Hz^-1 Mpc^-3)
    """

    @staticmethod
    def input_yaml_data_into_class(yaml_data, log_prints=False):
        """
        Function to initialize the EBL class with a yaml file or
        dictionary.

        :param yaml_data: dictionary
            It has the necessary data to initialize the EBL class.
        :param log_prints: Bool
            Boolean to decide whether logging prints are enabled.
        :return: class
            The EBL calculation class.
        """
        print(yaml_data['redshift_array'])
        z_array = np.geomspace(
            1e-6,
            yaml_data['redshift_array']['z_max'],
            yaml_data['redshift_array']['z_steps'])
        wv_array = np.geomspace(
            float(yaml_data['wavelength_array']['wv_min']),
            float(yaml_data['wavelength_array']['wv_max']),
            yaml_data['wavelength_array']['wv_steps'])
        return EBL_model(
            z_array, wv_array,
            h=float(yaml_data['cosmology_params']['cosmo'][0]),
            omegaM=float(
                yaml_data['cosmology_params']['cosmo'][1]),
            omegaBar=float(
                yaml_data['cosmology_params']['omegaBar']),
            t_intsteps=yaml_data['t_intsteps'],
            z_max=yaml_data['z_intmax'],
            log_prints=log_prints)

    def logging_info(self, text):
        if self._log_prints:
            logging.info(
                '%.2fs: %s' % (time.process_time()
                               - self._process_time, text))
            self._process_time = time.process_time()

    def __init__(self, z_array, lambda_array,
                 h=0.7, omegaM=0.3, omegaBar=0.0222 / 0.7 ** 2.,
                 log_prints=False,
                 t_intsteps=201,
                 z_max=35
                 ):
        """
        Initialize the source class.

        Parameters
        ----------
        z_array: array-like
            Array of redshifts at which to calculate the EBL.
        lambda_array: array-like [microns]
            Array of wavelengths at which to calculate the EBL.
        h: float
            Little H0, or h == H0 / 100 km/s/Mpc
        omegaM: float
            Fraction of matter in our current Universe.
        omegaBar: float
            Fraction of baryons in our current Universe.
        log_prints: Bool
            Whether to print the loggins of our procedures or not.
        t_intsteps: int
            Number of integration steps to compute with
            (uses Simpson's integration method)
        z_max: float
            Maximum redshift at which we form SSPs.
        """

        self._shifted_zz_emiss = None
        self._log_t_ssp_intcube = None
        self._process_time = time.process_time()
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        self._log_prints = log_prints

        self._cube = (len(lambda_array), len(z_array), t_intsteps)
        self._steps_integration_cube = np.linspace(0., 1., t_intsteps)

        self._ssp_log_time_init = None
        self._ssp_metall = None
        self._ssp_lumin_spline = None

        self._emiss_ihl_cube = 0.

        self._emiss_ssp_spline = None
        self._emiss_ihl_spline = None

        self._ebl_ssp_spline = None
        self._ebl_ihl_spline = None
        self._ebl_axion_spline = None
        self._ebl_total_spline = None

        self._h = h
        self._omegaM = omegaM
        self._omegaB0 = omegaBar
        self._cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                                    Ob0=omegaBar, Tcmb0=2.7255)

        self._z_max = z_max
        self._lambda_array = lambda_array[::-1]
        self._freq_array = log10_safe(
            c.c.value / lambda_array[::-1] * 1e6)
        self._freq_array = np.expand_dims(self._freq_array, axis=(1, 2))
        self._t_intsteps = t_intsteps

        tt = log10_safe(
            self._cosmo.lookback_time(z_array).to(u.yr).value)

        tt = np.insert(tt, 0, 0)
        z_array = np.insert(z_array, 0, 0)

        self._z_array = np.expand_dims(z_array, axis=(0, 2))

        self._t2z = UnivariateSpline(
            tt, log10_safe(self._z_array),
            s=0, k=1)

        self._kernel_emiss = None
        self._last_yaml_emiss = None
        self._last_ssp_log_time_init = None
        self._ebl_intcube = None
        self._shifted_freq = None
        self._ebl_z_intcube = None
        self._mean_metall_cube_log = None

        self.logging_info('Initialize class: end')
        return

    def emiss_ssp_spline(self, wv_array, zz_array):
        ff_array = np.log10(c.c.value / wv_array * 1e6)
        return 10 ** self._emiss_ssp_spline((ff_array, zz_array))

    def ebl_total_spline(self, wv_array, zz_array):
        ff_array = np.log10(c.c.value / wv_array * 1e6)
        return 10 ** self._ebl_total_spline(ff_array, zz_array, grid=False)

    def ebl_ssp_spline(self, wv_array, zz_array):
        ff_array = np.log10(c.c.value / wv_array * 1e6)
        return 10 ** self._ebl_ssp_spline(ff_array, zz_array, grid=False)

    def ebl_ihl_spline(self, wv_array, zz_array):
        ff_array = np.log10(c.c.value / wv_array * 1e6)
        return 10 ** self._ebl_ihl_spline(ff_array, zz_array, grid=False)

    def ssp_lumin_spline(self, wv_array=None,
                         freq_array=None,
                         age_array=6.,
                         metall_array=0.02):
        if wv_array is None and freq_array is None:
            print('No values given for either wavelength or frequency.'
                  ' Choose one!')
            exit()

        if wv_array is not None and freq_array is not None:
            print('Given both wavelength and frequency arrays.'
                  ' Choose only one!')
            exit()

        if wv_array is not None:
            freq_array = np.log10(c.c.value / wv_array * 1e10)
        return self._ssp_lumin_spline(
            xi=(freq_array, age_array, metall_array))

    @property
    def logging_prints(self):
        return self._log_prints

    @logging_prints.setter
    def logging_prints(self, new_print):
        self._log_prints = new_print
        return

    def t2z(self, tt):
        return 10 ** self._t2z(tt)

    def read_SSP_file(self, yaml_file):
        """
        Read Simple Stellar Population model spectra.

        Spectra units:
        --------------
        log10(erg s^-1 A^-1).

        Starburst99 output:
        http://www.stsci.edu/science/starburst99/
        The calculations assume that the value of 'Total stellar mass'
        in the simulation has been left as default, 1e6 Msun. If not,
        change the calculation of emissivity of '-6.' to
        log10(new total mass).

        """
        if yaml_file['ssp_type'] == 'example_SB99':
            path_ssp = (
                    data_path
                    + 'ssp_synthetic_spectra/starburst99'
                    + '/kroupa_padova/')

            yaml_file['ssp_type'] = 'SB99'
            yaml_file['file_name'] = '/kroupa_'
            yaml_file['ignore_rows'] = 6
            yaml_file['total_stellar_mass'] = 6.

        elif yaml_file['ssp_type'] == 'example_pegase3':
            path_ssp = (
                    data_path
                    + 'ssp_synthetic_spectra/pegase3'
                    + '/kroupa_padova/')

            yaml_file['ssp_type'] = 'generic'
            yaml_file['file_name'] = '/kroupa_'
            yaml_file['ignore_rows'] = 6
            yaml_file['total_stellar_mass'] = 6.
            yaml_file['L_lambda'] = True

        elif yaml_file['ssp_type'] == 'stripped_Goetberg19':
            path_ssp = (
                    data_path
                    + 'ssp_synthetic_spectra/starburst99'
                    + '/stripped_Goetberg19/')

            yaml_file['ssp_type'] = 'generic'
            yaml_file['ignore_rows'] = 0
            yaml_file['total_stellar_mass'] = 6.
            yaml_file['L_lambda'] = True

        else:
            path_ssp = data_path + yaml_file['path_ssp']

        if yaml_file['ssp_type'] == 'SB99':

            ssp_metall = np.sort(
                np.array(os.listdir(path_ssp), dtype=float))

            # Open one of the files and check for time steps and frequencies
            d = np.loadtxt(
                path_ssp + str(ssp_metall[0])
                + yaml_file['file_name']
                + str(ssp_metall[0]).replace('0.', '')
                + '.spectrum1',
                skiprows=yaml_file['ignore_rows'])

            # Get unique time steps, frequencies and metallicities,
            # and their respective spectral data
            t_total = np.unique(d[:, 0])
            l_total = np.unique(d[:, 1])

            dd_total = np.zeros((l_total.shape[0],
                                 t_total.shape[0],
                                 len(ssp_metall) + 2))

            for n_met, met in enumerate(ssp_metall):
                data = np.loadtxt(
                    path_ssp + str(met) +
                    yaml_file['file_name'] + str(met).replace('0.', '')
                    + '.spectrum1',
                    skiprows=yaml_file['ignore_rows'])

                dd_total[:, :, n_met + 1] = data[:, 2].reshape(
                    t_total.shape[0], l_total.shape[0]).T

            # Extend the stellar spectra to very low and high metallicities
            ssp_metall = np.insert(ssp_metall, 0, 1e-43)
            dd_total[:, :, 0] = dd_total[:, :, 1]

            ssp_metall = np.append(ssp_metall, 1.)
            dd_total[:, :, -1] = dd_total[:, :, -2]

            # Define the quantities we work with
            ssp_log_time = log10_safe(t_total)  # log(time/yrs)
            ssp_log_freq = log10_safe(  # log(frequency/Hz)
                c.c.value / l_total / 1e-10)
            ssp_log_emis = (
                    dd_total  # log(L_nu[erg/s/Hz/M_solar])
                    - float(yaml_file['total_stellar_mass'])
                    + np.log10(1e10 * c.c.value)
                    - 2. * ssp_log_freq[:, np.newaxis, np.newaxis])


        elif yaml_file['ssp_type'] == 'generic':
            """
            Generic datafile(s) for inputing luminosity data, dimensions
            (n+1) x (m+1).
            Each of the datafiles has to have the name of the metallicity
            of each dataset.

            In each file:
            - The [0,0] entry is empty.
            - The zeroth column [1:, 0] contains the wavelengths [Angstroms].
            - The first row [0, 1:] contains the time/age of the SSP [years].
            - The remaining values (n x m) are the luminosity L_lambda
                [erg/sec/A/Msun] or L_nu [erg/sec/Hz/Msun].

            This method assumes that all files share the wavelength and
            age array.
            """
            ssp_metall = np.sort(np.array(
                os.listdir(path_ssp), dtype=float))

            # Open one of the files and check for time steps and frequencies
            data_generic = np.loadtxt(
                path_ssp + str(ssp_metall[0]),
                skiprows=yaml_file['ignore_rows'])

            # Get unique time steps and frequencies
            t_total = data_generic[0, 1:]
            l_total = data_generic[1:, 0]

            # Load all the luminosity data for different metallicities
            ssp_log_emis = np.zeros(
                (l_total.shape[0], t_total.shape[0],
                 len(ssp_metall) + 2))

            for n_met, met in enumerate(ssp_metall):
                data = np.loadtxt(
                    path_ssp + str(met),
                    skiprows=yaml_file['ignore_rows'])
                ssp_log_emis[:, :, n_met + 1] = data[1:, 1:]

            # Extend the stellar spectra to very low metallicities
            ssp_metall = np.insert(ssp_metall, 0, 1e-43)
            ssp_log_emis[:, :, 0] = ssp_log_emis[:, :, 1]

            # Extend the stellar spectra to very high metallicities
            ssp_metall = np.append(ssp_metall, 1.)
            ssp_log_emis[:, :, -1] = ssp_log_emis[:, :, -2]

            # Calculate the logarithms of all the values
            ssp_log_time = log10_safe(t_total)

            ssp_log_freq = log10_safe(  # log(frequency/Hz)
                c.c.value / l_total / 1e-10)

            ssp_log_emis = (log10_safe(ssp_log_emis)
                            - float(yaml_file['total_stellar_mass']))

            if yaml_file['L_lambda']:
                ssp_log_emis += (np.log10(1e10 * c.c.value)
                                 - 2. * ssp_log_freq
                                 [:, np.newaxis, np.newaxis])

        else:
            ValueError('SSP type not recognized')

        ssp_log_emis[np.isnan(ssp_log_emis)] = -43.
        ssp_log_emis[
            np.invert(np.isfinite(ssp_log_emis))] = -43.

        self._ssp_lumin_spline = RegularGridInterpolator(
            points=(ssp_log_freq,
                    ssp_log_time,
                    log10_safe(ssp_metall)),
            values=ssp_log_emis,
            method='linear',
            bounds_error=False, fill_value=-43
        )

        self._ssp_metall = ssp_metall
        self._ssp_log_time_init = ssp_log_time[0]
        del ssp_log_freq, ssp_log_time, ssp_metall, ssp_log_emis
        self.logging_info('Reading of SSP file')
        return

    def emiss_ssp_calculation(self, yaml_data):
        """
        Calculation of SSP emissivity from the parameters given in
        the dictionary. If no sfr is given as an inout, the sfr data
        specified in the yaml file will be used to calculate the emissivities.

        Emissivity units:
        ----------
            -> Splines: log10(erg / s / Hz / Mpc**3 == 1e-7 W / Hz / Mpc**3)

        Parameters
        ----------
        yaml_data: dict
            Data necessary to reconstruct the EBL component from a SSP.
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        """
        self.logging_info('SSP parameters: %s' % yaml_data['name'])

        if (self._last_yaml_emiss is None
                or (yaml_data['ssp'] != self._last_yaml_emiss['ssp'])):
            self.read_SSP_file(yaml_data['ssp'])

        if (self._ssp_log_time_init != self._last_ssp_log_time_init
                or self._log_t_ssp_intcube is None):
            lookback_time_cube = self._cosmo.lookback_time(
                self._z_array).to(u.yr)

            # Array of time values we integrate the emissivity over
            # (in log10)
            self._log_t_ssp_intcube = np.log10(
                (self._cosmo.lookback_time(self._z_max)
                 - lookback_time_cube).to(u.yr).value)

            self._log_t_ssp_intcube = (
                    (self._log_t_ssp_intcube - self._ssp_log_time_init)
                    * self._steps_integration_cube
                    + self._ssp_log_time_init
            )

            self._shifted_zz_emiss = self.t2z(np.log10(
                lookback_time_cube.value
                + 10. ** self._log_t_ssp_intcube))

            self._mean_metall_cube_log = log10_safe(metall_model(
                zz_array=self._shifted_zz_emiss,
                metall_model=yaml_data['metall_formula'],
                metall_params=yaml_data['metall_params'],
                verbose=self._log_prints))

            self.logging_info(
                'SSP emissivity: set time integration cube')

        if (self._last_yaml_emiss is None
                or np.any(self._last_yaml_emiss['metall_params']
                          != yaml_data['metall_params'])):
            self._mean_metall_cube_log = log10_safe(metall_model(
                zz_array=self._shifted_zz_emiss,
                metall_model=yaml_data['metall_formula'],
                metall_params=yaml_data['metall_params'],
                verbose=self._log_prints))
            self.logging_info('Dust reem: mean metall calculation')

        # Interior of emissivity integral:
        # L{t(z)-t(z')} * dens(z') * |d(log10(t'))/dt'|
        self._kernel_emiss = (
                10. ** self._log_t_ssp_intcube  # Variable change,
                * np.log(10.)  # integration over y=log10(x)
                * 10. **  # L(t)
                self.ssp_lumin_spline(
                    freq_array=self._freq_array,
                    age_array=self._log_t_ssp_intcube,
                    metall_array=self._mean_metall_cube_log)
        )
        self.logging_info('SSP emissivity: set the initial kernel')

        # Dust absorption (applied in log10)
        fract_dust_Notabs = dust_abs_fraction(
            wv_array=self._lambda_array,
            models=yaml_data['dust_abs_models'],
            z_array=np.squeeze(self._z_array),
            dust_params=yaml_data['dust_abs_params'],
            verbose=self._log_prints)[:, :, np.newaxis]

        kernel_emiss = self._kernel_emiss * fract_dust_Notabs
        self.logging_info('SSP emissivity: set dust absorption')

        # Dust reemission loading ------------------------------------
        if (yaml_data['dust_reem'] is True
                and
                yaml_data['dust_reem_parametrization']['library']
                != 'black_bodies'):
            self.logging_info('Dust reem: enter')

            lumin_abs = (
                    10 ** self._freq_array
                    * np.log(10.)  # integration over y=log10(x)
                    * 10. **  # L(t)
                    self.ssp_lumin_spline(
                        freq_array=self._freq_array,
                        age_array=self._log_t_ssp_intcube,
                        metall_array=self._mean_metall_cube_log)
                    * (1. - fract_dust_Notabs)
            )
            self.logging_info('Dust reem: lumin_abs calc')

            l_int_abs = simpson(lumin_abs, x=self._freq_array, axis=0)
            self.logging_info('Dust reem: integration of Lssp')

            # dust_reem_spline = self.spline_dust_reemission(
            #     yaml_data['dust_reem_parametrization'])
            self.logging_info('Dust reem: dust spline creation')

            kernel_emiss += (
                    10. ** self._log_t_ssp_intcube  # Variable change,
                    * np.log(10.)
                    * 10 ** dust_reem(
                wave_array=np.log10(self._lambda_array)[:, np.newaxis, np.newaxis],
                l_tir_array=np.log10(l_int_abs)[np.newaxis, :, :],
                metall_array=self._mean_metall_cube_log,
                dust_reem_model=yaml_data['dust_reem_parametrization']['library'],
                dust_reem_params=yaml_data['dust_reem_parametrization']
                 )
            )
            self.logging_info('Dust reem: sum to kernel_emiss')

        # SFR multiplication
        kernel_emiss *= sfr_model(
            zz_array=self._shifted_zz_emiss,
            sfr_model=yaml_data['sfr_formula'],
            sfr_params=yaml_data['sfr_params'],
            verbose=self._log_prints)

        self.logging_info('SSP emissivity: calculate ssp kernel')

        # Calculate emissivity in units
        # [erg s^-1 Hz^-1 Mpc^-3] == [erg Mpc^-3]
        emiss_ssp_cube = simpson(
            kernel_emiss, x=self._log_t_ssp_intcube, axis=-1)

        self.logging_info('SSP emissivity: integrate emissivity')

        if (yaml_data['dust_reem'] is True and
                yaml_data['dust_reem_parametrization']['library']
                == 'black_bodies'):

            fract_reem = np.squeeze(
                (1. - fract_dust_Notabs) / fract_dust_Notabs)

            def bb_plank(T):
                freq = np.squeeze(self._freq_array)
                xx = ((c.h * 10 ** freq * u.Hz
                       / c.k_B / T / u.K).to(1)).value
                yy = (15. / np.pi ** 4. / 10 ** freq
                      * xx ** 4. / (np.exp(xx) - 1.))

                return yy

            norm_shape_reem = np.zeros_like(np.squeeze(self._freq_array))
            if type(yaml_data['dust_reem_parametrization']['T'])\
                    == np.float64:
                array_tt = np.array(
                    [yaml_data['dust_reem_parametrization']['T']])

            else:
                array_tt = np.asarray(
                    yaml_data['dust_reem_parametrization']['T'])
                array_fract = np.array(
                    yaml_data['dust_reem_parametrization']['fracts'])

            for nn, tt in enumerate(array_tt):
                if nn == len(array_tt) - 1:
                    norm_shape_reem += (  # min(
                            (1. - (array_fract))  # ,
                            # 1.)
                            * bb_plank(array_tt[-1])
                    )
                else:
                    norm_shape_reem += (
                            array_fract[nn]
                            * bb_plank(tt))

            int_emiss_reem = simpson(
                (fract_reem * emiss_ssp_cube
                 * 10 ** np.squeeze(self._freq_array, axis=2) * np.log(10)),
                x=np.squeeze(self._freq_array), axis=0)

            emiss_ssp_cube += (int_emiss_reem[np.newaxis, :]
                               * norm_shape_reem[:, np.newaxis])

        # Spline of the emissivity

        self._emiss_ssp_spline = RegularGridInterpolator(
            points=(np.squeeze(self._freq_array),
                    np.squeeze(self._z_array)),
            values=log10_safe(emiss_ssp_cube),
            method='linear',
            bounds_error=False, fill_value=-43
        )

        # Free memory and log the time
        del kernel_emiss, emiss_ssp_cube
        self.logging_info('SSP emissivity: end')
        self._last_yaml_emiss = copy.deepcopy(yaml_data)
        self._last_ssp_log_time_init = self._ssp_log_time_init.copy()

        return

    def ebl_ssp_calculation(self, yaml_data, emiss_spline=None):
        """
        Calculate the EBL SSP contribution.

        EBL units:
        ----------
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        yaml_data: dictionary
            Data necessary to reconstruct the EBL component from an SSP.
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        emiss_spline: RegularGridInterpolator or None
            Spline contaiting the emissivity of a source.
            Points: (log10(frequency), redshift)
            Values: log10(emissivity)
            We suggest:
            method='linear', bounds_error=False, fill_value=-43
        """
        if emiss_spline is None:
            self.emiss_ssp_calculation(yaml_data)
        else:
            self._emiss_ssp_spline = emiss_spline

        if (self._ebl_intcube is None
                or np.shape(self._ebl_intcube) != self._cube):
            self._ebl_ssp_cubes()

        self.logging_info('SSP EBL: calculation of z cube')

        # Calculate integration values
        ebl_intcube = self._ebl_intcube * 10. ** self._emiss_ssp_spline(
            (self._shifted_freq, self._ebl_z_intcube))
        self.logging_info('SSP EBL: calculation of kernel interpolation')

        # Integration of EBL from SSP
        ebl_ssp_cube = simpson(
            ebl_intcube, x=self._ebl_z_intcube, axis=-1)
        self.logging_info('SSP EBL: integration')

        # Spline of the SSP EBL intensity
        self._ebl_ssp_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_safe(ebl_ssp_cube),
            kx=1, ky=1)

        # Free memory and log the time
        del ebl_intcube
        self.logging_info('SSP EBL: done')
        return

    def _ebl_ssp_cubes(self, freq_positions=None):
        # Cubes needed to calculate the EBL from an SSP
        if freq_positions is None:
            self._ebl_z_intcube = (self._z_array
                                   + self._steps_integration_cube
                                   * (np.max(self._z_array)
                                      - self._z_array))

            self._shifted_freq = (self._freq_array + log10_safe(
                (1. + self._ebl_z_intcube) / (1. + self._z_array)))

            self._ebl_intcube = (10. ** self._freq_array
                                 * c.c.value / 4. / np.pi)
            self._ebl_intcube = (self._ebl_intcube
                                 / ((1. + self._ebl_z_intcube)
                                    * self._cosmo.H(
                                self._ebl_z_intcube).to(
                                u.s ** -1).value))
        else:
            # Careful, the expressions are only prepared for z=0
            self._ebl_z_intcube = np.ones((len(freq_positions),
                                           self._t_intsteps))
            self._ebl_z_intcube *= (
                    np.max(self._z_array)
                    * np.linspace(0., 1., self._t_intsteps))

            self._shifted_freq = (
                    freq_positions[:, np.newaxis]
                    + np.log10((1. + self._ebl_z_intcube)))

            self._ebl_intcube = (c.c.value * 10 ** freq_positions
                                 / 4. / np.pi)[:, np.newaxis]
            self._ebl_intcube = (self._ebl_intcube
                                 / ((1. + self._ebl_z_intcube)
                                    * self._cosmo.H(
                                self._ebl_z_intcube).to(
                                u.s ** -1).value))

        self.logging_info('Initialize cubes: calculation of division')

        # Mpc^-3 -> m^-3, erg/s -> nW
        self._ebl_intcube *= (u.erg * u.Mpc ** -3 * u.s ** -1
                              ).to(u.nW * u.m ** -3)

        self.logging_info('Initialize cubes: calculation of kernel')
        return

    def ebl_ssp_individualData(self, yaml_data, x_data, sfr=None,
                               emiss_spline=None):
        """
        Calculate the EBL SSP contribution, for the specific wavelength
        data that we have available (and redshift z=0).

        ONLY WORKS FOR REDSHIFT 0!!! (because of intcubes, the redshift
        dependency is not implemented. It does not seem useful.)

        Useful for fittings, because it allows to avoid
        big calculations with a wide grid on wavelengths and redshifts.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1

        Parameters
        ----------
        yaml_data: dictionary
            Data necessary to reconstruct the EBL component from an SSP.
        x_data: 1D array
            Wavelength array
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        emiss_spline: RegularGridInterpolator or None
            Spline containing the emissivity of a source.
            Points: (log10(frequency), redshift)
            Values: log10(emissivity)
            We suggest:
            method='linear', bounds_error=False, fill_value=-43
        """
        self.logging_info('SSP EBL: enter program')

        if emiss_spline is None:
            self.emiss_ssp_calculation(yaml_data, sfr=sfr)
        else:
            self._emiss_ssp_spline = emiss_spline

        self.logging_info('SSP EBL: emissivity calculated')

        new_individualFreq = np.log10(c.c.value / x_data[::-1] * 1e6)

        if (self._ebl_intcube is None
                or np.shape(self._ebl_intcube)[0] == self._cube[0]):
            self._ebl_ssp_cubes(freq_positions=new_individualFreq)

        self.logging_info('SSP EBL: calculation of z cube')

        # Calculate integration values
        ebl_intcube = (
                self._ebl_intcube * 10. ** self._emiss_ssp_spline(
            (self._shifted_freq, self._ebl_z_intcube)))
        self.logging_info('SSP EBL: calculation of kernel interpolation')

        return simpson(ebl_intcube, x=self._ebl_z_intcube,
                       axis=-1)[::-1]

    def emiss_intrahalo_calculation(self, log10_Aihl, alpha):
        """
        Emissivity contribution from Intra-Halo Light (IHL).
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be
        emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the
        galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter
        alpha, as (1 + z)**alpha.

        Emissivity units:
        ----------
            -> Cubes: erg / s / Hz / Mpc**3 == 1e-7 W / Hz / Mpc**3
            -> Splines: log10(erg / s / Hz / Mpc**3 == 1e-7 W / Hz / Mpc**3)

        Parameters
        ----------
        log10_Aihl: float
            Exponential of the IHL intensity. Default: -3.23.
        alpha: float
            Index of the redshift dependency of the IHL. Default: 1.
        """
        Aihl = 10 ** log10_Aihl
        f_ihl = lambda x: Aihl * (x / 1e12) ** 0.1

        # log10 of masses over which we integrate the IHL contribution
        m_min = np.log10(1e9)
        m_max = np.log10(1e13)

        # Use the spectrum of a synthetic old galaxy (taken from
        # the SWIRE library), and normalize it at a wavelength
        # of 2.2 microns.
        old_spectrum = np.loadtxt('Swire_library/Ell13_template_norm.sed')
        old_spline = UnivariateSpline(old_spectrum[:, 0],
                                      old_spectrum[:, 1],
                                      s=0, k=1, ext=1)

        # S_lambda = F_lambda * lambda
        old_spectrum[:, 1] *= (
                old_spectrum[:, 0] / old_spline(22000) / 22000.)
        old_spectrum[:, 0] *= 1e-4
        old_spectrum_spline = UnivariateSpline(
            np.log10(old_spectrum[:, 0]), np.log10(old_spectrum[:, 1]),
            s=0, k=1)

        # Initialize an object to calculate dn/dM
        mf = MassFunction(cosmo_model=self._cosmo, Mmin=m_min, Mmax=m_max)

        # Total luminosity of a galaxy at 2.2 microns
        L22 = 5.64e12 * (self._h / 0.7) ** (-2) * (
                mf.m / 2.7e14 * self._h / 0.7) ** 0.72 / 2.2e-6

        # The object to calculate dn/dM returns this quantity for
        # a specific redshift
        kernel_intrahalo = np.zeros(
            (len(self._freq_array), len(self._z_array)))

        # Calculate the kernel for different redshifts
        for nzi, zi in enumerate(self._z_array):
            mf.update(z=zi)
            lambda_luminosity = (
                    (f_ihl(mf.m) * L22 * (1 + zi) ** alpha)[:, np.newaxis]
                    * 10 ** old_spectrum_spline(
                np.log10(self._lambda_array[np.newaxis, :]))
            )
            kernel = (mf.m[:, np.newaxis]  # Variable change,
                      * np.log(10.)  # integration over log10(M)
                      * lambda_luminosity
                      * mf.dndm[:, np.newaxis]
                      )
            kernel_intrahalo[:, nzi] = simpson(
                kernel, x=np.log10(mf.m), axis=0)

        emiss_ihl_cube = kernel_intrahalo * c / (
                10 ** self._freq_array[:, np.newaxis]) ** 2. * u.s ** 2
        emiss_ihl_cube *= (
                u.solLum.to(u.W) * u.W / (u.Mpc * self._h) ** 3
        ).to(u.erg / u.s / u.Mpc ** 3)

        # Spline of the luminosity
        self._emiss_ihl_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_safe(emiss_ihl_cube.value),
            kx=1, ky=1)

        # Free memory and log the time
        del old_spectrum, old_spectrum_spline, old_spline, mf, L22
        del kernel_intrahalo, lambda_luminosity, kernel
        del emiss_ihl_cube
        self.logging_info('Calculation time for emissivity ihl')
        return

    def ebl_intrahalo_calculation(self, log10_Aihl, alpha):
        """
        EBL contribution from Intra-Halo Light (IHL).
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be
        emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the
        galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter
        alpha, as (1 + z)**alpha.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        log10_Aihl: float
            Exponential of the IHL intensity. Default: -3.23.
        alpha: float
            Index of the redshift dependency of the IHL. Default: 1.
        """
        if self._emiss_ihl_spline is None:
            self.emiss_intrahalo_calculation(log10_Aihl, alpha)

        # Integrate the EBL intensity kernel over values of z
        z_integr = (self._z_array
                    + (self._z_max - self._z_array)
                    * self._steps_integration_cube)

        kernel_ebl_intra = (
                10 ** ((self._emiss_ihl_spline.ev(
            (self._freq_array
             + np.log10((1. + z_integr) / (1. + self._z_array))
             ).flatten(),
            z_integr.flatten())
                       ).reshape(self._cube))
                / ((1. + z_integr)
                   * self._cosmo.H(z_integr).to(u.s ** -1).value))

        ebl_ihl_cube = simpson(kernel_ebl_intra, x=z_integr, axis=-1)

        ebl_ihl_cube *= u.erg * u.s / u.Mpc ** 3
        ebl_ihl_cube *= (c ** 2 / (self._lambda_array[:, np.newaxis]
                                   * 1e-6 * u.m * 4. * np.pi))

        ebl_ihl_cube = ebl_ihl_cube.to(u.nW / u.m ** 2).value

        # Spline of the IHL EBL intensity
        self._ebl_ihl_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_safe(ebl_ihl_cube),
            kx=1, ky=1)

        # Free memory and log the time
        del kernel_ebl_intra, z_integr, ebl_ihl_cube
        self.logging_info('Calculation time for ebl ihl')
        return

    def ebl_axion_calculation(self, wavelength, zz_array,
                              axion_mass, axion_gayy, factor=1e-3):
        """
        EBL contribution from axion decay.
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        axion_mass: float [eV]
            Value of (m_a * c**2) of the decaying axion.
        axion_gayy: float [GeV**-1]
            Coupling constant of the axion with two photons (decay).
        """
        if axion_mass < 1e-43:
            try:
                return 1e-43 * np.ones((len(wavelength), len(zz_array)))
            except TypeError:
                return 1e-43 * np.ones(1)

        else:
            axion_mass = axion_mass * u.eV
            axion_gayy = axion_gayy * u.GeV ** -1

            decay_wv_alp = (2. * c.c * c.h / axion_mass).to(u.micron).value

            wavelength = np.concatenate((
                wavelength, [decay_wv_alp,
                             decay_wv_alp * (1. - factor),
                             decay_wv_alp * (1. + factor)]
            ))
            wavelength = np.sort(wavelength)

            freq = (c.c.value / wavelength * 1e6 * u.s ** -1)

            z_star = (axion_mass * (1. + zz_array)
                      / (2. * c.h.to(u.eV * u.s) * freq)
                      - 1.)

            ebl_axion_cube = (
                (self._cosmo.Odm(0.) * self._cosmo.critical_density0
                 * c.c ** 3. / (64. * np.pi * u.sr)
                 * axion_gayy ** 2. * axion_mass ** 2.
                 * freq
                 / self._cosmo.H(z_star)
                 ).to(u.nW * u.m ** -2 * u.sr ** -1)
            ).value

            ebl_axion_cube[z_star < zz_array] = 1e-43

        # Free memory and log the time
        del z_star
        self.logging_info('Calculation time for ebl axions')
        return wavelength, ebl_axion_cube

    def ebl_sum_contributions(self):
        """
        Sum of the contributions to the EBL which have been
        previously calculated.
        If any of the components has not been calculated,
        its contribution will be 0.
        Components are Single Stellar Populations (SSP),
        Intra-Halo Light (IHL) and axion decay.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)
        """
        # Spline of the total EBL intensity
        total_ebl = 0.

        # if self.ebl_ssp_spline is not None:
        #     total_ebl += self.ebl_ssp_calculation()
        log10_ebl = np.log10(1.)
        self._ebl_total_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del log10_ebl
        self.logging_info('Calculation time for ebl total')
        return

    def write_ebl_to_ascii(self, output_path='',
                           name='ebl', source='ssp'):
        """
        Creates a file containing the values for wavelength, redshift and
        EBL values, of dimensions (n+1) x (m+1).
        The zeroth column [1:] contains the n wavelength values in mu meters.
        The first row [1:] contains the m redshift values.
        The remaining values (n x m) are the EBL photon density values
        in nW / m^2 / sr.
        The [0,0] entry is empty.

        Parameters
        ----------
        output_path: str
            Path where to create the txt file
        name: str
            name of the txt file
        """
        aaa = np.zeros((len(self._lambda_array) + 1,
                        len(np.squeeze(self._z_array)) + 1))
        aaa[1:, 0] = self._lambda_array[::-1]
        if source == 'ssp':
            aaa[1:, 1:] = (self.ebl_ssp_spline(
                wv_array=self._lambda_array[::-1],
                zz_array=self._z_array)).T.squeeze()
        elif source == 'ihl':
            aaa[1:, 1:] = (self._ebl_ihl_spline(
                wv_array=self._lambda_array[::-1],
                zz_array=self._z_array)).T.squeeze()
        else:
            print('Source not recognized, check another one.')
            print('Source types admitted: ssp, ihl')
            exit()

        aaa[0, 1:] = np.squeeze(self._z_array)

        np.savetxt(output_path + '/' + name + '.txt', aaa)

        return

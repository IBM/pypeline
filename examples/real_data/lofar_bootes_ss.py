# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (StandardSynthesis).
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as ProgressBar
from scipy.constants import speed_of_light

import pypeline.phased_array.bluebild.data_processor as data_proc
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as param_est
import pypeline.phased_array.util.data_gen.sky as sky
import pypeline.phased_array.util.data_gen.visibility as visibility
import pypeline.phased_array.util.gram as gr
import pypeline.phased_array.util.grid as grid
import pypeline.phased_array.util.io.image as image
import pypeline.phased_array.util.io.ms as measurement_set
import pypeline.util.math.sphere as sph

# Instrument
N_station = 24
ms_file = ('/home/sep/Documents/IBM/Data/RADIO-ASTRONOMY/'
           'LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms')
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
gram = gr.GramBlock()

# Observation
field_of_view = np.radians(5)
channel_idx = 0
frequency = ms.channels[channel_idx]
wl = speed_of_light / frequency
sky_model = sky.from_tgss_catalog(ms.field_center, field_of_view, N_src=20)

# Imaging
N_level = 4
N_bits = 32
pix_q, pix_l, pix_colat, pix_lon = grid.ea_harmonic_grid(
    direction=ms.field_center.cartesian.xyz.value,
    FoV=field_of_view,
    N=ms.instrument.nyquist_rate(wl))
pix_grid = np.stack(sph.pol2cart(1, pix_colat, pix_lon), axis=0)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = param_est.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t_idx, _, S in ProgressBar(ms.visibilities(channel_id=channel_idx,
                                               time_id=slice(None, None, 200),
                                               column='DATA_SIMULATED')):
    XYZ = ms.instrument(ms.time[t_idx])
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = data_proc.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, pix_grid, N_level, N_bits)
for t_idx, _, S in ProgressBar(ms.visibilities(channel_id=channel_idx,
                                               time_id=slice(None, None, 1),
                                               column='DATA_SIMULATED')):
    XYZ = ms.instrument(ms.time[t_idx])
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
I_std_c, I_lsq_c = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = param_est.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time[::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = data_proc.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, pix_grid, 1, N_bits)
for t_idx, _, S in ProgressBar(ms.visibilities(channel_id=channel_idx,
                                               time_id=slice(None, None, 50),
                                               column='DATA_SIMULATED')):
    XYZ = ms.instrument(ms.time[t_idx])
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S_c = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq_c = getattr(image, 'SphericalImageContainer_float' + str(N_bits))(I_std_c.image / S_c.image, I_std_c.grid)
I_std_eq = image.SphericalImage(I_std_eq_c)
I_std_eq.draw(catalog=sky_model, ax=ax[0])
ax[0].set_title('Bluebild Standardized Image')

I_lsq_eq_c = getattr(image, 'SphericalImageContainer_float' + str(N_bits))(I_lsq_c.image / S_c.image, I_lsq_c.grid)
I_lsq_eq = image.SphericalImage(I_lsq_eq_c)
I_lsq_eq.draw(catalog=sky_model, ax=ax[1])
ax[1].set_title('Bluebild Least-Squares Image')

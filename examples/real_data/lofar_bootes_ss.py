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

import pypeline.phased_array.bluebild.data_processor as data_proc
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as param_est
import pypeline.phased_array.util.data_gen as dgen
import pypeline.phased_array.util.gram as gr
import pypeline.phased_array.util.grid as grid
import pypeline.phased_array.util.io as io
import pypeline.phased_array.util.io.ms as measurement_set
import pypeline.util.math.sphere as sph

# Instrument
N_station = 24
ms_file = ('/home/sep/Documents/IBM/Data/RADIO-ASTRONOMY/'
           'LOFAR/BOOTES24_SB180-189.2ch8s_SIM.ms')
ms = measurement_set.LofarMeasurementSet(ms_file, N_station)
gram = gr.GramBlock()

# Observation
field_of_view = 5 * u.deg
channel_id = 0
frequency = ms.channels['FREQUENCY'][channel_id]
sky_model = dgen.from_tgss_catalog(ms.field_center, field_of_view, N_src=20)

# Imaging
N_level = 4
pix_q, pix_l, pix_colat, pix_lon = grid.ea_harmonic_grid(
    direction=ms.field_center.cartesian.xyz.value,
    FoV=field_of_view,
    N=ms.geometry.nyquist_rate(frequency))
pix_grid = sph.pol2cart(1, pix_colat, pix_lon)

### Intensity Field ===========================================================
# Parameter Estimation
I_est = param_est.IntensityFieldParameterEstimator(N_level, sigma=0.95)
for t, f, S in ProgressBar(ms.visibilities(channel_id=[channel_id],
                                           time_id=slice(None, None, 200),
                                           column='DATA_SIMULATED')):
    XYZ = ms.geometry(t)
    W = ms.beamformer(XYZ, f)
    G = gram(XYZ, W, f)
    S, _ = io.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = data_proc.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(frequency, pix_grid, N_level)
for t, f, S in ProgressBar(ms.visibilities(channel_id=[channel_id],
                                           time_id=slice(None, None, 50),
                                           column='DATA_SIMULATED')):
    XYZ = ms.geometry(t)
    W = ms.beamformer(XYZ, f)
    G = gram(XYZ, W, frequency)
    S, W = io.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
I_std, I_lsq = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = param_est.SensitivityFieldParameterEstimator(sigma=0.95)
for t in ProgressBar(ms.time['TIME'][::200]):
    XYZ = ms.geometry(t)
    W = ms.beamformer(XYZ, frequency)
    G = gram(XYZ, W, frequency)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = data_proc.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(frequency, pix_grid, N_level=1)
for t, f, S in ProgressBar(ms.visibilities(channel_id=[channel_id],
                                           time_id=slice(None, None, 50),
                                           column='DATA_SIMULATED')):
    XYZ = ms.geometry(t)
    W = ms.beamformer(XYZ, f)
    G = gram(XYZ, W, f)
    S, W = io.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = io.SphericalImage(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model, ax=ax[0])
ax[0].set_title('Bluebild Standardized Image')

I_lsq_eq = io.SphericalImage(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model, ax=ax[1])
ax[1].set_title('Bluebild Least-Squares Image')

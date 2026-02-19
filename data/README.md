# Data provenance

This folder contains input data used by the compact notebooks and plotting pipeline.
Most instrument noise curves were **digitized from figures in the cited papers** using
an online tool (automeris.io). The notes below summarize the sources and what is
included in each dataset (instrumental noise only vs. sky/QTN already included).

## Instrument data (digitized)

### Lunar probes
- **NCLE** (Chang’e‑4, Earth–Moon L2)
  - Source: Karapakula et al. 2024, Figs. 21–23
  - Quantity: SEFD\_rx in W/m²/Hz
  - Includes: receiver noise only (no Galactic/QTN)

- **DEX** (planned lunar farside array)
  - Source: Brinkerink et al. 2025, Fig. 4
  - Quantity: sensitivity in W/m²/Hz for t_int = 1 hr, Δν = 0.5ν
  - Includes: Galactic background; QTN negligible in this band

- **FARSIDE**
  - Source: Burns et al. 2021, Table 1
  - Quantity: sensitivity in Jy at 200 kHz and 15 MHz
  - Includes: Galactic + QTN

- **FarView**
  - Source: Polidan et al. 2024
  - Quantity: sensitivity in mJy at 15 MHz and 40 MHz for 1 min, Δν = 0.5ν
  - Includes: Galactic background

### Solar probes
- **Solar Orbiter / RPW**
  - Source: Maksimovic et al. 2020, Fig. 27
  - Quantity: receiver noise in V/√Hz (converted to SEFD\_rx using Eq. SEFD\_rx)
  - Includes: receiver noise only (no Galactic/QTN)

- **Parker Solar Probe / FIELDS**
  - Source: Pulupa et al. 2017, Fig. 3
  - Quantity: receiver noise in V/√Hz (converted to SEFD\_rx)
  - Includes: receiver noise only (no Galactic/QTN)

### Planetary probes
- **Cassini / RPWS**
  - Source: Gurnett et al. 2004, Fig. 24
  - Quantity: in‑flight E²\_rx noise (already includes environment)
  - Includes: environmental (Galactic/QTN) contributions

- **Juno / WAVES**
  - Source: Kurth et al. 2017, Fig. 27
  - Quantity: in‑flight E²\_rx noise (already includes environment)
  - Includes: environmental (Galactic/QTN) contributions

- **JUICE / RWI**
  - Source: Wahlund et al. 2025, Fig. 17
  - Quantity: expected E\_rx noise (digitized)
  - Includes: receiver noise only (Galactic/QTN added separately)

## Trajectories
Files under `data/trajectories/` are extracted from JPL HORIZONS and used to compute
flyby‑dependent effective observing times (t_eff).

JPL HORIZONS: https://ssd.jpl.nasa.gov/horizons/app.html

## Bounds
Bounds for ALP–photon coupling and dark‑photon kinetic mixing used in the final plots
are taken from the AxionLimits compilation:

https://cajohare.github.io/AxionLimits/

# Predictor provenance, verified 27 August 2026

Every predictor in the master table traces to Environment90m
(Garcia Marquez et al. 2026, ESSD 18, 1541-1559), downloadable as zipped
per-tile CSVs from the IGB server. Verified by downloading one tile and
matching values against the master table on shared subc_id.

| family | source | transform | verification |
|---|---|---|---|
| CLI | chelsa_bioclim_v2_1 / 1981-2010_observed | (x - 2731.5)/10 for min,max,mean; x/10 for sd | r = 1.000000, max abs diff 5e-5 (rounding) |
| SOL | soilgrids250m_v2_0 | none | r = 1.000000, factor 1 |
| TOP | hydrography90m_v1_0 | none | r = 1.000000, factor 1 |
| LAC | esa_cci_landcover_v2_1_1, year 2020 | none | exact zero difference |

Notes:
- Server file names use zero-padded bioclim codes (bio01), the dictionary
  does not (bio1).
- Variable order in the dictionary is NOT the order of the l_/u_ numbering:
  acdwrb maps to l_SOL45-48, not l_SOL1-4. Align by name, never by row position.
- Land cover 2019 and 2020 are identical across our subcatchments for every
  class tested, so the choice does not affect values; 2020 is what we state.
- Master table stores four decimals, so mean and sd agree to ~1e-4, not exactly.
- Some SOL columns hold the string 'na'; parse with errors="coerce".

Tiles covering 99% of the panel (20): h02v02 h03v02 h03v03 h04v02 h04v03
h05v02 h05v04 h07v02 h08v01 h08v02 h09v01 h09v02 h10v01 h10v02 h10v04
h14v02 h14v03 h15v02 h15v03 h16v02

Base URL:
https://public.igb-berlin.de/index.php/s/zw56kEd25NsQqcQ/download?path=%2F
plus the file_path from _file_lists/env90m_{soil,hydro,observedclimate,landcover}_paths_file_sizes.txt
with "/" replaced by "%2F".

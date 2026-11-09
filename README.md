# Peptide-Membrane-SMD-workflow


TAT PEPTIDE  ·  FULL SIMULATION WORKFLOW
Peptide  →  Membrane  →  SMD
Sequence: GRKKRRQRRRPQ
════════════════════════════════════════════════════════════════════


OVERVIEW
────────────────────────────────────────────────────────────────────
  Stage 1   Peptide build, MD simulation and analysis
  Stage 2   Peptide-membrane system build, MD simulation and analysis
  Stage 3   Pre-MD simulation for SMD system preparation
  Stage 4   SMD simulation and analysis


REQUIRED TOOLS
────────────────────────────────────────────────────────────────────
  a)  PEP-FOLD server (or AlphaFold / similar)  ·  peptide modeling
  b)  CHARMM36m force field: charmm36-feb2026_cgenff-5.0.ff.tgz
      https://mackerell.umaryland.edu/charmm_ff.shtml
      (extract the .ff folder into the working directory)
  c)  CHARMM-GUI server
  d)  GROMACS
  e)  PyMOL or VMD  ·  visual inspection of structures and trajectories
  f)  Python  ·  plotting .xvg / .xpm outputs and building the FEL

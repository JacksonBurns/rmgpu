# Job 00 Gate Report

## Environment Versions

```
aimsim_core==2.2.3
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-types==0.8.0
astartes==1.3.3
attrs==26.1.0
cantera==3.2.0
chemicals==1.5.2
chemprop==2.3.1
click==8.4.2
ConfigArgParse==1.7.5
cuda-bindings==13.3.1
cuda-pathfinder==1.7.0
cuda-toolkit==13.0.3.0
cuik_molmaker_pin==2026.3.5
descriptastorus==2.8.0
dill==0.4.1
filelock==3.32.4
flexcache==0.3
flexparser==0.4
fluids==1.3.1
frozenlist==1.8.0
fsspec==2026.7.0
greenlet==3.5.5
idna==3.19
iniconfig==2.3.0
Jinja2==3.1.6
joblib==1.5.3
lightning==2.6.5
lightning-utilities==0.15.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mhfp==1.9.6
mordredcommunity==2.0.7
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
myerson==1.0.1
narwhals==2.25.0
networkx==3.6.1
numpy==2.4.6
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.3.33
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1785888127/work
padelpy==0.1.17
pandas==3.0.5
pandas-flavor==0.8.1
pillow==12.3.0
Pint==0.25.3
platformdirs==4.11.4
pluggy==1.6.0
polars==1.44.0
polars-runtime-32==1.44.0
propcache==0.5.2
psutil==7.2.2
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.21.0
pytest==9.1.1
python-dateutil==2.9.0.post0
pytorch-lightning==2.6.5
PyYAML==6.0.3
rdkit==2026.3.5
rich==15.0.0
-e git+ssh://git@github.com/JacksonBurns/rmgpu.git@605d74fefbbb06ed1aca00ca13181d418ab3fb27#egg=rmgpu
ruamel.yaml==0.19.1
scikit-learn==1.9.0
scipy==1.17.1
six==1.17.0
SQLAlchemy==2.0.52
sympy==1.14.0
tabulate==0.10.0
thermo==0.6.1
threadpoolctl==3.6.0
torch==2.13.0
torchdae==0.1.1
torchmetrics==1.9.0
tqdm==4.70.0
triton==3.7.1
typing-inspection==0.4.4
typing_extensions==4.16.0
xarray==2026.7.0
yarl==1.24.5

```

## Environment Changes Since Job-00

- **openbabel** (conda-forge) added to the `rmgpu` env in job-01/step-08
  (2026-08-25). Required because rmgpu's `Molecule.to_smiles` mirrors
  RMG-Py's translator, which canonicalizes N/S-containing molecules with
  OpenBabel (`rmgpu/molecule/molecule.py` falls back to RDKit if the import
  fails, but then `gates/gate_01.py` smiles_parity fails for NO2 and HNO3).
  Install: `conda install -n rmgpu -c conda-forge openbabel`.

## Checkpoint Inventory

From step-01 report:

- **Path:** `/home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt`

- **Format:** `.ckpt` (PyTorch zip archive)

- **Loaded via chemprop example pattern:** `models.MPNN.load_from_checkpoint(checkpoint_path)` succeeded.

- **Featurizer:** `SimpleMoleculeMolGraphFeaturizer` (from chemprop example).

- **Output dims:** single-task regression model, one prediction per molecule.

- **1-molecule predict test:** ethane (CC) predicted value = `2.166739`.

The checkpoint inventory is complete for job 04 to consume.

## rmgdb Install Method

rmgdb is importable (confirmed in check_env.py output).

## Gate Result

PASS

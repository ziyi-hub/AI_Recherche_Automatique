# AI_Recherche_Automatique
Documents multimédia : description et recherche automatique

## Liens utils ##
- [Cours](http://gbgi9u07.imag.fr/doku.php)

## Verifier cuda-gdb ##
- cuda-gdb --version
- [Installation](https://developer.nvidia.com/nvidia-cuda-toolkit-12_2_0-developer-tools-mac-hosts#cuda-gdb)

## Access ssh grid5000
- ssh ziwang@access.grid5000.fr
- https://intranet.grid5000.fr/notebooks/hub/home
- scp ziwang@access.grid5000.fr:/mnt/home/lille/ziwang/TP3.ipynb ./TP3.ipynb
- scp ziwang@access.grid5000.fr:/mnt/home/rennes/ziwang/Projet_2.ipynb ./Projet_5.ipynb
- nvidia-smi
- 82.390% epoche 60
- 80.250% epoche 50

## Install without terminal
- !pip install torchvision

## grid5000 cluster
- Modifiez (à chaque fois) les options suivantes comme indiqué :
- Select a site : → Lille
- Requested Resources : → /gpu=1
- Walltime : 1:30 (ou moins selon la durée restante dans le conteneur)
- Container ID (for inner jobs): 1939271 (29/03/2023 de 09h30 à 11h30)
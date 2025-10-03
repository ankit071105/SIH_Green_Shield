# Colab Auto-Run Instructions (one-click training on PlantVillage)

1. Open Google Colab and create a new notebook.
2. Set Runtime -> Change runtime type -> GPU.
3. Upload your `kaggle.json` file via the Colab file UI (or save it in Drive and copy).
4. Run these commands in Colab cells (or paste the provided script):

```bash
# install dependencies
!pip install -q kaggle torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# set up kaggle credentials (if you uploaded kaggle.json)
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# download PlantVillage (emmarex/plantdisease)
!kaggle datasets download -d emmarex/plantdisease -p /content/plantvillage
!unzip -q /content/plantvillage/plantdisease.zip -d /content/plantvillage

# prepare train/val (85/15) and run training script shipped in the project
# assume project files are available in /content/project
# you can upload the project zip to Colab or copy files into workspace
!python colab_train.py --data_dir /content/plantvillage/PlantVillage --epochs 12 --bs 32 --lr 3e-4 --out /content/best_model.pt --save_to_drive
```

After training, download `/content/best_model.pt` from Colab or from Drive and place it into `models/best_model.pt` in the project folder locally.

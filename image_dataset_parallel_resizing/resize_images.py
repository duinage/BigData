from mpi4py import MPI
from PIL import Image
from pathlib import Path
import shutil

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

base_path = Path("data")
output_path = base_path / "data_resized"

if rank == 0:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()
    # load images
    images = [(path, Image.open(path).convert("RGB")) for path in list(base_path.glob("*/*.jpg"))]
    
    # split images
    images = [images[i::size] for i in range(size)]
else:
    images = None

# pass images to each process
images = comm.scatter(images, root=0)

# resize and save images in each process
for path, img in images:
    resized_img = img.resize((64, 64), Image.Resampling.LANCZOS)

    output_filename = output_path / f"{path.stem}_resized{path.suffix}"
    resized_img.save(output_filename, "JPEG")
    print(f"Rank {rank}: Resized and saved {output_filename}")

# wait until all processes done
comm.Barrier()
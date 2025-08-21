import os
import configs
from utils import data
from utils import process
import gc

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def CPG_generator():
    """
    Generates Code Property Graph (CPG) datasets from raw data.

    :return: None
    """
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)

    # Here, taking the Devign dataset as an example,
    # specific modifications need to be made according to different dataset formats.
    filtered = data.clean(raw)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    cpg_files = [f for f in os.listdir(PATHS.cpg) if f.endswith('.bin')]
    print(cpg_files)
    # Create CPG with graphs json files
    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        del graphs
        gc.collect()
        
if __name__ == "__main__":
    CPG_generator()
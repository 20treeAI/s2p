# Copying stereo matching code here so that we can modify it to parallelize
# the stereo matching step using kubernetes.

import os.path
import os
from pathlib import Path
import shutil

import numpy as np
import rasterio
import rasterio.merge


from s2p import common
from s2p import block_matching
from s2p import masking

k8s_client = None
v1_batch = None


def get_client():
    global k8s_client, v1_batch
    if k8s_client is None:
        from kubernetes import client as k8s_client
        from kubernetes import config

        if os.getenv("KUBERNETES_SERVICE_HOST"):
            # If I am running in Kubernetes, use the "in cluster" configuration
            config.load_incluster_config()
        else:
            config.load_kube_config()

    if v1_batch is None:
        v1_batch = k8s_client.BatchV1Api()

    return k8s_client, v1_batch


def create_mgm_multi_k8s_job(args, in_paths, out_path, k8s_params, env={}):
    assert len(in_paths) == 2, "Expected two input paths"
    assert out_path.endswith(".tif"), "Expected output path to end with .tif"
    assert all(
        p.startswith("/gfs") for p in in_paths + [out_path]
    ), "Expected all paths to start with /gfs"

    k8s_client, v1_batch = get_client()
    tolerations = [
        k8s_client.V1Toleration(
            key="hub.jupyter.org/dedicated", effect="NoSchedule", value="user"
        ),
        k8s_client.V1Toleration(
            key="overstory_nodetype", effect="NoSchedule", value=k8s_params.get("nodepool", "default")
        ),
    ]
    env_vars = [k8s_client.V1EnvVar(name=k, value=v) for k, v in env.items()]
    args = args + in_paths + [out_path]
    job = k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(
            generate_name=k8s_params.get("generate_name"), namespace=k8s_params.get("namespace")
        ),
        spec=k8s_client.V1JobSpec(
            ttl_seconds_after_finished=300,
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(labels={"app": k8s_params.get("app_name")}),
                spec=k8s_client.V1PodSpec(
                    containers=[
                        k8s_client.V1Container(
                            name=k8s_params.get("container_name"),
                            image=k8s_params.get("image_name"),
                            image_pull_policy="Always",
                            args=args,
                            env=env_vars,
                            volume_mounts=[
                                k8s_client.V1VolumeMount(
                                    name="gfs-data", mount_path=k8s_params.get("gfs_mount_path", "/gfs")
                                )
                            ],
                            resources=k8s_client.V1ResourceRequirements(
                                requests={"cpu": k8s_params.get("cpu", 8), "memory": k8s_params.get("ram", "8Gi")},
                            ),
                        )
                    ],
                    restart_policy="Never",
                    volumes=[
                        k8s_client.V1Volume(
                            name="gfs-data",
                            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=k8s_params.get("claim_name")
                            ),
                        )
                    ],
                    node_selector={"cloud.google.com/gke-nodepool": k8s_params.get("nodepool", "default")},
                    tolerations=tolerations,
                ),
            ),
        ),
    )

    api_response = v1_batch.create_namespaced_job(k8s_params.get("namespace"), job)
    return api_response.metadata.name


def _get_gfs_dirs(path):
    """Helper function to get the GFS dir to store the output, locally and on k8s."""
    if os.path.isdir("/gfs"):
        gfs_base = gfs_k8s = "/gfs"
    elif os.path.isdir("/home/jovyan/gfs"):
        gfs_base = "/home/jovyan/gfs"
        gfs_k8s = "/gfs"
    else:
        raise ValueError(
            "No locally mounted GFS directory found at /gfs or /home/jovyan/gfs."
        )

    tiles_index = path.find("/tiles/")

    if tiles_index != -1:
        dir_name = Path(path[:tiles_index]).name
        search = os.path.join(dir_name, "tiles")
        tiles_index = path.find(search)
        if tiles_index == -1:
            raise ValueError(f"Error {search} not found in {path}.")
        path_end = path[tiles_index:]
    else:
        raise ValueError(f"Path to /tiles/ not found in path. ({path})")
    path_start = "s2p_pipelines/mgm_multi/"
    return os.path.join(gfs_base, path_start, path_end), os.path.join(
        gfs_k8s, path_start, path_end
    )


def stereo_matching_k8s(tile, i, cfg, k8s_params):
    """Create a k8s job that runs stereo matching on a tile."""
    if not cfg["matching_algorithm"].startswith("mgm_multi"):
        raise NotImplementedError(
            "Only mgm_multi is supported for parallelization using k8s."
        )
    local_out_dir = os.path.join(tile["dir"], "pair_{}".format(i))
    gfs_dir, gfs_k8s_dir = _get_gfs_dirs(local_out_dir)
    print(f"Local GFS: {gfs_dir}")
    print(f"K8S GFS: {gfs_k8s_dir}")
    os.makedirs(gfs_dir, exist_ok=True)

    x, y = tile["coordinates"][:2]

    print("estimating disparity on tile {} {} pair {}...".format(x, y, i))
    rect1 = os.path.join(local_out_dir, "rectified_ref.tif")
    rect2 = os.path.join(local_out_dir, "rectified_sec.tif")
    disp = os.path.join(local_out_dir, "rectified_disp.tif")
    disp_min, disp_max = np.loadtxt(os.path.join(local_out_dir, "disp_min_max.txt"))

    # Use the local GFS directory to copy the files
    rect1_gfs = os.path.join(gfs_dir, "rectified_ref.tif")
    rect2_gfs = os.path.join(gfs_dir, "rectified_sec.tif")
    shutil.copy(rect1, rect1_gfs)
    shutil.copy(rect2, rect2_gfs)

    # Then set the GFS directory to the k8s directory used by the job
    disp_gfs = os.path.join(gfs_k8s_dir, "rectified_disp.tif")
    rect1_gfs = os.path.join(gfs_k8s_dir, "rectified_ref.tif")
    rect2_gfs = os.path.join(gfs_k8s_dir, "rectified_sec.tif")

    # Block matching part
    max_disp_range = cfg["max_disp_range"]

    # limit disparity bounds
    if disp_min is not None and disp_max is not None:
        with rasterio.open(rect1, "r") as f:
            width = f.width
        if disp_max - disp_min > width:
            center = 0.5 * (disp_min + disp_max)
            disp_min = int(center - 0.5 * width)
            disp_max = int(center + 0.5 * width)

    # round disparity bounds
    if disp_min is not None:
        disp_min = int(np.floor(disp_min))
    if disp_max is not None:
        disp_max = int(np.ceil(disp_max))

    if max_disp_range is not None and disp_max - disp_min > max_disp_range:
        raise block_matching.MaxDisparityRangeError(
            "Disparity range [{}, {}] greater than {}".format(
                disp_min, disp_max, max_disp_range
            )
        )

    # Running mgm multi on k8s
    env = {}
    env["OMP_NUM_THREADS"] = str(cfg["omp_num_threads"])
    env["REMOVESMALLCC"] = str(cfg["stereo_speckle_filter"])
    env["MINDIFF"] = str(cfg["mgm_mindiff_control"])
    env["TESTLRRL"] = str(cfg["mgm_leftright_control"])
    env["TESTLRRL_TAU"] = str(cfg["mgm_leftright_threshold"])
    env["CENSUS_NCC_WIN"] = str(cfg["census_ncc_win"])
    env["SUBPIX"] = "2"
    regularity_multiplier = cfg["stereo_regularity_multiplier"]
    nb_dir = cfg["mgm_nb_directions"]

    P1 = (
        8 * regularity_multiplier
    )  # penalizes disparity changes of 1 between neighbor pixels
    P2 = 32 * regularity_multiplier  # penalizes disparity changes of more than 1
    conf = "{}_confidence.tif".format(os.path.splitext(disp_gfs)[0])
    args = [
        "-r",
        str(disp_min),
        "-R",
        str(disp_max),
        "-S",
        str(6),
        "-s",
        "vfit",
        "-t",
        "census",
        "-O",
        str(nb_dir),
        "-P1",
        str(P1),
        "-P2",
        str(P2),
        "-confidence_consensusL",
        str(conf),
    ]
    return create_mgm_multi_k8s_job(args, [rect1_gfs, rect2_gfs], disp_gfs, k8s_params, env=env)


def postprocess_stereo_matching_k8s(tile, i, cfg):
    """
    After the k8s job has finished, copy the results back to the local directory,
    create the rejection mask and apply erosion
    """
    out_dir = os.path.join(tile["dir"], "pair_{}".format(i))
    rect1 = os.path.join(out_dir, "rectified_ref.tif")
    rect2 = os.path.join(out_dir, "rectified_sec.tif")
    disp = os.path.join(out_dir, "rectified_disp.tif")
    disp_gfs = os.path.join("/gfs", disp)
    shutil.copy(disp_gfs, disp)
    mask = os.path.join(out_dir, "rectified_mask.png")

    block_matching.create_rejection_mask(disp, rect1, rect2, mask, cfg["temporary_dir"])
    # add margin around masked pixels
    masking.erosion(mask, mask, cfg["msk_erosion"])

    if cfg["clean_intermediate"]:
        if len(cfg["images"]) > 2:
            common.remove(rect1)
        common.remove(rect2)
        common.remove(os.path.join(out_dir, "disp_min_max.txt"))

# Shared task -> dataset-date map for the RoboCasa365 data-prep jobs.
#
# The release path embeds a capture date per task (ATOMIC_TASK_DATASETS[task]["target"]
# ["human_path"] in robocasa/utils/dataset_registry.py), so it cannot be derived from the task
# name. Sourced by data_prep_*_jeanzay.slurm; keep in step with the registry.
#
#   source experiments/13_robocasa365/task_paths.sh
#   task_date OpenDrawer   -> 20250816

task_date() {
    case "$1" in
        OpenDrawer)              echo 20250816 ;;
        PickPlaceCounterToStove) echo 20250818 ;;
        TurnOnMicrowave)         echo 20250813 ;;
        # Stage 5's two additions. Both have a `target` split, so the usual
        # `download_demos.py --split target --source human` applies. Note CoffeeSetupMug's
        # PRETRAIN path is 20250819 -- the date below is the target one, which is what every
        # stage here consumes, and the two differ for this task exactly as they do for
        # TurnOnMicrowave.
        CloseBlenderLid)         echo 20250822 ;;
        CoffeeSetupMug)          echo 20250813 ;;
        *) echo "unknown task: $1 (add it from dataset_registry.py)" >&2; return 1 ;;
    esac
}

# Root the downloader and the replay/convert stages agree on. On Jean Zay this is
# robocasa/macros_private.py's DATASET_BASE_PATH; note it differs from the CLEPS layout.
task_input_dir() {
    echo "$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$1/$(task_date "$1")"
}

task_output_dir() {
    echo "$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/$1"
}

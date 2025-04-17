import numpy as np
import os
from scipy.linalg import expm, norm
import logging
import argparse
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def M(axis, theta):
    norm_axis = axis / norm(axis) if norm(axis) > 0 else axis
    return expm(np.cross(np.eye(3), norm_axis * theta))

DEFAULT_TRANSLATION_RANGE = 0.2
DEFAULT_SCALING_RANGE = [0.8, 1.2]
DEFAULT_SCALE_OUT_RANGE = [1.2, 1.5]
DEFAULT_JITTER_SIGMA = 0.03
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_UPSAMPLE_RATE = 1.5
AVAILABLE_AUGMENTATIONS = ['rotate', 'translate', 'scale', 'flip', 'jitter', 'dropout', 'scale_out_dropout', 'upsample']
DEFAULT_COMBINATION_PROBABILITY = 0.3


def _apply_single_augmentation(
    aug_type,
    coords,
    intensity,
    translation_range,
    scaling_range,
    scale_out_range,
    jitter_sigma,
    dropout_rate,
    upsample_rate
):
    """Applies a single specified augmentation. Returns (new_coords, new_intensity)."""
    augmented_coords = coords.copy()
    final_intensity = intensity.copy()

    if aug_type == 'rotate':
        angle_z = np.random.uniform(-np.pi, np.pi)
        axis_z = np.array([0, 0, 1])
        R_z = M(axis_z, angle_z)
        augmented_coords = np.dot(augmented_coords, R_z.T)
    elif aug_type == 'translate':
        translation_vector = np.random.uniform(-translation_range, translation_range, 3)
        augmented_coords += translation_vector
    elif aug_type == 'scale':
        scale_factor = np.random.uniform(scaling_range[0], scaling_range[1])
        augmented_coords *= scale_factor
    elif aug_type == 'flip':
        flip_axis = random.choice(['x', 'y'])
        if flip_axis == 'x': augmented_coords[:, 0] *= -1
        else: augmented_coords[:, 1] *= -1
    elif aug_type == 'jitter':
        jitter_noise = np.random.normal(0, jitter_sigma, augmented_coords.shape)
        augmented_coords += jitter_noise
    elif aug_type == 'dropout':
        num_points = augmented_coords.shape[0]
        num_to_drop = int(num_points * dropout_rate)
        if num_to_drop > 0 and num_points > num_to_drop:
            drop_indices = np.random.choice(num_points, num_to_drop, replace=False)
            augmented_coords = np.delete(augmented_coords, drop_indices, axis=0)
            final_intensity = np.delete(final_intensity, drop_indices, axis=0)
    elif aug_type == 'scale_out_dropout':
        scale_factor = np.random.uniform(scale_out_range[0], scale_out_range[1])
        augmented_coords *= scale_factor
        num_points = augmented_coords.shape[0]
        num_to_drop = int(num_points * dropout_rate)
        if num_to_drop > 0 and num_points > num_to_drop:
            drop_indices = np.random.choice(num_points, num_to_drop, replace=False)
            augmented_coords = np.delete(augmented_coords, drop_indices, axis=0)
            final_intensity = np.delete(final_intensity, drop_indices, axis=0)
    elif aug_type == 'upsample':
        num_points = augmented_coords.shape[0]
        num_to_add = int(num_points * (upsample_rate - 1.0))
        if num_to_add > 0:
            upsample_indices = np.random.choice(num_points, num_to_add, replace=True)
            augmented_coords = np.vstack((augmented_coords, augmented_coords[upsample_indices]))
            final_intensity = np.vstack((final_intensity, final_intensity[upsample_indices]))

    return augmented_coords, final_intensity


def apply_augmentations_and_save(
    input_bin_path,
    output_dir,
    augmentation_suffix,
    translation_range=DEFAULT_TRANSLATION_RANGE,
    scaling_range=DEFAULT_SCALING_RANGE,
    scale_out_range=DEFAULT_SCALE_OUT_RANGE,
    jitter_sigma=DEFAULT_JITTER_SIGMA,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    upsample_rate=DEFAULT_UPSAMPLE_RATE,
    augmentations_to_choose=AVAILABLE_AUGMENTATIONS,
    combination_probability=DEFAULT_COMBINATION_PROBABILITY
):
    try:
        raw_points = np.fromfile(input_bin_path, dtype=np.float32)
        if raw_points.size == 0:
             logging.warning(f"Input file is empty: {input_bin_path}. Skipping.")
             return None
        try:
            original_points_with_intensity = raw_points.reshape(-1, 4)
        except ValueError:
            logging.error(f"Cannot reshape raw data from {input_bin_path} to Nx4. Size: {raw_points.size}. Skipping.")
            return None

        coords = original_points_with_intensity[:, :3]
        intensity = original_points_with_intensity[:, 3:]

        augmented_coords = coords.copy()
        final_intensity = intensity.copy()
        applied_augmentations_list = []

        combine_augs = random.random() < combination_probability

        if combine_augs and len(augmentations_to_choose) >= 2:
            num_to_combine = 2
            chosen_types = random.sample(augmentations_to_choose, num_to_combine)
            applied_augmentations_list = chosen_types

            for aug_type in chosen_types:
                augmented_coords, final_intensity = _apply_single_augmentation(
                    aug_type, augmented_coords, final_intensity,
                    translation_range, scaling_range, scale_out_range,
                    jitter_sigma, dropout_rate, upsample_rate
                )
        else:
            chosen_single_type = random.choice(augmentations_to_choose)
            applied_augmentations_list = [chosen_single_type]

            augmented_coords, final_intensity = _apply_single_augmentation(
                chosen_single_type, augmented_coords, final_intensity,
                translation_range, scaling_range, scale_out_range,
                jitter_sigma, dropout_rate, upsample_rate
            )

        augmented_points_with_intensity = np.hstack((augmented_coords, final_intensity))

        base_filename = os.path.basename(input_bin_path)
        name, ext = os.path.splitext(base_filename)
        if not ext: ext = '.bin'
        output_filename = f"{name}{augmentation_suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        os.makedirs(output_dir, exist_ok=True)

        augmented_points_with_intensity.astype(np.float32).tofile(output_path)
        logging.debug(f"Saved augmented cloud ({applied_augmentations_list}) to: {output_path}")
        return output_filename

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_bin_path}")
        return None


def run_offline_augmentation(
    input_dir,
    output_dir,
    num_augmentations_per_file,
    translation_range=DEFAULT_TRANSLATION_RANGE,
    scaling_range=DEFAULT_SCALING_RANGE,
    scale_out_range=DEFAULT_SCALE_OUT_RANGE,
    jitter_sigma=DEFAULT_JITTER_SIGMA,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    upsample_rate=DEFAULT_UPSAMPLE_RATE,
    max_points_threshold=float('inf'),
    combination_probability=DEFAULT_COMBINATION_PROBABILITY
):
    logging.info(f"Starting offline augmentation.")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Augmentations per file: {num_augmentations_per_file}")
    log_params = (
        f"Trans={translation_range}, Scale={scaling_range}, ScaleOut={scale_out_range}, "
        f"Jitter={jitter_sigma}, Drop={dropout_rate}, MaxPoints={max_points_threshold}, "
        f"Upsample={upsample_rate}, "
        f"CombineProb={combination_probability}"
    )
    logging.info(f"Using parameters: {log_params}")

    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    total_augmentations_created = 0

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.bin'):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                files_to_process.append((full_path, relative_path))

    if not files_to_process:
        logging.warning(f"No .bin files found in {input_dir}.")
        return

    logging.info(f"Found {len(files_to_process)} .bin files. Processing...")

    for input_path, relative_path in files_to_process:
        output_dir_relative = os.path.dirname(relative_path)
        output_abs_dir = os.path.join(output_dir, output_dir_relative)

        try:
            raw_points = np.fromfile(input_path, dtype=np.float32)
            if raw_points.size == 0:
                logging.warning(f"Skipping empty file: {input_path}")
                continue

            num_points = raw_points.size // 4
            if num_points >= max_points_threshold:
                logging.info(f"Skipping {input_path} (Points: {num_points} >= Threshold: {max_points_threshold})")
                continue
        except FileNotFoundError:
            logging.error(f"Input file not found during point count check: {input_path}")
            continue
        except Exception as e:
            logging.error(f"Error reading file {input_path} for point count: {e}")
            continue

        os.makedirs(output_abs_dir, exist_ok=True)

        augmentations_created_for_file = 0
        for i in range(num_augmentations_per_file):
            augmentation_suffix = f"_aug{i+1}"
            created_filename = apply_augmentations_and_save(
                input_bin_path=input_path,
                output_dir=output_abs_dir,
                augmentation_suffix=augmentation_suffix,
                translation_range=translation_range,
                scaling_range=scaling_range,
                scale_out_range=scale_out_range,
                jitter_sigma=jitter_sigma,
                dropout_rate=dropout_rate,
                upsample_rate=upsample_rate,
                augmentations_to_choose=AVAILABLE_AUGMENTATIONS,
                combination_probability=combination_probability
            )

            if created_filename:
                augmentations_created_for_file += 1

        total_augmentations_created += augmentations_created_for_file
        processed_count += 1
        if processed_count % 100 == 0:
             logging.info(f"Processed {processed_count}/{len(files_to_process)} files... ({total_augmentations_created} augmentations created)")

    logging.info("Offline augmentation complete.")
    logging.info(f"Total augmentations created: {total_augmentations_created}")
    logging.info(f"Augmented data saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform offline point cloud augmentation by applying random transformations (rotation, translation, scaling, flip, jitter, dropout, scale_out+dropout, upsample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to the input directory containing .bin files (searched recursively)."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to the output directory where augmented .bin files will be saved."
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=5,
        help="Number of augmented versions to create per original file. Each version gets ONE random transformation."
    )
    parser.add_argument(
        "--translation_range",
        type=float,
        default=DEFAULT_TRANSLATION_RANGE,
        help="Max distance for random translation (+/- meters)."
    )
    parser.add_argument(
        "--scaling_min",
        type=float,
        default=DEFAULT_SCALING_RANGE[0],
        help="Minimum general scaling factor."
    )
    parser.add_argument(
        "--scaling_max",
        type=float,
        default=DEFAULT_SCALING_RANGE[1],
        help="Maximum general scaling factor."
    )
    parser.add_argument(
        "--scale_out_min",
        type=float,
        default=DEFAULT_SCALE_OUT_RANGE[0],
        help="Minimum scaling factor (>1) for the 'scale_out_dropout' augmentation."
    )
    parser.add_argument(
        "--scale_out_max",
        type=float,
        default=DEFAULT_SCALE_OUT_RANGE[1],
        help="Maximum scaling factor (>1) for the 'scale_out_dropout' augmentation."
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=DEFAULT_JITTER_SIGMA,
        help="Standard deviation of Gaussian noise for jittering."
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Fraction of points to randomly drop for dropout augmentation (0.0 to 1.0)."
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=int(1e9),
        help="Maximum number of points allowed in an original file for it to be augmented. Files with >= points will be skipped."
    )
    parser.add_argument(
        "--combination_probability",
        type=float,
        default=DEFAULT_COMBINATION_PROBABILITY,
        help="Probability (0.0 to 1.0) of applying a combination of two distinct augmentations instead of just one."
    )
    parser.add_argument(
        "--upsample_rate",
        type=float,
        default=DEFAULT_UPSAMPLE_RATE,
        help="Target factor for random upsampling by duplicating points (e.g., 1.2 = aim for 20% more points)."
    )

    args = parser.parse_args()

    scaling_range_arg = [args.scaling_min, args.scaling_max]
    scale_out_range_arg = [args.scale_out_min, args.scale_out_max]

    run_offline_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations_per_file=args.num_augmentations,
        translation_range=args.translation_range,
        scaling_range=scaling_range_arg,
        scale_out_range=scale_out_range_arg,
        jitter_sigma=args.jitter_sigma,
        dropout_rate=args.dropout_rate,
        upsample_rate=args.upsample_rate,
        max_points_threshold=args.max_points,
        combination_probability=args.combination_probability
    )

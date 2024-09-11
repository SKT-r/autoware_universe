#!/bin/bash
set -eux
# mv ~/train_data/200~400_changed_pose.tsv ~/train_data/cams_meta.tsv
# ./localization/nerf_based_localizer/training_tool/script/build_and_exec_training.sh ~/result_200-400_pose_changed ~/train_data
# mv ~/train_data/400~600_changed_pose.tsv ~/train_data/cams_meta.tsv
# ./localization/nerf_based_localizer/training_tool/script/build_and_exec_training.sh ~/result_400-600_pose_changed ~/train_data
# mv ~/train_data/600~800_changed_pose.tsv ~/train_data/cams_meta.tsv
./localization/nerf_based_localizer/training_tool/script/build_and_exec_training.sh ~/result_600-800_pose_changed ~/train_data
mv ~/train_data/800~1000_changed_pose.tsv ~/train_data/cams_meta.tsv
./localization/nerf_based_localizer/training_tool/script/build_and_exec_training.sh ~/result_800-1000_pose_changed ~/train_data

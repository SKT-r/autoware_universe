// Copyright 2023 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "../../src/nerf/localizer.hpp"
#include "../../src/nerf/stop_watch.hpp"
#include "../../src/nerf/utils.hpp"
#include "main_functions.hpp"

#include <experimental/filesystem>
#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>

namespace fs = std::experimental::filesystem::v1;

void test(const std::string & train_result_dir, const std::string & dataset_dir)
{
  torch::NoGradGuard no_grad_guard;
  LocalizerParam param;
  param.train_result_dir = train_result_dir;
  param.resize_factor = 8;
  Localizer localizer(param);

  Dataset dataset(dataset_dir);
  const std::string save_dir = train_result_dir + "/test_result/";
  fs::create_directories(save_dir);

  Timer timer;

  float score_sum = 0.0f;
  float time_sum = 0.0f;

  const int grid_size = 3; // グリッドのサイズを指定
  // const float grid_spacing = 1.0f / (grid_size - 1); // 各グリッドポイントの間隔を計算
  const float grid_spacing = 0.05; 

  // 最初位置全然変わらんから50番目ぐらいからスタートして5枚ずつ進めていく
  for (int32_t i = 0; i < dataset.n_images; i=i+100) {
    torch::Tensor initial_pose = dataset.poses[i];
    torch::Tensor image_tensor = dataset.images[i];

    printf("どうなってる？\n");
    std::cout << "\rinitial_pose_x:" << initial_pose[0][3] << "\rinitial_pose_y:" << initial_pose[1][3]  << std::flush;

    std::cout << "initial_pose elements:" << std::endl;
    for (int64_t j = 0; j < initial_pose.size(0); ++j) {
      for (int64_t k = 0; k < initial_pose.size(1); ++k) {
        std::cout << initial_pose[j][k].item<float>() << " ";
      }
      std::cout << std::endl;
    }
    printf("\n");

    image_tensor = utils::resize_image(image_tensor, localizer.infer_height(), localizer.infer_width());

    std::ofstream csv_file(save_dir + std::to_string(i) + ".csv");
    csv_file << "pose[0][0], pose[0][1], pose[0][2], pose_x, pose[1][0], pose[1][1], pose[1][2], pose_y, pose[2][0], pose[2][1], pose[2][2], pose_z, score\n";


    for (int gx = 0; gx < 1; gx++) {
      for (int gy = 0; gy < 1; gy++) {
        // グリッド上の各ポイントの座標を計算
        // float x_offset = (gx-grid_size/2) * grid_spacing;
        // float y_offset = (gy-grid_size/2) * grid_spacing;
        float x_offset = (gx-grid_size/2) * 0.25f / dataset.radius;
        float y_offset = (gy-grid_size/2) * 0.25f / dataset.radius;
        printf("各項: %d %d %d %f %f\n", gx, gy, grid_size/2, grid_spacing, dataset.radius);
        
        torch::Tensor grid_pose = initial_pose.clone();
        if (grid_pose.sizes().size() >= 2 && grid_pose.size(0) > 0 && grid_pose.size(1) >= 2) {
          //ここだけ後で合わせる
          // grid_pose[1][1] += x_offset;
          // grid_pose[2][2] += y_offset;
          grid_pose[0][3] += x_offset;
          grid_pose[1][3] += y_offset;
        } else {
          std::cerr << "Error: grid_pose has an unexpected shape." << std::endl;
          continue;
        }
        std::cout << "\rgrid_pose_x:" << grid_pose[0][3] << "\rgrid_pose_y:" << grid_pose[1][3] << "\rinitial_pose_x:" << initial_pose[0][3] << "\rinitial_pose_y:" << initial_pose[1][3]  << std::flush;

        std::cout << "\rgx:" << gx << ", gy:" << gy << ", x_off" << x_offset << ", y_off" << y_offset  << std::flush;
        printf("\n");

        std::cout << "grid_pose elements:" << std::endl;
        for (int64_t j = 0; j < grid_pose.size(0); ++j) {
          for (int64_t k = 0; k < grid_pose.size(1); ++k) {
            std::cout << grid_pose[j][k].item<float>() << " ";
          }
          std::cout << std::endl;
        }
        printf("\n");


        timer.start();
        torch::Tensor nerf_image = localizer.render_image(grid_pose).cpu();
        time_sum += timer.elapsed_seconds();
        torch::Tensor diff = (nerf_image - image_tensor).abs();
        torch::Tensor loss = (diff * diff).mean(-1).sum();
        torch::Tensor score = (localizer.infer_height() * localizer.infer_width()) / (loss + 1e-6f);

        std::cout << "\rscore[" << i << "][" << gx << "][" << gy << "] = " << score.item<float>() << "\n" << std::flush;
        score_sum += score.item<float>();


        csv_file << 
        grid_pose[0][0].item<float>() << "," << grid_pose[0][1].item<float>() << "," << grid_pose[0][2].item<float>() << "," << grid_pose[0][3].item<float>() << "," <<
        grid_pose[1][0].item<float>() << "," << grid_pose[1][1].item<float>() << "," << grid_pose[1][2].item<float>() << "," << grid_pose[1][3].item<float>() << "," <<
        grid_pose[2][0].item<float>() << "," << grid_pose[2][1].item<float>() << "," << grid_pose[2][2].item<float>() << "," << grid_pose[2][3].item<float>() << "," << score.item<float>() << "\n";
        std::stringstream ss;
        ss << save_dir << std::setfill('0') << std::setw(8) << i << "_" << gx << "_" << gy << ".png";
        utils::write_image_tensor(ss.str(), nerf_image);

        std::stringstream ssr;
        ssr << save_dir << std::setfill('0') << std::setw(8) << i << "diff" << ".png";
        utils::write_image_tensor(ssr.str(), diff.flip(2));
      }
    }
    std::stringstream sst;
    sst << save_dir << i << "_resize" << ".png";
    utils::write_image_tensor(sst.str(), image_tensor);

    csv_file.close();
  }

  const float average_time = time_sum / dataset.n_images;
  const float average_score = score_sum / dataset.n_images;

  std::ofstream summary(train_result_dir + "/summary.tsv");
  summary << std::fixed;
  summary << "average_time\taverage_score" << std::endl;
  summary << average_time << "\t" << average_score << std::endl;
  std::cout << "\ntime = " << average_time << ", score = " << average_score << std::endl;
}

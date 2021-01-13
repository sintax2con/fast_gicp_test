#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pclomp/ndt_omp.h>

#include <chrono>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fstream>
#include <iostream>
#include <pclomp/ndt_omp_impl.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

Eigen::Isometry3f transPrev;

/**
 * @brief rotation matrix to euler angle
 */
template <typename T>
static void matrix2euler(Eigen::Matrix<T, 3, 3>& rot_matrix, Eigen::Matrix<T, 3, 1>& euler) {
    Eigen::Quaternion<T> q(rot_matrix);
    // euler = rot_matrix.eulerAngles(2, 1, 0);
    double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    euler[2] = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
    if (fabs(sinp) >= 1)
        euler[1] = copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
    else
        euler[1] = asin(sinp);
    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    euler[0] = atan2(siny_cosp, cosy_cosp);
}
// benchmark for PCL's registration methods
template <typename Registration>
pcl::PointCloud<pcl::PointXYZ>::Ptr test_pcl(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                                             const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                                             const Eigen::Isometry3f true_pose, double& dis_error,
                                             double& angle_error) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

    double fitness_score = 0.0;

    reg.setInputTarget(target);
    // voxelized
    pcl::PointXYZ dummyPoint;
    pcl::PointCloud<pcl::PointXYZ>::Ptr dummyCloud(new pcl::PointCloud<pcl::PointXYZ>());
    dummyCloud->points.push_back(dummyPoint);
    reg.setMaximumIterations(1);
    reg.setInputSource(dummyCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr dummyAlignedCloud(new pcl::PointCloud<pcl::PointXYZ>());
    reg.align(*dummyAlignedCloud);

    // single run
    auto t1 = std::chrono::high_resolution_clock::now();
    reg.setInputSource(source);
    reg.align(*aligned);
    auto t2 = std::chrono::high_resolution_clock::now();
    fitness_score = reg.getFitnessScore();
    double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    std::cout << "single:" << single << "[msec] " << std::flush;

    Eigen::Matrix4f final_trans = reg.getFinalTransformation();
    Eigen::Matrix3f final_trans_ori = final_trans.block<3, 3>(0, 0);
    Eigen::Vector3f true_position = true_pose.translation();
    Eigen::Matrix3f true_ori = true_pose.rotation();
    Eigen::Vector3f final_euler;
    Eigen::Vector3f true_euler;
    matrix2euler(final_trans_ori, final_euler);
    matrix2euler(true_ori, true_euler);
    // 10 times
    // t1 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 10; i++) {
    //     // reg.setInputTarget(target);
    //     // reg.setInputSource(source);
    //     reg.align(*aligned);
    // }
    // t2 = std::chrono::high_resolution_clock::now();
    double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    double error = std::sqrt((final_trans(0, 3) - true_position[0]) * (final_trans(0, 3) - true_position[0]) +
                             (final_trans(1, 3) - true_position[1]) * (final_trans(1, 3) - true_position[1]) +
                             (final_trans(2, 3) - true_position[2]) * (final_trans(2, 3) - true_position[2]));
    dis_error = error;
    angle_error = std::abs(true_euler[0] - final_euler[0]) / M_PI * 180.0;
    std::cout << "fitness_score:" << fitness_score << std::endl;
    // std::cout << "10times:" << multi << "[msec] fitness_score:" << fitness_score << std::endl;
    std::cout << "result: " << final_trans(0, 3) << " " << final_trans(1, 3) << " " << final_trans(2, 3) << std::endl;
    std::cout << "dis error: " << error << std::endl;
    std::cout << "angle error: " << angle_error << std::endl;
    return aligned;
}

// benchmark for fast_gicp registration methods
template <typename Registration>
pcl::PointCloud<pcl::PointXYZ>::Ptr test(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                                         const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                                         const Eigen::Isometry3f true_pose, double& dis_error, double& angle_error) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

    double fitness_score = 0.0;

    // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
    // to prevent this for benchmarking, force clear source and target clouds
    reg.clearTarget();
    reg.clearSource();
    reg.setInputTarget(target);

    // voxelized
    pcl::PointXYZ dummyPoint;
    pcl::PointCloud<pcl::PointXYZ>::Ptr dummyCloud(new pcl::PointCloud<pcl::PointXYZ>());
    dummyCloud->points.push_back(dummyPoint);
    reg.setMaximumIterations(1);
    reg.setInputSource(dummyCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr dummyAlignedCloud(new pcl::PointCloud<pcl::PointXYZ>());
    reg.align(*dummyAlignedCloud);

    // single run
    auto t1 = std::chrono::high_resolution_clock::now();
    reg.setInputSource(source);
    reg.align(*aligned);
    auto t2 = std::chrono::high_resolution_clock::now();
    fitness_score = reg.getFitnessScore();
    double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

    std::cout << "single:" << single << "[msec] " << std::flush;
    Eigen::Matrix4f final_trans = reg.getFinalTransformation();
    Eigen::Matrix3f final_trans_ori = final_trans.block<3, 3>(0, 0);
    Eigen::Vector3f true_position = true_pose.translation();
    Eigen::Matrix3f true_ori = true_pose.rotation();
    Eigen::Vector3f final_euler;
    Eigen::Vector3f true_euler;
    matrix2euler(final_trans_ori, final_euler);
    matrix2euler(true_ori, true_euler);
    // 10 times
    // t1 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 10; i++) {
    //     // reg.setInputTarget(target);
    //     // reg.setInputSource(source);
    //     reg.align(*aligned);
    // }
    // t2 = std::chrono::high_resolution_clock::now();
    // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    double error = std::sqrt((final_trans(0, 3) - true_position[0]) * (final_trans(0, 3) - true_position[0]) +
                             (final_trans(1, 3) - true_position[1]) * (final_trans(1, 3) - true_position[1]) +
                             (final_trans(2, 3) - true_position[2]) * (final_trans(2, 3) - true_position[2]));
    dis_error = error;
    angle_error = std::abs(true_euler[0] - final_euler[0]) / M_PI * 180.0;
    // std::cout << "10times:" << multi << "[msec] fitness_score:" << fitness_score << std::endl;
    std::cout << "fitness_score:" << fitness_score << std::endl;
    std::cout << "result: " << final_trans(0, 3) << " " << final_trans(1, 3) << " " << final_trans(2, 3) << std::endl;
    std::cout << "dis error: " << error << std::endl;
    std::cout << "angle error: " << angle_error << std::endl;

    return aligned;
    // for some tasks like odometry calculation,
    // you can reuse the covariances of a source point cloud in the next registration
    // t1 = std::chrono::high_resolution_clock::now();
    // pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_ = target;
    // pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_ = source;
    // for (int i = 0; i < 10; i++) {
    //     reg.swapSourceAndTarget();
    //     reg.clearSource();

    //     reg.setInputTarget(target_);
    //     reg.setInputSource(source_);
    //     reg.align(*aligned);

    //     target_.swap(source_);
    // }
    // t2 = std::chrono::high_resolution_clock::now();
    // double reuse = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

    // std::cout << "10times_reuse:" << reuse << "[msec] fitness_score:" << fitness_score << std::endl;
}

/**
 * @brief main
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: gicp_align target_pcd source_pcd" << std::endl;
        return 0;
    }
    
    std::ofstream out_file;
    out_file.open("./compare.txt");
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_trans(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_trans(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_ndt_omp(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_fgicp_omp(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_vfgicp_omp(new pcl::PointCloud<pcl::PointXYZ>());
    /// target_cloud and source_cloud are the same
    if (pcl::io::loadPCDFile(argv[1], *target_cloud)) {
        std::cerr << "failed to open " << argv[1] << std::endl;
        return 1;
    }
    if (pcl::io::loadPCDFile(argv[2], *source_cloud)) {
        std::cerr << "failed to open " << argv[2] << std::endl;
        return 1;
    }

    // remove invalid points around origin
    source_cloud->erase(
        std::remove_if(source_cloud->begin(), source_cloud->end(),
                       [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
        source_cloud->end());
    target_cloud->erase(
        std::remove_if(target_cloud->begin(), target_cloud->end(),
                       [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
        target_cloud->end());

    // downsampling
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*filtered);
    target_cloud = filtered;

    filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud = filtered;

    double dis_error = 0, angle_error = 0;
    double max_dis_error = 0, max_ang_error = 0;
    double avg_dis_error = 0, avg_ang_error = 0;

    std::vector<double> dis_increment{-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2};
    std::vector<double> angle_increment{-3, -2.5, -2, -1, -0., 1.0, 1.5, 2.0, 2.5, 3.0};
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gficp_omp;
    gficp_omp.setMaximumIterations(20);
    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgic_omp;
    vgic_omp.setMaximumIterations(20);
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp;
    ndt_omp.setMaximumIterations(20);
    std::vector<int> num_threads = {1, 8};
    std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> search_methods = {
        {"KDTREE", pclomp::KDTREE}, {"DIRECT7", pclomp::DIRECT7}, {"DIRECT1", pclomp::DIRECT1}};

    for (auto iter_dis = dis_increment.begin(); iter_dis != dis_increment.end(); iter_dis++) {
        for (auto iter_ang = angle_increment.begin(); iter_ang != angle_increment.end(); iter_ang++) {
            Eigen::Vector3f tPrev(*iter_dis, 0.0, 0);
            Eigen::AngleAxisf roll(Eigen::AngleAxisf(0.0 / 180.0 * M_PI, Eigen::Vector3f::UnitX()));
            Eigen::AngleAxisf pitch(Eigen::AngleAxisf(0.0 / 180.0 * M_PI, Eigen::Vector3f::UnitY()));
            Eigen::AngleAxisf yaw(Eigen::AngleAxisf(*iter_ang / 180.0 * M_PI, Eigen::Vector3f::UnitZ()));
            Eigen::Quaternionf qPrev;
            qPrev = yaw * pitch * roll;
            transPrev = Eigen::Isometry3f::Identity();
            transPrev.pretranslate(tPrev);
            transPrev.prerotate(qPrev);
            pcl::transformPointCloud(*target_cloud, *target_cloud_trans, transPrev);
            out_file << *iter_dis << "  " << *iter_ang << "  ";
            std::cout << std::setw(6) << std::left << *iter_dis << "  " << *iter_ang << std::endl;
            std::cout << "--- gficp_omp ---" << std::endl;
            gficp_omp.setNumThreads(8);
            std::cout << "cloud size: " << target_cloud_trans->points.size() << "   " << source_cloud->points.size()
                      << std::endl;
            aligned_fgicp_omp = test(gficp_omp, target_cloud_trans, source_cloud, transPrev, dis_error, angle_error);
            out_file << dis_error << "  " << angle_error << "  ";

            std::cout << "--- vgic_omp ---" << std::endl;
            vgic_omp.setNumThreads(8);
            aligned_vfgicp_omp = test(vgic_omp, target_cloud_trans, source_cloud, transPrev, dis_error, angle_error);
            out_file << dis_error << "  " << angle_error << "  ";

            //   ndt_omp.setNumThreads(8);
            // for (int n : num_threads) {
            for (const auto& search_method : search_methods) {
                std::cout << "--- ndt_omp (" << search_method.first << ", "
                          << "8"
                          << " threads) ---" << std::endl;
                ndt_omp.setNumThreads(8);
                ndt_omp.setNeighborhoodSearchMethod(search_method.second);
                aligned_ndt_omp =
                    test_pcl(ndt_omp, target_cloud_trans, source_cloud, transPrev, dis_error, angle_error);
                out_file << dis_error << "  " << angle_error << "  ";
            }
            out_file << std::endl;
            // }
        }
    }

    // std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" <<
    // std::endl;

    // std::cout << "--- pcl_gicp ---" << std::endl;
    // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> pcl_gicp;
    // aligned = test_pcl(pcl_gicp, target_cloud, source_cloud);

    // std::cout << "--- pcl_ndt ---" << std::endl;
    // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> pcl_ndt;
    // pcl_ndt.setResolution(1.0);
    // aligned = test_pcl(pcl_ndt, target_cloud, source_cloud);

    // std::cout << "--- fgicp_st ---" << std::endl;
    // fast_gicp::FastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ> fgicp_st;
    // aligned = test(fgicp_st, target_cloud, source_cloud);

    // std::cout << "--- gficp_omp ---" << std::endl;
    // fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gficp_omp;
    // // fast_gicp uses all the CPU cores by default
    // // gficp_omp.setNumThreads(8);
    // gficp_omp.setNumThreads(8);
    // aligned_fgicp_omp = test(gficp_omp, target_cloud, source_cloud);

    // std::cout << "--- vgicp_omp_st ---" << std::endl;
    // fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
    // vgicp.setResolution(1.0);
    // vgicp.setNumThreads(1);
    // aligned = test(vgicp, target_cloud, source_cloud);

    // std::cout << "--- vgicp_mt ---" << std::endl;
    // vgicp.setNumThreads(8);
    // aligned_vfgicp_omp = test(vgicp, target_cloud, source_cloud);

    // std::vector<int> num_threads = {1, 8};
    // std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> search_methods = {
    //     {"KDTREE", pclomp::KDTREE}, {"DIRECT7", pclomp::DIRECT7}, {"DIRECT1", pclomp::DIRECT1}};

    // pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp;
    // //   ndt_omp.setNumThreads(8);
    // for (int n : num_threads) {
    //     for (const auto& search_method : search_methods) {
    //         std::cout << "--- ndt_omp (" << search_method.first << ", " << n << " threads) ---" << std::endl;
    //         ndt_omp.setNumThreads(n);
    //         ndt_omp.setNeighborhoodSearchMethod(search_method.second);
    //         aligned_ndt_omp = test_pcl(ndt_omp, target_cloud, source_cloud);
    //     }
    // }

    // visulization
    pcl::visualization::PCLVisualizer vis("vis");
    vis.setBackgroundColor(255, 255, 255);
    vis.addCoordinateSystem(1.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 0.0, 255.0, 0.0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_ndt_omp_handler(aligned_ndt_omp, 0.0, 255.0,
                                                                                            0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_fgicp_omp_handler(aligned_fgicp_omp, 205.0,
                                                                                              205.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_vfgicp_omp_handler(aligned_vfgicp_omp,
                                                                                               100.0, 149.0, 237.0);
    vis.addPointCloud(target_cloud, target_handler, "target");
    // vis.addPointCloud(source_cloud, source_handler, "source");
    vis.addPointCloud(aligned_ndt_omp, aligned_ndt_omp_handler, "aligned_ndt_omp");
    vis.addPointCloud(aligned_fgicp_omp, aligned_fgicp_omp_handler, "aligned_fgicp_omp");
    vis.addPointCloud(aligned_vfgicp_omp, aligned_vfgicp_omp_handler, "aligned_vfgicp_omp");
    vis.spin();

#ifdef USE_VGICP_CUDA
    std::cout << "--- vgicp_cuda (parallel_kdtree) ---" << std::endl;
    fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;
    vgicp_cuda.setResolution(1.0);
    // vgicp_cuda uses CPU-based parallel KDTree in covariance estimation by default
    // on a modern CPU, it is faster than GPU_BRUTEFORCE
    // vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
    test(vgicp_cuda, target_cloud, source_cloud);

    std::cout << "--- vgicp_cuda (gpu_bruteforce) ---" << std::endl;
    // use GPU-based bruteforce nearest neighbor search for covariance estimation
    // this would be a good choice if your PC has a weak CPU and a strong GPU (e.g., NVIDIA Jetson)
    vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
    test(vgicp_cuda, target_cloud, source_cloud);

    std::cout << "--- vgicp_cuda (gpu_rbf_kernel) ---" << std::endl;
    // use RBF-kernel-based covariance estimation
    // extremely fast but maybe a bit inaccurate
    vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
    // kernel width (and distance threshold) need to be tuned
    vgicp_cuda.setKernelWidth(0.5);
    test(vgicp_cuda, target_cloud, source_cloud);
#endif

    return 0;
}

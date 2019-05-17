#include "stdafx.h"

#include <math.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "VisualizerData.h"

#define VISUALIZER(x) x

using CloudType = pcl::PointCloud<pcl::PointXYZ>;

float randf() { return rand() / static_cast<float>(RAND_MAX); }

void testCorrelationScore()
{
    auto makeCloud = [&](float scale)
    {
        CloudType::Ptr cloud(new CloudType());

        for (float x = 0.0; x < 1.0; x += 0.02)
        {
            for (float y = 0.0; y < 1.0; y += 0.02)
            {
                // The offset is to make sure to have consistent normals directions.
                const float z = 3.0 + scale * std::sin(x*M_PI*1.0) * std::sin(y*M_PI*1.0);
                cloud->push_back({ 
                    x + 0.03f * randf(),
                    y + 0.03f * randf(),
                    z + 0.03f * randf()});
            }
        }

        return cloud;
    };

    const auto source = makeCloud(0.3);
    const auto target = makeCloud(0.3);

    const int N = source->size();

    pcl::Correspondences correspondences;
    correspondences.reserve(N);

    for (auto i = 0; i < N; ++i)
        correspondences.emplace_back(i, i, 0);

    VISUALIZER(pcv::VisualizerData viewer("correlation-score"));
    VISUALIZER(viewer.addCloud(*source, "source").setColor(0.4, 0.4, 1.0).setOpacity(0.5).setSize(3));
    VISUALIZER(viewer.addCloud(*target, "target").setColor(1.0, 0.0, 0.0).setOpacity(0.5).setSize(3));
    VISUALIZER(viewer.addCorrespondences(*source, *target, correspondences, "correspondences").setOpacity(0.2));

    auto computeCorrelation = [&](int idx)
    {
        float x{ 0 };
        float y{ 0 };
        float xx{ 0 };
        float xy{ 0 };
        float yy{ 0 };

        for (const auto& c : correspondences)
        {
            const auto s = source->at(c.index_query).getVector3fMap()[idx];
            const auto t = target->at(c.index_match).getVector3fMap()[idx];
            
            x += s;
            y += t;
            xx += s * s;
            yy += t * t;
            xy += s * t;
        }

        return (N*xy - x * y) / (std::sqrt(N*xx - x * x) * std::sqrt(N*yy - y * y));
    };

    std::cout << "correlation x: " << computeCorrelation(0) << std::endl;
    std::cout << "correlation y: " << computeCorrelation(1) << std::endl;
    std::cout << "correlation z: " << computeCorrelation(2) << std::endl;

    //// Compute correlation.
    //auto prods{ Eigen::Matrix3f::Zero() };
    //auto sumsS{ Eigen::Vector3f::Zero() };
    //auto sumsT{ Eigen::Vector3f::Zero() };
    //for (const auto& c : correspondences)
    //{
    //    const auto s = source->at(c.index_query).getVector3fMap();
    //    const auto t = target->at(c.index_match).getVector3fMap();

    //    prods += s * t.transpose();
    //    sumsS += s;
    //    sumsT += t;
    //}

    //Eigen::Matrix4f correlation{ Eigen::Matrix4f::Zero() };
}

void testSomething()
{
    std::string path = "../data/";

    CloudType::Ptr model(new CloudType);
    pcl::io::loadPCDFile(path + "model.pcd", *model);
}

int main(int argc, char* argv[])
{
    testCorrelationScore();

    return 0;
}


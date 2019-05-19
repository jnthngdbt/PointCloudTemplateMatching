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
    auto makeCloud = [&](float scale, float freq)
    {
        CloudType::Ptr cloud(new CloudType());

        for (float x = 0.0; x < 1.0; x += 0.02)
        {
            for (float y = 0.0; y < 1.0; y += 0.02)
            {
                // The offset is to make sure to have consistent normals directions.
                const float z = 3.0 + scale * std::sin(x*M_PI*freq) * std::sin(y*M_PI*freq);
                cloud->push_back({ 
                    x + 0.03f * randf(),
                    y + 0.03f * randf(),
                    z + 0.03f * randf()});
            }
        }

        return cloud;
    };

    const auto source = makeCloud(0.3, 1.0);
    const auto target = makeCloud(0.3, 1.1);

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
}

void testCubeWarp()
{
    CloudType::Ptr cube(new CloudType());

    int idx = 0;
    std::vector<int> idxFeature;
    const float gridSize = 0.2;
    for (float x = 0.0; x <= 1.0; x += gridSize)
        for (float y = 0.0; y <= 1.0; y += gridSize)
            for (float z = 0.0; z <= 1.0; z += gridSize)
                if ((x == 0 || x == 1) || (y == 0) || (z == 0 || z == 1))
                {
                    cube->push_back({ x,y,z });
                    idxFeature.emplace_back(idx++);
                }

    VISUALIZER(pcv::VisualizerData viewer("cube-warp"));
    VISUALIZER(viewer.addCloud(*cube, "cube", 0).addFeature(idxFeature, "idx").setColor(0.4, 0.4, 1.0).setOpacity(1.0).setSize(8));

    Eigen::Matrix4f transf; // converted to affine at transform
    transf.setIdentity();
    transf(3) = randf();
    transf(7) = randf();
    transf(11) = -randf();
    //transf.setRandom();
    //for (int i = 0; i < 16; ++i) transf(i) = randf();

    std::cout << transf << std::endl;

    CloudType::Ptr warp(new CloudType(*cube));
    //pcl::transformPointCloud(*cube, *warp, proj); // does not accept projective

    for (int i = 0; i < cube->size(); ++i)
    {
        const auto& p = cube->at(i).getVector4fMap();
        auto q = transf * p;
        warp->at(i) = pcl::PointXYZ(q[0], q[1], q[2]);
        warp->at(i).x /= q[3];
        warp->at(i).y /= q[3];
        warp->at(i).z /= q[3];
    }

    VISUALIZER(viewer.addCloud(*warp, "warp", 1).addFeature(idxFeature, "idx").setColor(1.0, 0.4, 0.4).setOpacity(1.0).setSize(8));


}

void testSomething()
{
    std::string path = "../data/";

    CloudType::Ptr model(new CloudType);
    pcl::io::loadPCDFile(path + "model.pcd", *model);
}

int main(int argc, char* argv[])
{
    srand(time(NULL)); // random seed, to have different random at each run

    //testCorrelationScore();
    testCubeWarp();

    return 0;
}


#include "stdafx.h"

#include <math.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "VisualizerData.h"

#define VISUALIZER_CALL(x) x

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

    VISUALIZER_CALL(pcv::VisualizerData viewer("correlation-score"));
    VISUALIZER_CALL(viewer.addCloud(*source, "source").setColor(0.4, 0.4, 1.0).setOpacity(0.5).setSize(3));
    VISUALIZER_CALL(viewer.addCloud(*target, "target").setColor(1.0, 0.0, 0.0).setOpacity(0.5).setSize(3));
    VISUALIZER_CALL(viewer.addCorrespondences(*source, *target, correspondences, "correspondences").setOpacity(0.2));

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

    VISUALIZER_CALL(pcv::VisualizerData viewer("cube-warp"));
    VISUALIZER_CALL(viewer.addCloud(*cube, "cube", 0).addFeature(idxFeature, "idx").setColor(0.4, 0.4, 1.0).setOpacity(1.0).setSize(8));

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

    VISUALIZER_CALL(viewer.addCloud(*warp, "warp", 1).addFeature(idxFeature, "idx").setColor(1.0, 0.4, 0.4).setOpacity(1.0).setSize(8));
}

void testCorrelationAxis()
{
    int N = 1000;
    double makeScale = 0.2;

    auto makeDomain = [&]()
    {
        std::vector<float> x(N, 0);
        for (int i = 0; i < N; ++i)
            x[i] = i * 1.0 / N - 0.5; // [-0.5, 0.5] 
        return x;
    };

    auto makeWave = [&](double offset, double scale, double freq, double phaseRad)
    {
        const auto x = makeDomain();
        std::vector<float> y(N, 0);
        for (int i = 0; i < N; ++i)
            y[i] = offset + makeScale * scale * std::cos(x[i] * M_PI * freq + phaseRad);
        return y;
    };

    auto makeBump = [&](double scale)
    {
        return makeWave(-0.5 * scale * makeScale, scale, 1, 0);
    };

    auto rotate = [&](std::vector<float>& x, std::vector<float>& y, double angleDeg)
    {
        const auto angleRad = angleDeg * M_PI / 180.0;
        const auto c = std::cos(angleRad);
        const auto s = std::sin(angleRad);

        std::vector<float> xin(x);
        std::vector<float> yin(y);

        for (int i = 0; i < N; ++i)
        {
            x[i] = xin[i] * c - yin[i] * s;
            y[i] = xin[i] * s + yin[i] * c;
        }
    };

    auto add = [&](const std::vector<float>& x, const std::vector<float>& y)
    {
        std::vector<float> r(N, 0);
        for (int i = 0; i < N; ++i)
            r[i] = x[i] + y[i];
        return r;
    };

    auto computeCorrelation = [&](const std::vector<float>& x, const std::vector<float>& y)
    {
        const int N = x.size();

        float Cx{ 0 };
        float Cy{ 0 };
        float Cxx{ 0 };
        float Cxy{ 0 };
        float Cyy{ 0 };

        for (int i = 0; i < N; ++i)
        {
            Cx += x[i];
            Cy += y[i];
            Cxx += x[i] * x[i];
            Cyy += y[i] * y[i];
            Cxy += x[i] * y[i];
        }

        return (N*Cxy - Cx * Cy) / (std::sqrt(N*Cxx - Cx * Cx) * std::sqrt(N*Cyy - Cy * Cy));
    };

    auto visualize = [&](
        const std::string name, 
        const std::vector<float>& x1, const std::vector<float>& y1,
        const std::vector<float>& x2, const std::vector<float>& y2)
    {
        auto getVal = [](float v) { return v; };

        VISUALIZER_CALL(pcv::VisualizerData viewer(name));
        VISUALIZER_CALL(viewer.addPlot(x1, y1, "x1-y1", getVal, getVal, 0).setSize(2).setColor(1,0,0));
        VISUALIZER_CALL(viewer.addPlot(x2, y2, "x2-y2", getVal, getVal, 0).setSize(2).setColor(0.4,0.4,1.0));
        VISUALIZER_CALL(viewer.addPlot(x1, x2, "x1-x2", getVal, getVal, 1).setSize(2).setColor(0,1,0));
        VISUALIZER_CALL(viewer.addPlot(y1, y2, "y1-y2", getVal, getVal, 2).setSize(2).setColor(0,1,0));
    };

    auto print = [&](
        const std::string name, 
        const std::vector<float>& x1, const std::vector<float>& y1,
        const std::vector<float>& x2, const std::vector<float>& y2)
    {
        std::cout << name << " correlation xx, yy: " << computeCorrelation(x1, x2) << ", " << computeCorrelation(y1, y2) << std::endl;
    };

    {
        const std::string name = "same-sine-scaled";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0);
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-scaled-offset";
        auto x1 = makeDomain();
        auto y1 = makeWave(0.3, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0);
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-scaled-ortho";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0.5 * M_PI); // orthogonal
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-otho-rotated";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0.5 * M_PI); // orthogonal
        const double angleStep = 10.0;
        for (double angle = 0.0; angle < 90.0; angle += angleStep)
        {
            rotate(x1, y1, angleStep);
            rotate(x2, y2, angleStep);
            print(name, x1, y1, x2, y2);
            visualize(name, x1, y1, x2, y2);
        }
    }

    {
        const std::string name = "same-sine-scaled-ortho-bump";
        auto x1 = makeDomain();
        auto y1 = add(makeBump(5), makeWave(0, 0.4, 10, 0));
        auto x2 = makeDomain();
        auto y2 = add(makeBump(5), makeWave(0, 0.8, 10, 0.5 * M_PI)); // orthogonal
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-scaled-phased";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0.1 * M_PI);
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-scaled";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0);
        print(name, x1, y1, x2, y2);
        visualize(name, x1, y1, x2, y2);
    }

    {
        const std::string name = "same-sine-scaled-rotated";
        auto x1 = makeDomain();
        auto y1 = makeWave(0, 1, 10, 0);
        auto x2 = makeDomain();
        auto y2 = makeWave(0, 2, 10, 0);
        const double angleStep = 10.0;
        for (double angle = 0.0; angle < 90.0; angle += angleStep)
        {
            rotate(x1, y1, angleStep);
            rotate(x2, y2, angleStep);
            print(name, x1, y1, x2, y2);
            visualize(name, x1, y1, x2, y2);
        }
    }
}

int main(int argc, char* argv[])
{
    VISUALIZER_CALL(pcv::VisualizerData::clearSavedData(0.5));

    srand(time(NULL)); // random seed, to have different random at each run

    //testCorrelationScore();
    //testCubeWarp();
    testCorrelationAxis();

    return 0;
}


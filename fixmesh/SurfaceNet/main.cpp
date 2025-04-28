#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <array>

#include "MMSurfaceNet.h"
#include "MMGeometryOBJ.h"

bool loadMeta(const std::string& metaFilename, int arraySize[3], float voxelSize[3]) {
    std::ifstream metaFile(metaFilename);
    if (!metaFile.is_open()) {
        std::cerr << "Failed to open metadata file: " << metaFilename << std::endl;
        return false;
    }

    metaFile >> arraySize[0] >> arraySize[1] >> arraySize[2];
    metaFile >> voxelSize[0] >> voxelSize[1] >> voxelSize[2];

    metaFile.close();
    return true;
}

unsigned short* loadRawLabels(const std::string& rawFilename, size_t numElements) {
    std::ifstream rawFile(rawFilename, std::ios::binary);
    if (!rawFile.is_open()) {
        std::cerr << "Failed to open raw label file: " << rawFilename << std::endl;
        return nullptr;
    }

    unsigned short* labels = new unsigned short[numElements];
    rawFile.read(reinterpret_cast<char*>(labels), sizeof(unsigned short) * numElements);
    rawFile.close();
    return labels;
}

void writeOBJ(const MMGeometryOBJ::OBJData& data, const std::string& filename) {
    std::ofstream obj(filename);
    if (!obj.is_open()) {
        std::cerr << "Failed to write OBJ file: " << filename << std::endl;
        return;
    }

    for (const auto& v : data.vertexPositions) {
        obj << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }

    for (const auto& f : data.triangles) {
        obj << "f " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }

    obj.close();
    std::cout << "Exported OBJ to: " << filename << std::endl;
}

void writeCombinedOBJ(const std::vector<MMGeometryOBJ::OBJData>& objDatas,
    const std::vector<int>& labels,
    const std::string& filename)
{
    std::ofstream objFile(filename);
    if (!objFile.is_open()) {
        std::cerr << "Failed to write combined OBJ file: " << filename << std::endl;
        return;
    }

    size_t vertexOffset = 0;

    for (size_t i = 0; i < objDatas.size(); ++i) {
        int label = labels[i];
        if (label == 0)           // skip background if you wish
            continue;

        const auto& data = objDatas[i];

        objFile << "g label_" << label << '\n';

        // write vertices
        for (const auto& v : data.vertexPositions)
        objFile << "v " << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';

        // write faces with **corrected** indices
        for (const auto& f : data.triangles)
        objFile << "f "
        << (f[0] + vertexOffset) << ' '
        << (f[1] + vertexOffset) << ' '
        << (f[2] + vertexOffset) << '\n';

        vertexOffset += data.vertexPositions.size();
    }

    std::cout << "Exported combined OBJ to: " << filename << std::endl;
}


int main() {
    std::vector<std::string> caseDirs = {
        "../../examples/SNResult/case1/",
        "../../examples/SNResult/case2/",
        "../../examples/SNResult/case3/"
    };

    for (const auto& caseDir : caseDirs) {
        std::cout << "Processing: " << caseDir << std::endl;

        std::string metaFilename = caseDir + "voxel_labels_meta.txt";
        std::string rawFilename = caseDir + "voxel_labels.raw";

        int arraySize[3];
        float voxelSize[3];
        if (!loadMeta(metaFilename, arraySize, voxelSize)) return 1;

        size_t numElements = static_cast<size_t>(arraySize[0]) * arraySize[1] * arraySize[2];
        unsigned short* labels = loadRawLabels(rawFilename, numElements);
        if (!labels) return 1;

        // Construct SurfaceNet
        MMSurfaceNet* surfaceNet = new MMSurfaceNet(labels, arraySize, voxelSize);

        MMSurfaceNet::RelaxAttrs relaxAttrs = {20, 0.1f, 1.0f};
        surfaceNet->relax(relaxAttrs);

        // Export as OBJ
        MMGeometryOBJ exporter(surfaceNet);
        std::vector<int> labelList = exporter.labels();

        // Export each label
        for (int label : labelList) {
            std::ostringstream filename;
            filename << caseDir << "label_" << label << ".obj";
            MMGeometryOBJ::OBJData objData = exporter.objData(label);
            writeOBJ(objData, filename.str());
        }

        // Export everything
        std::vector<MMGeometryOBJ::OBJData> allObjData;
        for (int label : labelList) {
            MMGeometryOBJ::OBJData objData = exporter.objData(label);
            allObjData.push_back(objData);
        }
        writeCombinedOBJ(allObjData, labelList, caseDir + "combined_labels.obj");

        delete surfaceNet;
        delete[] labels;

    }

    return 0;
}

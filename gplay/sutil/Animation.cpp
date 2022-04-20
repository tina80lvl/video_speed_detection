//
// Created by dipet on 13.04.2020.
//

#include "Animation.h"

std::vector<int> sutil::Animation::getNodes() const
{
    std::vector<int> keys;
    for(auto& item : translations_) {
        keys.emplace_back(item.first);
    }
    for(auto& item : scales_) {
        keys.emplace_back(item.first);
    }
    for(auto& item : rotations_) {
        keys.emplace_back(item.first);
    }
    return keys;
}

sutil::Matrix4x4 sutil::Animation::computeLinear(float t, float t1, float t2, const sutil::Matrix4x4 &v1, const sutil::Matrix4x4 &v2)
{
    // y1 = a * x1 + b;
    // b = y1 - a * x1
    // y2 = a * x2 + y1 - a * x1
    // y2 - y1 = a * (x2 - x1)
    // a = (y2 - y1) / (x2 - x1)
    auto a = (v2 - v1) / (t2 - t1);
    auto b = v1 - a * t1;
    return a * t + b;
}

sutil::Matrix4x4 sutil::Animation::getMatrix(
    int node,
    float time,
    const std::unordered_map<int, sutil::Animation::AnimationChannel> &channels
    )
{
    auto it = channels.find(node);
    if(it == channels.end()) {
        return Matrix4x4::identity();
    }

    const auto& channel = it->second;

    size_t prev, next;
    float prev_time = std::nan("");
    float next_time = std::nan("");
    // TODO ускорить
    for(size_t i = 0; i < channel.time_points.size(); ++i) {
        next_time = channel.time_points.get<float>(i);
        next = i;
        if(next_time <= time) {
            prev_time = next_time;
            prev = i;
            continue;
        }
        if(next_time >= time) {
            break;
        }
    }
    if(next == 0) {
        prev = 0;
        prev_time = next_time;
    }

    Matrix4x4 v1 = Matrix4x4::identity();
    Matrix4x4 v2 = Matrix4x4::identity();
    if(channel.target == "translation" || channel.target == "scale") {
        v1 = Matrix4x4::translate(channel.values.get<float3>(prev));
        v2 = Matrix4x4::translate(channel.values.get<float3>(next));
    } else if(channel.target == "scale") {
        auto val = channel.values.get<float4>(prev);
        v1 = Quaternion(val.w, val.x, val.y, val.z).rotationMatrix();
        val = channel.values.get<float4>(next);
        v2 = Quaternion(val.w, val.x, val.y, val.z).rotationMatrix();
    } else {
        LOGC(ERROR) << "Animation does not support target " << channel.target << std::endl;
        exit(1);
    }

    return computeLinear(time, prev_time, next_time, v1, v2);
}
